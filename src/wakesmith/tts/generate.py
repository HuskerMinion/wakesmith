"""Synthetic speech clips from Piper's multi-speaker VITS generator models.

Ported from rhasspy/piper-sample-generator (MIT), with:
- an explicit torch device (Intel XPU, CUDA, MPS or CPU) instead of hard-coded CUDA;
- a restricted loader: the checkpoint is a pickled torch module, so only the classes it needs may be
  created while loading, and its checksum is verified at download (see assets.py);
- random speaker pairs and voice settings per batch (upstream cycles speakers in order, so the first
  thousand clips all share speaker 0);
- NaN/inf phoneme durations clamped, and a failed batch retried at half size instead of dropped;
- output resampled to 16 kHz on the device, written as clip_NNNNNN.wav; generation resumes.

Run one process per GPU to use several cards (see pipeline.py).
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import pickle
import random
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import phonemize
from .vits import commons

logger = logging.getLogger(__name__)
OUT_RATE = 16000
HOP = 256  # VITS samples per phoneme frame


# --- restricted checkpoint loading --------------------------------------------------------------

_ALLOWED_PREFIXES = (
    "torch.",
    "collections.OrderedDict",
    "builtins.set",
    "builtins.getattr",
    "__builtin__.set",
    "__builtin__.getattr",
)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):  # type: ignore[override]
        if module.startswith("piper_train.vits."):
            module = "wakesmith.tts.vits." + module.removeprefix("piper_train.vits.")
            return getattr(importlib.import_module(module), name)
        full = f"{module}.{name}"
        if full.startswith(_ALLOWED_PREFIXES) or module.startswith("torch."):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to load {full} from a TTS checkpoint")


class _PickleModule:
    Unpickler = _Unpickler
    load = pickle.load
    __name__ = "wakesmith_restricted_pickle"


def load_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    model = torch.load(path, map_location="cpu", weights_only=False, pickle_module=_PickleModule)
    model.eval().to(device)
    config = json.loads(Path(f"{path}.json").read_text())
    return model, config


# --- synthesis ----------------------------------------------------------------------------------


def _slerp(v1: torch.Tensor, v2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Spherical interpolation per row (speaker embeddings), t: (batch, 1)."""
    n1, n2 = v1 / v1.norm(dim=-1, keepdim=True), v2 / v2.norm(dim=-1, keepdim=True)
    dot = (n1 * n2).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    sin = torch.sin(theta)
    lerp = sin.abs() < 1e-4
    s1 = torch.where(lerp, 1 - t, torch.sin((1 - t) * theta) / sin.clamp_min(1e-8))
    s2 = torch.where(lerp, t, torch.sin(t * theta) / sin.clamp_min(1e-8))
    return s1 * v1 + s2 * v2


@torch.no_grad()
def _synthesize(
    model,
    ids: list[list[int]],
    spk1,
    spk2,
    slerp_t,
    length_scale: float,
    noise_scale: float,
    noise_w: float,
    device: torch.device,
) -> list[np.ndarray]:
    lens = torch.tensor([len(i) for i in ids], device=device)
    pad = max(len(i) for i in ids)
    x = torch.tensor([i + [1] * (pad - len(i)) for i in ids], device=device)  # pad with '^' id 1
    x, m_p, logs_p, x_mask = model.enc_p(x, lens)
    g = _slerp(model.emb_g(spk1), model.emb_g(spk2), slerp_t).unsqueeze(-1)
    if model.use_sdp:
        logw = model.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_w)
    else:
        logw = model.dp(x, x_mask, g=g)
    # The stochastic duration predictor occasionally returns NaN/inf; cap each phoneme at ~0.6 s.
    logw = torch.nan_to_num(logw, nan=0.0, posinf=4.0, neginf=-4.0).clamp(-4.0, 4.0)
    w_ceil = torch.ceil(torch.exp(logw) * x_mask * length_scale)
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_mask = commons.sequence_mask(y_lengths, int(y_lengths.max())).unsqueeze(1).to(x_mask.dtype)
    attn = commons.generate_path(w_ceil, x_mask.unsqueeze(2) * y_mask.unsqueeze(-1))
    m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
    logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
    z = model.flow(z_p, y_mask, g=g, reverse=True)
    audio = model.dec(z * y_mask, g=g)[:, 0]  # (batch, samples) at the model rate
    ends = (w_ceil.sum([1, 2]) * HOP).long().tolist()
    return [audio[i, : ends[i]] for i in range(len(ids))]


@dataclass
class VoiceSettings:
    length_scales: Sequence[float] = (0.75, 0.9, 1.0, 1.15, 1.3)
    noise_scales: Sequence[float] = (0.5, 0.667, 0.8)
    noise_ws: Sequence[float] = (0.6, 0.8, 1.0)
    slerp_range: tuple[float, float] = (0.2, 0.8)


class Generator:
    def __init__(
        self, model_path: Path, device: str | torch.device = "cpu", seed: int | None = None
    ):
        self.device = torch.device(device)
        self.model, self.config = load_model(Path(model_path), self.device)
        self.voice = self.config["espeak"]["voice"]
        self.rate = self.config["audio"]["sample_rate"]
        self.n_speakers = self.config["num_speakers"]
        self.id_map = self.config["phoneme_id_map"]
        self.rng = random.Random(seed)
        if seed is not None:
            torch.manual_seed(seed)
        import torchaudio.functional as AF

        self._resample = lambda a: AF.resample(a, self.rate, OUT_RATE)

    def batch(
        self, texts: Sequence[str], settings: VoiceSettings | None = None
    ) -> list[np.ndarray]:
        """One batch, shared voice settings, random speaker pair per clip. float32 at 16 kHz."""
        settings = settings or VoiceSettings()
        r = self.rng
        ids = [phonemize.phoneme_ids(t, self.id_map, self.voice) for t in texts]
        n = len(ids)
        spk1 = torch.tensor([r.randrange(self.n_speakers) for _ in range(n)], device=self.device)
        spk2 = torch.tensor([r.randrange(self.n_speakers) for _ in range(n)], device=self.device)
        t = torch.tensor([[r.uniform(*settings.slerp_range)] for _ in range(n)], device=self.device)
        clips = _synthesize(
            self.model,
            ids,
            spk1,
            spk2,
            t,
            r.choice(settings.length_scales),
            r.choice(settings.noise_scales),
            r.choice(settings.noise_ws),
            self.device,
        )
        out = []
        for c in clips:
            a = self._resample(c.float().unsqueeze(0))[0]
            a = a / max(float(a.abs().max()), 0.01) * 0.95  # peak-normalize like piper
            out.append(_trim(a.cpu().numpy()))
        return out


def _trim(a: np.ndarray, thr: float = 1e-3) -> np.ndarray:
    nz = np.flatnonzero(np.abs(a) > thr)
    return a[nz[0] : nz[-1] + 1] if len(nz) else a


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(OUT_RATE)
        w.writeframes(pcm.tobytes())


def count_clips(out_dir: Path) -> int:
    return len(list(out_dir.glob("clip_[0-9][0-9][0-9][0-9][0-9][0-9].wav")))


def generate(
    phrases: Sequence[str],
    out_dir: Path,
    n: int,
    model_path: Path,
    device: str = "cpu",
    batch_size: int = 64,
    seed: int | None = None,
    settings: VoiceSettings | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Write `n` clips of `phrases` (cycled at random) to out_dir/clip_NNNNNN.wav, resuming.

    Returns {"written": .., "failed": ..}. A batch that raises (e.g. the occasional XPU kernel error)
    is retried at half size down to 1 clip; only clips that fail alone are skipped."""
    out_dir.mkdir(parents=True, exist_ok=True)
    done = count_clips(out_dir)
    if done >= n:
        return {"written": 0, "failed": 0}
    gen = Generator(model_path, device, seed)
    rng = random.Random(seed)
    idx, failed = done, 0

    def run(texts: list[str]) -> None:
        nonlocal idx, failed
        try:
            clips = gen.batch(texts, settings)
        except RuntimeError as e:
            if len(texts) == 1:
                failed += 1
                logger.warning("TTS failed for %r: %s", texts[0], e)
                return
            half = len(texts) // 2
            run(texts[:half])
            run(texts[half:])
            return
        for c in clips:
            if idx >= n:
                break
            write_wav(out_dir / f"clip_{idx:06d}.wav", c)
            idx += 1

    while idx < n:
        texts = [rng.choice(phrases) for _ in range(min(batch_size, n - idx))]
        run(texts)
        if on_progress:
            on_progress(idx, n)
    if gen.device.type == "xpu":
        torch.xpu.empty_cache()
    return {"written": idx - done, "failed": failed}


def batches_needed(n: int, batch_size: int) -> int:
    return math.ceil(n / batch_size)
