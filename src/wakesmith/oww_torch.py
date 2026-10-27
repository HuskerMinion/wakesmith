"""openWakeWord's frontend (melspectrogram + Google's speech embedding) in PyTorch, for a GPU.

The two ONNX models are small and fixed, so they are written out here and their weights read straight from
the ONNX files (no `onnx` package: a few lines of protobuf decoding find the initializers):
- melspectrogram.onnx: |STFT|^2 as two Conv1d (512 taps, hop 160, no padding), a 257x32 mel matrix,
  10*log10(max(x, 1e-10)), then everything below (max - 80 dB) clipped up. ONNX takes that max over the
  whole input tensor; wakesmith feeds it one clip at a time, so here it is per clip.
- embedding_model.onnx: 20 Conv2d (batch norm fused in), each but the last followed by LeakyReLU(0.2) and
  max(x, -0.4), with 5 max-pools; a (76, 32) mel window -> 96 values.

Same features as the ONNX models on the CPU within float32 rounding (see tests). TF32 is turned off while
it runs: on NVIDIA cards cuDNN would otherwise round convolution inputs to 10-bit mantissas.
"""

from __future__ import annotations

import contextlib
import math
from pathlib import Path

import numpy as np
import torch
import torch._dynamo  # must load before TensorFlow does: imported after it, it segfaults
import torch.nn.functional as F

from . import assets

N_STEPS, WINDOW, STRIDE = 16, 76, 8

# --- ONNX initializers ---------------------------------------------------------------------------

_DTYPES = {1: np.float32, 7: np.int64}


def _varint(b: bytes, i: int) -> tuple[int, int]:
    v = s = 0
    while True:
        c = b[i]
        i += 1
        v |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return v, i


def _fields(b: bytes):
    """(field number, wire type, value) of a protobuf message; value is bytes for length-delimited."""
    i = 0
    while i < len(b):
        key, i = _varint(b, i)
        num, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 1:
            v, i = b[i : i + 8], i + 8
        elif wt == 2:
            n, i = _varint(b, i)
            v, i = b[i : i + n], i + n
        elif wt == 5:
            v, i = b[i : i + 4], i + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wt}")
        yield num, wt, v


def onnx_initializers(path: Path) -> dict[str, np.ndarray]:
    """name -> array for an ONNX model's initializers stored as raw data (ModelProto.graph = 7,
    GraphProto.initializer = 5; TensorProto dims = 1, data_type = 2, name = 8, raw_data = 9)."""
    out = {}
    for num, _, graph in _fields(Path(path).read_bytes()):
        if num != 7:
            continue
        for gnum, _, t in _fields(graph):
            if gnum != 5:
                continue
            dims, dtype, name, raw = [], None, "", None
            for tnum, wt, v in _fields(t):
                if tnum == 1:
                    if wt == 2:  # packed
                        j = 0
                        while j < len(v):
                            d, j = _varint(v, j)
                            dims.append(d)
                    else:
                        dims.append(v)
                elif tnum == 2:
                    dtype = v
                elif tnum == 8:
                    name = v.decode()
                elif tnum == 9:
                    raw = v
            if raw is not None and dtype in _DTYPES:
                out[name] = np.frombuffer(raw, _DTYPES[dtype]).reshape(dims).copy()
    return out


# --- the models -----------------------------------------------------------------------------------


class Frontend:
    """openWakeWord features on a torch device: Frontend("xpu:0").features(clips) as oww.Frontend's."""

    def __init__(self, device: str = "cpu", batch: int = 64):
        self.device = torch.device(device)
        self.batch = batch  # clips per call: ~1 GB of GPU memory at 64 (128: 2 GB, no faster)
        dev = lambda a: torch.from_numpy(a).to(self.device)
        mel = onnx_initializers(assets.fetch(assets.ASSETS["oww-melspectrogram"], progress=False))
        self.stft = dev(
            np.concatenate([mel["0.stft.conv_real.weight"], mel["0.stft.conv_imag.weight"]])
        )
        self.melw = dev(mel["1.melW"])
        emb = onnx_initializers(assets.fetch(assets.ASSETS["oww-embedding"], progress=False))
        self.convs = []  # (weight, bias or None, pad width by 1, max-pool after)
        pools = {2: (2, 2), 6: (1, 2), 10: (2, 2), 14: (1, 2), 18: (2, 2)}
        for i in range(20):
            pre = f"model/conv2d{'' if i == 0 else f'_{i}'}/Conv2D"
            if i == 19:
                w, b = emb[f"{pre}/ReadVariableOp:0"], None
            else:
                w, b = emb[f"{pre}_weights_fused_bn"], emb[f"{pre}_bias_fused_bn"]
            self.convs.append(
                (dev(w), None if b is None else dev(b), w.shape[3] == 3, pools.get(i))
            )

    @contextlib.contextmanager
    def _exact(self):
        """No TF32 on CUDA (it changes the features); restored afterwards."""
        cudnn, mm = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
        try:
            with torch.inference_mode():
                yield
        finally:
            torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = cudnn, mm

    def melspec(self, audio: torch.Tensor) -> torch.Tensor:
        """(B, samples) int16-valued float32 -> (B, frames, 32), scaled as openWakeWord does (x/10 + 2)."""
        spec = F.conv1d(audio[:, None, :], self.stft, stride=160)  # (B, 514, frames)
        re, im = spec[:, :257], spec[:, 257:]
        power = (re * re + im * im).transpose(1, 2)  # (B, frames, 257)
        # 10*log10 as the ONNX graph computes it
        db = torch.log(torch.clamp(power @ self.melw, min=1e-10)) * 10.0 / 2.3025851
        top = db.amax(dim=(1, 2), keepdim=True)
        return torch.maximum(db, top - 80.0) / 10.0 + 2.0

    def embed(self, windows: torch.Tensor) -> torch.Tensor:
        """(n, 76, 32) mel windows -> (n, 96)."""
        x = windows[:, None]
        for w, b, pad, pool in self.convs:
            if pad:
                x = F.pad(x, (1, 1))
            x = F.conv2d(x, w, b)
            if b is not None:
                x = torch.clamp_min(F.leaky_relu(x, 0.2), -0.4)
            if pool:
                x = F.max_pool2d(x, pool, pool)
        return x.reshape(len(x), 96)

    def _same_length(self, audio: np.ndarray) -> np.ndarray:
        """(B, samples) -> (B, 16, 96): the last 16 embeddings of each clip, zero-padded at the front."""
        out = np.zeros((len(audio), N_STEPS, 96), np.float32)
        frames = (audio.shape[1] - 512) // 160 + 1
        count = max(0, (frames - WINDOW) // STRIDE + 1)
        n = min(count, N_STEPS)
        if not n:
            return out
        for s in range(0, len(audio), self.batch):
            x = torch.from_numpy(np.ascontiguousarray(audio[s : s + self.batch], np.float32))
            mel = self.melspec(x.to(self.device))  # (b, frames, 32)
            starts = range((count - n) * STRIDE, count * STRIDE, STRIDE)
            win = torch.stack([mel[:, t : t + WINDOW] for t in starts], 1)  # (b, n, 76, 32)
            emb = self.embed(win.reshape(-1, WINDOW, 32)).reshape(len(x), n, 96)
            out[s : s + len(x), N_STEPS - n :] = emb.float().cpu().numpy()
        return out

    def features(self, clips: list[np.ndarray] | np.ndarray) -> np.ndarray:
        """Clips (int16-valued float samples) -> (len(clips), 16, 96), as oww.Frontend.features."""
        out = np.zeros((len(clips), N_STEPS, 96), np.float32)
        by_len: dict[int, list[int]] = {}
        for i, c in enumerate(clips):
            by_len.setdefault(len(c), []).append(i)
        with self._exact():
            for idx in by_len.values():
                out[idx] = self._same_length(np.stack([clips[i] for i in idx]))
        return out


# --- the DNN head's training ----------------------------------------------------------------------


class HeadTrainer:
    """oww.train's model and optimizer in PyTorch, for a GPU (oww.train runs the loop).

    The model is oww.build_model's: Flatten, (Dense, LayerNorm(eps 1e-5), ReLU) x (blocks + 1), Dense(1),
    sigmoid. Its weights are kept in Keras' layout and order (kernels (in, out), then bias; gamma, beta),
    so they move between this and the Keras model with get_weights/set_weights unchanged, starting from
    the Keras model's own seeded initialisation. They live in one flat tensor (the layers use views of
    it), so the optimizer is a handful of device calls per step instead of a handful per weight.

    The optimizer is Keras 3's AdamW as it computes: every variable decays first (v -= v * wd * lr),
    then Adam with Keras' epsilon placement (lr * sqrt(1 - b2^t) / (1 - b1^t) * m / (sqrt(v) + eps));
    torch's AdamW puts epsilon after the bias correction. new_optimizer() starts it over, as each of
    oww.train's phases makes a new Keras optimizer."""

    def __init__(self, weights: list[np.ndarray], device: str = "cpu", weight_decay: float = 1e-2):
        self.device = torch.device(device)
        self.shapes = [tuple(w.shape) for w in weights]
        self.sizes = [int(np.prod(sh)) for sh in self.shapes]
        flat = np.concatenate([np.asarray(w, np.float32).ravel() for w in weights])
        self.flat = torch.from_numpy(flat).to(self.device).requires_grad_(True)
        self.wd, self.b1, self.b2, self.eps = weight_decay, 0.9, 0.999, 1e-7
        self.new_optimizer()

    def new_optimizer(self) -> None:
        self.t = 0
        self.m = torch.zeros_like(self.flat)
        self.v = torch.zeros_like(self.flat)

    def _views(self) -> list[torch.Tensor]:
        return [
            t.view(sh)
            for t, sh in zip(torch.split(self.flat, self.sizes), self.shapes, strict=True)
        ]

    def get_weights(self) -> list[np.ndarray]:
        flat = self.flat.detach().cpu().numpy()
        return [
            w.reshape(sh).copy()
            for w, sh in zip(np.split(flat, np.cumsum(self.sizes)[:-1]), self.shapes)
        ]

    def set_weights(self, weights: list[np.ndarray]) -> None:
        flat = np.concatenate([np.asarray(w, np.float32).ravel() for w in weights])
        with torch.no_grad():
            self.flat.copy_(torch.from_numpy(flat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 16, 96) -> (B,) scores in (0, 1)."""
        h = x.reshape(len(x), -1)
        ps = self._views()
        for i in range(0, len(ps) - 2, 4):
            h = torch.addmm(ps[i + 1], h, ps[i])
            h = torch.relu(F.layer_norm(h, h.shape[-1:], ps[i + 2], ps[i + 3], eps=1e-5))
        return torch.sigmoid(torch.addmm(ps[-1], h, ps[-2]))[:, 0]

    def predict(self, x: np.ndarray | torch.Tensor, batch: int = 4096) -> np.ndarray:
        out = []
        with torch.no_grad():
            for s in range(0, len(x), batch):
                xb = x[s : s + batch]
                if isinstance(xb, np.ndarray):
                    xb = torch.from_numpy(np.ascontiguousarray(xb))
                out.append(self.forward(xb.to(self.device).float()).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def step(self, x, y, lam, perm, smooth, gamma, neg_w, lr) -> torch.Tensor:
        """One training step on a raw batch (x (B, 16, 96) float16 or float32, y (B,) numpy): mixup with
        `lam` and `perm`, label smoothing, focal loss weighted by neg_w on negatives, AdamW at `lr`.
        Returns the loss (a device tensor: reading it waits for the device)."""
        x = torch.from_numpy(x).to(self.device, non_blocking=True).float()
        y = torch.from_numpy(y).to(self.device, non_blocking=True)
        idx = torch.from_numpy(perm).to(self.device, non_blocking=True)
        x = lam * x + (1 - lam) * x[idx]
        y = lam * y + (1 - lam) * y[idx]
        y = y * (1 - smooth) + 0.5 * smooth
        pred = torch.clamp(self.forward(x), 1e-7, 1 - 1e-7)
        bce = -(y * torch.log(pred) + (1 - y) * torch.log(1 - pred))
        p_t = y * pred + (1 - y) * (1 - pred)
        w = torch.where(y < 0.5, neg_w, 1.0)
        loss = torch.mean(bce * torch.pow(1 - p_t, gamma) * w)
        (g,) = torch.autograd.grad(loss, self.flat)
        self.t += 1
        alpha = lr * math.sqrt(1 - self.b2**self.t) / (1 - self.b1**self.t)
        with torch.no_grad():
            self.flat.mul_(1 - self.wd * lr)
            self.m.lerp_(g, 1 - self.b1)
            self.v.lerp_(g * g, 1 - self.b2)
            self.flat.addcdiv_(self.m, self.v.sqrt().add_(self.eps), value=-alpha)
        return loss.detach()
