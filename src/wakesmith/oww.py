"""openWakeWord-compatible model: features through openWakeWord's frozen frontend, a small DNN head
trained in Keras, exported natively to .tflite ([1,16,96] float32 -> [1,1]).

Features: every clip is augmented `ROUNDS` times in memory (2 s, phrase at the end, noise, reverb)
and turned into the last 16 speech embeddings (melspectrogram.onnx -> embedding_model.onnx), on
every CPU core. Nothing augmented is written to disk.

Training follows the usual openWakeWord-style recipe: mixed batches (positives, synthetic and real
negatives, noise, 2,000 h of general audio from ACAV100M), focal loss with mixup and label smoothing,
a negative-class weight ramped up over training, three phases at falling learning rates, and an
average of the best late checkpoints.
"""

from __future__ import annotations

import json
import math
import os
import random
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import soundfile as sf

from . import assets
from .project import Project

SR = 16000
CLIP_S = 2.0
ROUNDS = 3
N_STEPS = 16  # embeddings per example
WINDOW, STRIDE = 76, 8  # mel frames per embedding, frames between embeddings


# --- frontend ------------------------------------------------------------------------------------


def pcm_scale(a: np.ndarray) -> np.ndarray:
    """float audio in [-1, 1] -> int16-valued float32. openWakeWord feeds its mel model raw int16
    sample values (not normalized), and the ACAV100M negative features were made that way, so
    training features must use the same scale or they won't match the runtime."""
    return np.clip(np.round(a * 32767.0), -32768, 32767).astype(np.float32)


class Frontend:
    """openWakeWord's melspectrogram + speech-embedding ONNX models."""

    def __init__(self, threads: int = 1, providers: list | None = None):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        prov = providers or ["CPUExecutionProvider"]
        mel = assets.fetch(assets.ASSETS["oww-melspectrogram"], progress=False)
        emb = assets.fetch(assets.ASSETS["oww-embedding"], progress=False)
        self.mel = ort.InferenceSession(str(mel), opts, providers=prov)
        self.emb = ort.InferenceSession(str(emb), opts, providers=prov)
        self._mel_in = self.mel.get_inputs()[0].name
        self._emb_in = self.emb.get_inputs()[0].name

    def melspec(self, audio: np.ndarray) -> np.ndarray:
        """(samples,) int16-valued float32 -> (frames, 32), scaled as openWakeWord does (x/10 + 2)."""
        out = self.mel.run(None, {self._mel_in: audio[None, :].astype(np.float32)})[0]
        return np.squeeze(out) / 10.0 + 2.0

    def embed_windows(self, windows: np.ndarray) -> np.ndarray:
        """(n, 76, 32) mel windows -> (n, 96) embeddings."""
        if not len(windows):
            return np.zeros((0, 96), np.float32)
        out = self.emb.run(None, {self._emb_in: windows[..., None].astype(np.float32)})[0]
        return out.reshape(len(windows), 96)

    def last_windows(self, audio: np.ndarray, n: int = N_STEPS) -> np.ndarray:
        """Mel windows for the last `n` embeddings of a clip (left-padded with zeros later)."""
        mel = self.melspec(audio)
        count = max(0, (mel.shape[0] - WINDOW) // STRIDE + 1)
        first = max(0, count - n)
        return (
            np.stack([mel[i * STRIDE : i * STRIDE + WINDOW] for i in range(first, count)])
            if count
            else np.zeros((0, WINDOW, 32), np.float32)
        )

    def features(self, clips: list[np.ndarray], chunk: int = 128) -> np.ndarray:
        """Clips -> (len(clips), 16, 96). The windows of `chunk` clips go through the embedding
        model in one call (bounded: the model's activations grow with the batch)."""
        if len(clips) > chunk:
            return np.concatenate(
                [self.features(clips[i : i + chunk], chunk) for i in range(0, len(clips), chunk)]
            )
        sets = [self.last_windows(c) for c in clips]
        emb = self.embed_windows(np.concatenate(sets)) if sets else np.zeros((0, 96))
        out = np.zeros((len(clips), N_STEPS, 96), np.float32)
        k = 0
        for i, s in enumerate(sets):
            out[i, N_STEPS - len(s) :] = emb[k : k + len(s)]
            k += len(s)
        return out


# --- augmentation + feature extraction ----------------------------------------------------------

_W: dict = {}


def _init_worker(noise_dir: str, rir_dir: str) -> None:
    import warnings

    warnings.filterwarnings("ignore")
    from microwakeword.audio.augmentation import Augmentation

    seed = int.from_bytes(os.urandom(4), "little")
    random.seed(seed)
    np.random.seed(seed)
    _W["aug"] = Augmentation(
        augmentation_duration_s=CLIP_S,
        augmentation_probabilities={
            "SevenBandParametricEQ": 0.25,
            "TanhDistortion": 0.25,
            "PitchShift": 0.15,
            "BandStopFilter": 0.1,
            "AddColorNoise": 0.1,
            "AddBackgroundNoise": 0.75,
            "Gain": 1.0,
            "RIR": 0.5,
        },
        impulse_paths=[rir_dir],
        background_paths=[noise_dir],
        background_min_snr_db=0,
        background_max_snr_db=15,
        min_jitter_s=0.0,
        max_jitter_s=0.2,
    )
    _W["fe"] = Frontend(threads=1)


def _chunk(args: tuple[list[str], int]) -> np.ndarray:
    files, rounds = args
    clips = []
    for f in files:
        a, sr = sf.read(f, dtype="float32", always_2d=True)
        if sr != SR:
            raise ValueError(f"{f}: {sr} Hz (want 16 kHz)")
        for _ in range(rounds):
            clips.append(pcm_scale(_W["aug"].augment_clip(a[:, 0])))
    return _W["fe"].features(clips)


def extract(files: list[str], rounds: int, workers: int, desc: str) -> np.ndarray:
    from tqdm import tqdm

    if not files:
        return np.zeros((0, N_STEPS, 96), np.float16)
    noise = str(assets.fetch(assets.ASSETS["musan-noise"]))
    rirs = str(assets.fetch(assets.ASSETS["mit-rirs"]))
    chunks = [files[i : i + 64] for i in range(0, len(files), 64)]
    out = []
    with get_context("fork").Pool(
        workers, initializer=_init_worker, initargs=(noise, rirs)
    ) as pool:
        for feats in tqdm(
            pool.imap(_chunk, [(c, rounds) for c in chunks]), total=len(chunks), desc=desc
        ):
            out.append(feats.astype(np.float16))
    return np.concatenate(out)


def _noise_clips(n: int, seed: int = 0) -> list[np.ndarray]:
    """Random 2 s stretches of background noise, as pure-noise negatives."""
    files = sorted(Path(assets.fetch(assets.ASSETS["musan-noise"])).rglob("*.wav"))
    rng = np.random.default_rng(seed)
    out = []
    for f in rng.choice(files, size=n):
        info = sf.info(str(f))
        need = int(CLIP_S * SR)
        start = int(rng.integers(0, max(1, info.frames - need)))
        a, _ = sf.read(str(f), start=start, frames=need, dtype="float32", always_2d=True)
        out.append(pcm_scale(np.pad(a[:, 0], (0, max(0, need - len(a))))))
    return out


def build_features(p: Project, workers: int, log=print) -> None:
    d = p.path("oww", "features")
    d.mkdir(parents=True, exist_ok=True)

    def originals(split: str) -> list[str]:
        return sorted(str(x) for x in p.path("clips", split).glob("clip_[0-9]*.wav"))

    sets = {
        "positive_train": (
            originals("positive_train")
            + sorted(str(x) for x in p.path("real", "train").glob("*.wav")) * p.real.copies,
            ROUNDS,
        ),
        "negative_train": (
            originals("negative_train")
            + sorted(str(x) for x in p.path("real_neg", "train").glob("*.wav"))
            * p.real.negative_copies,
            ROUNDS,
        ),
        "positive_test": (originals("positive_test"), 1),
        "negative_test": (originals("negative_test"), 1),
    }
    for name, (files, rounds) in sets.items():
        log(f"oww features {name}: {len(files)} clips x{rounds}")
        np.save(d / f"{name}.npy", extract(files, rounds, workers, f"oww {name}"))
    fe = Frontend(threads=max(1, workers))
    noise = _noise_clips(3000)
    np.save(d / "noise_train.npy", fe.features(noise[:2500]).astype(np.float16))
    np.save(d / "noise_test.npy", fe.features(noise[2500:]).astype(np.float16))


# --- training ------------------------------------------------------------------------------------


def build_model(layer_dim: int, blocks: int):
    import tensorflow as tf

    x = inp = tf.keras.Input((N_STEPS, 96), batch_size=None, name="embeddings")
    x = tf.keras.layers.Flatten()(x)
    for _ in range(blocks + 1):
        x = tf.keras.layers.Dense(layer_dim)(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
        x = tf.keras.layers.ReLU()(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="score")(x)
    return tf.keras.Model(inp, out)


class _Batches:
    """Mixed batches drawn from in-memory sets plus contiguous blocks of the ACAV100M memmap."""

    def __init__(self, p: Project, seed: int = 0):
        d = p.path("oww", "features")
        self.rng = np.random.default_rng(seed)
        self.sets = {
            k: np.load(d / f"{k}.npy") for k in ("positive_train", "negative_train", "noise_train")
        }
        self.acav = np.load(assets.fetch(assets.ASSETS["oww-acav-features"]), mmap_mode="r")
        self.n = {"positive_train": 50, "negative_train": 50, "noise_train": 50, "acav": 1024}

    def __call__(self) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for k, n in self.n.items():
            if k == "acav":
                blocks = []
                for _ in range(16):  # 16 random contiguous blocks: random enough, fast to read
                    s = int(self.rng.integers(0, len(self.acav) - n // 16))
                    blocks.append(np.asarray(self.acav[s : s + n // 16]))
                x = np.concatenate(blocks)
            else:
                arr = self.sets[k]
                x = arr[self.rng.integers(0, len(arr), n)]
            xs.append(x.astype(np.float32))
            ys.append(np.full(len(x), 1.0 if k == "positive_train" else 0.0, np.float32))
        return np.concatenate(xs), np.concatenate(ys)


def _validation(p: Project) -> dict[str, np.ndarray]:
    d = p.path("oww", "features")
    val = np.load(assets.fetch(assets.ASSETS["oww-validation-features"]))
    if val.ndim == 2:  # (N, 96) continuous embeddings -> non-overlapping 16-step windows
        val = val[: len(val) // N_STEPS * N_STEPS].reshape(-1, N_STEPS, 96)
    neg = np.concatenate([np.load(d / "negative_test.npy"), np.load(d / "noise_test.npy")])
    return {
        "pos": np.load(d / "positive_test.npy").astype(np.float32),
        "neg": neg.astype(np.float32),
        "val": val.astype(np.float32),
    }


def _metrics(model, v: dict[str, np.ndarray], threshold: float = 0.5) -> dict[str, float]:
    pred = {k: model.predict(x, batch_size=4096, verbose=0)[:, 0] for k, x in v.items()}
    recall = float((pred["pos"] >= threshold).mean())
    hours = len(v["val"]) * CLIP_S / 3600
    fpph = float((pred["val"] >= threshold).sum() / hours)
    acc = float(
        ((pred["pos"] >= threshold).sum() + (pred["neg"] < threshold).sum())
        / (len(pred["pos"]) + len(pred["neg"]))
    )
    return {"recall": recall, "fpph": fpph, "accuracy": acc}


def train(p: Project, log=print) -> Path:
    import tensorflow as tf

    cfg = p.openwakeword
    tf.keras.utils.set_random_seed(0)
    model = build_model(cfg.layer_dim, cfg.blocks)
    batches = _Batches(p)
    val = _validation(p)
    checkpoints: list[tuple[dict[str, float], list[np.ndarray]]] = []
    history = []
    base_lr, wd, smooth, gamma, alpha = 1e-4, 1e-2, 0.05, 2.0, 0.2

    def focal(y, pred):
        pred = tf.clip_by_value(pred, 1e-7, 1 - 1e-7)
        bce = -(y * tf.math.log(pred) + (1 - y) * tf.math.log(1 - pred))
        p_t = y * pred + (1 - y) * (1 - pred)
        return bce * tf.pow(1 - p_t, gamma)

    def run_phase(phase: int, steps: int, lr0: float, max_w: float) -> None:
        opt = tf.keras.optimizers.AdamW(learning_rate=lr0, weight_decay=wd)
        warm, hold = (steps // 5, steps // 3) if phase == 1 else (0, 0)

        @tf.function
        def step_fn(x, y, neg_w, lr):
            opt.learning_rate.assign(lr)
            with tf.GradientTape() as tape:
                pred = model(x, training=True)[:, 0]
                w = tf.where(y < 0.5, neg_w, 1.0)
                loss = tf.reduce_mean(focal(y, pred) * w)
            opt.apply_gradients(
                zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables)
            )
            return loss

        every = max(1, steps // 20)
        for s in range(steps):
            x, y = batches()
            lam = np.float32(np.random.beta(alpha, alpha))
            perm = np.random.permutation(len(x))
            x = lam * x + (1 - lam) * x[perm]
            y = lam * y + (1 - lam) * y[perm]
            y = y * (1 - smooth) + 0.5 * smooth
            if s < warm:
                lr = lr0 * s / max(1, warm)
            elif s < warm + hold:
                lr = lr0
            else:
                prog = (s - warm - hold) / max(1, steps - warm - hold)
                lr = lr0 * 0.5 * (1 + math.cos(math.pi * prog))
            neg_w = 1.0 + (max_w - 1.0) * s / max(1, steps)
            loss = step_fn(x, y, np.float32(neg_w), np.float32(lr))
            if s >= steps * 3 // 4 and s % every == 0:
                m = _metrics(model, val)
                checkpoints.append((m, [w.copy() for w in model.get_weights()]))
                history.append({"phase": phase, "step": s, "loss": float(loss), **m})
                log(
                    f"oww phase {phase} step {s}/{steps}: loss {float(loss):.4f} "
                    f"recall {m['recall']:.3f} fpph {m['fpph']:.2f}"
                )

    max_w = cfg.max_negative_weight
    run_phase(1, cfg.steps, base_lr, max_w)
    for phase, lr in ((2, base_lr * 0.1), (3, base_lr * 0.01)):
        if _metrics(model, val)["fpph"] > cfg.target_fp_per_hour:
            max_w *= 2
            log(f"oww: false-trigger rate above target, negative weight now {max_w:.0f}")
        run_phase(phase, cfg.steps // 10, lr, max_w)

    # Average the late checkpoints that are good on every metric at once.
    fp = np.percentile([m["fpph"] for m, _ in checkpoints], 10)
    rc = np.percentile([m["recall"] for m, _ in checkpoints], 90)
    ac = np.percentile([m["accuracy"] for m, _ in checkpoints], 90)
    chosen = [
        w for m, w in checkpoints if m["fpph"] <= fp and m["recall"] >= rc and m["accuracy"] >= ac
    ] or [max(checkpoints, key=lambda c: c[0]["recall"])[1]]
    model.set_weights([np.mean(ws, axis=0) for ws in zip(*chosen)])
    final = _metrics(model, val)
    log(
        f"oww: averaged {len(chosen)} checkpoints: recall {final['recall']:.3f} "
        f"fpph {final['fpph']:.2f}"
    )
    out = p.path("oww", "model")
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "model.keras")
    (out / "history.json").write_text(json.dumps({"history": history, "final": final}, indent=1))
    return export_tflite(model, out / "model.tflite")


def export_tflite(model, path: Path) -> Path:
    """Float32 .tflite with a fixed [1, 16, 96] input, as the openWakeWord runtime feeds it."""
    import tensorflow as tf

    inp = tf.keras.Input((N_STEPS, 96), batch_size=1, name="embeddings")
    fixed = tf.keras.Model(inp, model(inp))
    path.write_bytes(tf.lite.TFLiteConverter.from_keras_model(fixed).convert())
    return path


# --- scoring -------------------------------------------------------------------------------------


class Scorer:
    """Score a recording like a live listener: a 2 s window every 80 ms, peak score."""

    def __init__(self, tflite: Path):
        from ai_edge_litert.interpreter import Interpreter

        self.it = Interpreter(model_path=str(tflite))
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]["index"]
        self.out = self.it.get_output_details()[0]["index"]
        self.fe = Frontend(threads=4)

    def score(self, x: np.ndarray) -> float:
        self.it.set_tensor(self.inp, x[None].astype(np.float32))
        self.it.invoke()
        return float(self.it.get_tensor(self.out)[0, 0])

    def peak(self, path: Path) -> float:
        a, _ = sf.read(str(path), dtype="int16", always_2d=True)
        a = np.concatenate([np.zeros(SR, np.int16), a[:, 0], np.zeros(SR // 2, np.int16)])
        a = a.astype(np.float32)  # openWakeWord's frontend takes int16-scaled float samples
        win, hop = int(CLIP_S * SR), 1280
        clips = [a[e - win : e] for e in range(win, len(a) + 1, hop)]
        feats = self.fe.features(clips)
        return max(self.score(f) for f in feats)
