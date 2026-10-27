"""openWakeWord-compatible model: features through openWakeWord's frozen frontend, a small DNN head
trained in Keras, exported natively to .tflite ([1,16,96] float32 -> [1,1]).

Features: every clip is augmented `ROUNDS` times in memory (2 s, phrase at the end, noise, reverb)
and turned into the last 16 speech embeddings (melspectrogram.onnx -> embedding_model.onnx): on every
CPU core, or with a GPU, augmented on the CPU cores and embedded on the GPU (oww_torch, the same
models in PyTorch). Nothing augmented is written to disk.

Training follows the usual openWakeWord-style recipe: mixed batches (positives, synthetic and real
negatives, noise, 2,000 h of general audio from ACAV100M), focal loss with mixup and label smoothing,
a negative-class weight ramped up over training, three phases at falling learning rates, and an
average of the best late checkpoints. On a GPU the same loop runs the model and Keras' AdamW in
PyTorch (oww_torch.HeadTrainer) and copies the result into the Keras model for export; on the CPU,
Keras trains it. Finished features are kept while their inputs are unchanged (stamp.json).
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import soundfile as sf

from . import assets, conversational, hardware
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

    def features(self, clips: list[np.ndarray], chunk: int = 16) -> np.ndarray:
        """Clips -> (len(clips), 16, 96). The windows of `chunk` clips go through the embedding
        model in one call. Small chunks are both lighter and faster on CPU: 8 clips per call use
        ~0.5 GB per process, 128 use ~2.4 GB and run slower."""
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


def _init_worker(noise_dir: str, rir_dir: str, frontend: bool = True) -> None:
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
    if frontend:
        _W["fe"] = Frontend(threads=1)


def _augment(files: list[str], rounds: int) -> list[np.ndarray]:
    clips = []
    for f in files:
        a, sr = sf.read(f, dtype="float32", always_2d=True)
        if sr != SR:
            raise ValueError(f"{f}: {sr} Hz (want 16 kHz)")
        for _ in range(rounds):
            clips.append(pcm_scale(_W["aug"].augment_clip(a[:, 0])))
    return clips


def _chunk(args: tuple[list[str], int]) -> np.ndarray:
    """CPU path: augmented clips -> features, in the worker."""
    if "fe" not in _W:  # a pool started for the GPU path, which then fell back to the CPU
        _W["fe"] = Frontend(threads=1)
    return _W["fe"].features(_augment(*args), chunk=8)


def _chunk_audio(args: tuple[list[str], int]) -> np.ndarray:
    """GPU path: augmented clips only, as int16 (pcm_scale's values are whole numbers: no loss, half
    the bytes through the pipe); the features are made on the GPU in the main process."""
    return np.stack(_augment(*args)).astype(np.int16)


def feature_device(gpus: list) -> str:
    """Where the features' frontend runs: as microWakeWord training, the first GPU with >= 6 GB, else
    the CPU. WAKESMITH_OWW_FEATURES=cpu keeps them on the CPU (ONNX Runtime in every worker)."""
    from .mww import train_device

    if os.environ.get("WAKESMITH_OWW_FEATURES", "").lower() == "cpu":
        return "cpu"
    return train_device(gpus)


# a driver or device problem when PyTorch starts a GPU: the CPU path still works
_GPU_ERRORS = (RuntimeError, AssertionError, ImportError, OSError)


@contextlib.contextmanager
def _cuda_visible(device: str):
    """The pipeline hides CUDA cards (CUDA_VISIBLE_DEVICES=-1) from TensorFlow; PyTorch reads the
    variable when it starts CUDA, so the original setting is put back for that moment."""
    hidden = os.environ.get("CUDA_VISIBLE_DEVICES")
    orig = os.environ.get("WAKESMITH_CUDA_DEVICES")
    if device.startswith("cuda") and orig:
        if orig == "all":
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = orig
    try:
        yield
    finally:
        if hidden is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = hidden


def _gpu_frontend(device: str, log=print):
    """oww_torch.Frontend on `device`, or None (with a log line) if it can't start there."""
    try:
        with _cuda_visible(device):
            from . import oww_torch

            fe = oww_torch.Frontend(device)
            fe.features([np.zeros(int(CLIP_S * SR), np.float32)])  # starts the device now
        return fe
    except _GPU_ERRORS as e:
        log(f"oww features: {device} unavailable ({e}); using the CPU")
        return None


def extract(files: list[str], rounds: int, pool, desc: str, gpu=None) -> np.ndarray:
    """Features of `files` x `rounds` augmentations. `pool` augments (and, without `gpu`, makes the
    features too); with `gpu` (an oww_torch.Frontend) the features are made here, on its device,
    while the workers augment the next chunks."""
    from tqdm import tqdm

    if not files:
        return np.zeros((0, N_STEPS, 96), np.float16)
    chunks = [(files[i : i + 64], rounds) for i in range(0, len(files), 64)]
    out = []
    for res in tqdm(
        pool.imap(_chunk_audio if gpu else _chunk, chunks), total=len(chunks), desc=desc
    ):
        feats = gpu.features(res.astype(np.float32)) if gpu else res
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


FEATURES_VERSION = 1  # raise when augmentation or the frontend changes what the features hold
FEATURE_FILES = [
    f"{k}.npy"
    for k in (
        "positive_train",
        "negative_train",
        "positive_test",
        "negative_test",
        "noise_train",
        "noise_test",
    )
]


def _feature_sets(p: Project) -> dict[str, tuple[list[str], int]]:
    """set name -> (input files, augmentation rounds); real recordings repeat once per copy."""

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
            + conversational.clips(p)  # everyday sentences (experimental; empty when off)
            + sorted(str(x) for x in p.path("real_neg", "train").glob("*.wav"))
            * p.real.negative_copies,
            ROUNDS,
        ),
        "positive_test": (originals("positive_test"), 1),
        "negative_test": (originals("negative_test"), 1),
    }
    return sets


def _stamp(p: Project, sets: dict[str, tuple[list[str], int]]) -> dict:
    """What the features were made from: per set the rounds, the number of input files (real
    recordings count once per copy) and a digest of their names, sizes and times, so new, removed or
    regenerated clips and recordings and changed copy settings all differ."""
    import hashlib

    out = {"version": FEATURES_VERSION, "clip_s": CLIP_S, "sets": {}}
    for name, (files, rounds) in sets.items():
        h = hashlib.sha1()
        for f in files:
            st = os.stat(f)
            h.update(f"{os.path.relpath(f, p.dir)}:{st.st_size}:{st.st_mtime_ns}\n".encode())
        out["sets"][name] = {"rounds": rounds, "files": len(files), "digest": h.hexdigest()}
    return out


def features_current(p: Project, stamp: dict) -> bool:
    d = p.path("oww", "features")
    try:
        old = json.loads((d / "stamp.json").read_text())
    except (OSError, ValueError):
        return False
    return old == stamp and all((d / f).exists() for f in FEATURE_FILES)


def build_features(p: Project, workers: int, log=print, device: str = "cpu") -> dict | None:
    """Augmented features of every set. device "cpu": each worker runs openWakeWord's ONNX frontend
    on its own clips. A GPU (e.g. "xpu:0"): the workers only augment and this process runs the
    frontend in PyTorch on the GPU (the embedding model is ~80% of the CPU path's time), same features
    within float32 rounding. The workers are forked before the GPU is touched in this process.

    Skipped when oww/features/stamp.json matches the inputs (_stamp): a rerun after a failed training
    doesn't redo them. Deleting stamp.json forces a rebuild."""
    d = p.path("oww", "features")
    d.mkdir(parents=True, exist_ok=True)

    sets = _feature_sets(p)
    stamp = _stamp(p, sets)
    if features_current(p, stamp):
        log("oww features: up to date (same clips, recordings and settings); kept")
        return {"skipped": "features up to date"}
    (d / "stamp.json").unlink(missing_ok=True)  # a build cut off half-way must not look current
    on_gpu = device != "cpu"
    noise_dir = str(assets.fetch(assets.ASSETS["musan-noise"]))
    rirs = str(assets.fetch(assets.ASSETS["mit-rirs"]))
    # one pool for every set, forked before this process opens the GPU (forking after it isn't safe)
    nproc = max(1, workers - 1) if on_gpu else workers  # on a GPU, this process feeds it
    with get_context("fork").Pool(
        nproc, initializer=_init_worker, initargs=(noise_dir, rirs, not on_gpu)
    ) as pool:
        gpu = _gpu_frontend(device, log) if on_gpu else None
        log(f"oww features: frontend on {device if gpu else f'{nproc} CPU workers'}")
        for name, (files, rounds) in sets.items():
            log(f"oww features {name}: {len(files)} clips x{rounds}")
            np.save(d / f"{name}.npy", extract(files, rounds, pool, f"oww {name}", gpu))
    fe = gpu or Frontend(threads=max(1, workers))
    noise = _noise_clips(3000)
    np.save(d / "noise_train.npy", fe.features(noise[:2500]).astype(np.float16))
    np.save(d / "noise_test.npy", fe.features(noise[2500:]).astype(np.float16))
    (d / "stamp.json").write_text(json.dumps(stamp, indent=1))
    return None


# --- training ------------------------------------------------------------------------------------


def build_model(layer_dim: int, blocks: int):
    tf = hardware.tensorflow_on_cpu()

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
            xs.append(x)  # float16 as stored; float32 in the trainer (exact)
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


def _metrics(head, v: dict, threshold: float = 0.5) -> dict[str, float]:
    pred = {k: head.predict(x) for k, x in v.items()}
    recall = float((pred["pos"] >= threshold).mean())
    hours = len(v["val"]) * CLIP_S / 3600
    fpph = float((pred["val"] >= threshold).sum() / hours)
    acc = float(
        ((pred["pos"] >= threshold).sum() + (pred["neg"] < threshold).sum())
        / (len(pred["pos"]) + len(pred["neg"]))
    )
    return {"recall": recall, "fpph": fpph, "accuracy": acc}


class _KerasHead:
    """The Keras model and AdamW, behind the interface of oww_torch.HeadTrainer. The step is the
    graph oww.train has always run: gamma, like then, is a Python constant in it (TensorFlow
    rewrites pow(x, 2.0) with a constant exponent), so the CPU path's numbers are unchanged."""

    def __init__(self, model, weight_decay: float):
        self.model, self.wd = model, weight_decay
        self.new_optimizer()

    def new_optimizer(self) -> None:
        tf = hardware.tensorflow_on_cpu()
        self._opt = tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=self.wd)
        self._step = None

    def _build(self, gamma: float):
        tf = hardware.tensorflow_on_cpu()
        model, opt = self.model, self._opt

        def focal(y, pred):
            pred = tf.clip_by_value(pred, 1e-7, 1 - 1e-7)
            bce = -(y * tf.math.log(pred) + (1 - y) * tf.math.log(1 - pred))
            p_t = y * pred + (1 - y) * (1 - pred)
            return bce * tf.pow(1 - p_t, gamma)

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

        return step_fn

    def step(self, x, y, lam, perm, smooth, gamma, neg_w, lr):
        if self._step is None:
            self._step = self._build(float(gamma))
        x = x.astype(np.float32)
        x = lam * x + (1 - lam) * x[perm]
        y = lam * y + (1 - lam) * y[perm]
        y = y * (1 - smooth) + 0.5 * smooth
        return self._step(x, y, np.float32(neg_w), np.float32(lr))

    def predict(self, x) -> np.ndarray:
        return self.model.predict(x, batch_size=4096, verbose=0)[:, 0]

    def get_weights(self) -> list[np.ndarray]:
        return self.model.get_weights()

    def set_weights(self, weights: list[np.ndarray]) -> None:
        self.model.set_weights(weights)


def train_device(gpus: list) -> str:
    """Where the head trains: the first GPU with >= 6 GB (oww_torch), else the CPU (Keras).
    WAKESMITH_OWW_TRAINER=tf keeps Keras on the CPU; =torch uses PyTorch even on the CPU."""
    from .mww import train_device as first_big_gpu

    trainer = os.environ.get("WAKESMITH_OWW_TRAINER", "").lower()
    if trainer == "tf":
        return "tf"
    dev = first_big_gpu(gpus)
    if dev == "cpu":
        return "torch-cpu" if trainer == "torch" else "tf"
    return dev


def train(p: Project, log=print, device: str = "tf") -> Path:
    """Train the head. device "tf": Keras on the CPU; a torch device ("xpu:0", "cuda:0") or "torch-cpu":
    the same recipe in oww_torch.HeadTrainer, started from the Keras model's initial weights; the
    result is copied back into the Keras model, which is saved and exported as before."""
    tf = hardware.tensorflow_on_cpu()

    cfg = p.openwakeword
    tf.keras.utils.set_random_seed(0)  # also seeds np.random: mixup draws
    model = build_model(cfg.layer_dim, cfg.blocks)
    batches = _Batches(p)
    val = _validation(p)
    checkpoints: list[tuple[dict[str, float], list[np.ndarray]]] = []
    history = []
    base_lr, wd, smooth, gamma, alpha = 1e-4, 1e-2, 0.05, 2.0, 0.2
    if device != "tf":
        dev = "cpu" if device == "torch-cpu" else device
        try:
            with _cuda_visible(dev):
                import torch

                from . import oww_torch

                head = oww_torch.HeadTrainer(model.get_weights(), dev, weight_decay=wd)
                val = {
                    k: torch.from_numpy(x).to(dev) for k, x in val.items()
                }  # once, not per check
        except _GPU_ERRORS as e:
            log(f"oww: {dev} unavailable ({e}); training on the CPU")
            device = "tf"
    if device == "tf":
        head = _KerasHead(model, wd)
    log(f"oww: training on {'the CPU (Keras)' if device == 'tf' else device + ' (PyTorch)'}")

    def run_phase(phase: int, steps: int, lr0: float, max_w: float) -> None:
        head.new_optimizer()
        warm, hold = (steps // 5, steps // 3) if phase == 1 else (0, 0)
        every = max(1, steps // 20)
        prog_file = p.path("logs", "oww-progress.json")
        for s in range(steps):
            if s % 100 == 0:
                prog_file.write_text(json.dumps({"phase": phase, "done": s, "total": steps}))
            x, y = batches()
            lam = np.float32(np.random.beta(alpha, alpha))
            perm = np.random.permutation(len(x))
            if s < warm:
                lr = lr0 * s / max(1, warm)
            elif s < warm + hold:
                lr = lr0
            else:
                prog = (s - warm - hold) / max(1, steps - warm - hold)
                lr = lr0 * 0.5 * (1 + math.cos(math.pi * prog))
            neg_w = 1.0 + (max_w - 1.0) * s / max(1, steps)
            loss = head.step(x, y, lam, perm, smooth, gamma, neg_w, lr)
            if s >= steps * 3 // 4 and s % every == 0:
                m = _metrics(head, val)
                checkpoints.append((m, [w.copy() for w in head.get_weights()]))
                history.append({"phase": phase, "step": s, "loss": float(loss), **m})
                log(
                    f"oww phase {phase} step {s}/{steps}: loss {float(loss):.4f} "
                    f"recall {m['recall']:.3f} fpph {m['fpph']:.2f}"
                )

    max_w = cfg.max_negative_weight
    run_phase(1, cfg.steps, base_lr, max_w)
    for phase, lr in ((2, base_lr * 0.1), (3, base_lr * 0.01)):
        if _metrics(head, val)["fpph"] > cfg.target_fp_per_hour:
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
    head.set_weights([np.mean(ws, axis=0) for ws in zip(*chosen)])
    final = _metrics(head, val)
    if device != "tf":  # the Keras model gets the weights and must score as the trainer did
        model.set_weights(head.get_weights())
        pos = val["pos"][:2048].cpu().numpy()
        diff = float(np.abs(model.predict(pos, verbose=0)[:, 0] - head.predict(pos)).max())
        log(f"oww: weights copied into Keras: max score difference {diff:.1e}")
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
    """Float32 .tflite with a fixed [1, 16, 96] input, as the openWakeWord runtime feeds it.

    The layers are written out as plain ops: Keras's LayerNormalization converts with a NEG op that
    small device interpreters lack, so it becomes Mean, Sub, Mul, Add and Rsqrt here."""
    tf = hardware.tensorflow_on_cpu()

    layers = [l for l in model.layers if not isinstance(l, tf.keras.layers.InputLayer)]
    weights = {id(v): tf.constant(np.array(v)) for l in layers for v in l.weights}  # taken eagerly
    const = lambda v: weights[id(v)]

    @tf.function(input_signature=[tf.TensorSpec([1, N_STEPS, 96], tf.float32)])
    def forward(x):
        h = x
        for l in layers:
            if isinstance(l, tf.keras.layers.Flatten):
                h = tf.reshape(h, [1, -1])
            elif isinstance(l, tf.keras.layers.Dense):
                h = tf.matmul(h, const(l.kernel)) + const(l.bias)
                if l.activation is tf.keras.activations.sigmoid:
                    h = tf.sigmoid(h)
                elif l.activation is not tf.keras.activations.linear:
                    raise ValueError(f"unsupported activation in {l.name}")
            elif isinstance(l, tf.keras.layers.LayerNormalization):
                d = h - tf.reduce_mean(h, axis=-1, keepdims=True)
                var = tf.reduce_mean(d * d, axis=-1, keepdims=True)
                h = d * tf.math.rsqrt(var + l.epsilon) * const(l.gamma) + const(l.beta)
            elif isinstance(l, tf.keras.layers.ReLU):
                h = tf.nn.relu(h)
            else:
                raise TypeError(f"unsupported layer {l.name}")
        return h

    conv = tf.lite.TFLiteConverter.from_concrete_functions([forward.get_concrete_function()], model)
    path.write_bytes(conv.convert())
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
        if len(a) < win:  # a very short take: still one full window
            a = np.concatenate([np.zeros(win - len(a), np.float32), a])
        clips = [a[e - win : e] for e in range(win, len(a) + 1, hop)]
        feats = self.fe.features(clips)
        return max(self.score(f) for f in feats)
