"""microWakeWord: features, training, streaming model, scoring.

Features are microWakeWord's 40-bin, 10 ms spectrograms of augmented clips, built on every CPU
core and stored as RaggedMmap shards (<project>/mww/features/<set>/<split>/shardNN_mmap). Training
runs the upstream notebook's mixednet in wakesmith's PyTorch port of microWakeWord's trainer (mww_torch:
on a GPU, else the CPU); microWakeWord itself exports the deliverable, its streaming int8 model
(tflite_stream_state_internal_quant).
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from . import assets, conversational
from .project import Project
from .real import RealData

NEGATIVE_SETS = [  # (asset, features dir name, sampling weight, truncation)
    ("mww-speech", "speech", 10.0, "random"),
    ("mww-dinner-party", "dinner_party", 10.0, "random"),
    ("mww-no-speech", "no_speech", 5.0, "random"),
    ("mww-dinner-party-eval", "dinner_party_eval", 0.0, "split"),
]
MODEL_ARGS = [
    "mixednet",
    "--pointwise_filters",
    "64,64,64,64",
    "--repeat_in_block",
    "1, 1, 1, 1",
    "--mixconv_kernel_sizes",
    "[5], [7,11], [9,15], [23]",
    "--residual_connection",
    "0,0,0,0",
    "--first_conv_filters",
    "32",
    "--first_conv_kernel_size",
    "5",
    "--stride",
    "3",
]
EVAL_INTERVAL = 500  # training steps between validations (and "Step #" log lines)
# The frontend's float output is uint16 * 0.0390625 exactly; storing uint16 halves the disk use and
# the trainer rescales uint16 sets itself (as for upstream's negative sets).
_FEATURE_SCALE = 0.0390625


def model_path(p: Project) -> Path:
    return p.path(
        "mww", "trained", "tflite_stream_state_internal_quant", "stream_state_internal_quant.tflite"
    )


def _originals(d: Path) -> list[str]:
    return sorted(str(x) for x in d.glob("clip_[0-9][0-9][0-9][0-9][0-9][0-9].wav"))


def feature_jobs(p: Project) -> list[tuple[str, str, list[str], int, int]]:
    """(set, split, files, repeat, slide_frames) for every feature set."""
    real = RealData(p)
    pos_train = _originals(p.path("clips", "positive_train"))
    real_pos = sorted(str(x) for x in p.path("real", "train").glob("*.wav")) * p.real.copies
    if not p.real.sampling_weight:
        pos_train += real_pos
    pos_test = _originals(p.path("clips", "positive_test"))
    adv_train = _originals(p.path("clips", "negative_train"))
    adv_test = _originals(p.path("clips", "negative_test"))
    random.Random(0).shuffle(pos_test)
    random.Random(0).shuffle(adv_test)
    h, ha = len(pos_test) // 2, len(adv_test) // 2
    jobs = [
        ("positive", "training", pos_train, 2, 10),
        ("positive", "validation", pos_test[:h], 1, 10),
        ("positive", "testing", pos_test[h:], 1, 1),
        ("adversarial", "training", adv_train, 1, 10),
        ("adversarial", "validation", adv_test[:ha], 1, 10),
        ("adversarial", "testing", adv_test[ha:], 1, 1),
    ]
    if p.real.sampling_weight and real_pos:
        jobs.append(("real_positive", "training", real_pos, 1, 10))
    if p.microwakeword.partial_weight:
        # cut-off phrases and sound-alikes ("hey" of "hey jarvis", "okay" of "okay nabu"), each clip
        # cut short in _shard: the model learns to wait for the whole phrase
        rng = random.Random(2)
        synth = _originals(p.path("clips", "positive_train"))
        cut = rng.sample(synth, k=min(10000, len(synth)))
        cut += rng.sample(adv_train, k=min(5000, len(adv_train)))
        jobs.append(("partial", "training", cut, 1, 10))
    if p.conversational.clips > 0 and (conv := conversational.clips(p)):
        # everyday sentences (experimental), each cut at a pause in _shard so the words near the
        # phrase land in the model's window
        jobs.append(("conversational", "training", conv, 1, 10))
    if real.of_kind("not_wake"):
        items, _ = real.training_items("not_wake")
        neg = [str(real.file(s, f)) for s, f in items]
        if neg:
            jobs.append(("real_negative", "training", neg * p.real.negative_copies, 1, 10))
    return jobs


_AUG = None


def impulse_paths(p: Project, rir_dir: str) -> list[str]:
    """Room impulse responses for the augmentation: the MIT set, plus with microwakeword.house_rirs the
    house's claps (sessions.house_rirs), repeated so they are picked about as often as the MIT set
    (the augmentation picks a file at random from the list)."""
    out = [rir_dir]
    if p.microwakeword.house_rirs:
        from . import sessions

        house = [str(f) for f in sessions.house_rirs()]
        if house:
            mit = sum(
                f.lower().endswith((".wav", ".flac"))
                for _, _, files in os.walk(rir_dir, followlinks=True)
                for f in files
            )
            out += house * max(1, mit // len(house))
    return out


def _init_worker(noise_dir: str, rir_paths: list[str]) -> None:
    global _AUG
    import warnings

    warnings.filterwarnings("ignore")
    from microwakeword.audio.augmentation import Augmentation

    seed = int.from_bytes(os.urandom(4), "little")
    random.seed(seed)
    np.random.seed(seed)
    # Settings from microWakeWord's basic_training_notebook.
    _AUG = Augmentation(
        augmentation_duration_s=3.2,
        augmentation_probabilities={
            "SevenBandParametricEQ": 0.1,
            "TanhDistortion": 0.1,
            "PitchShift": 0.1,
            "BandStopFilter": 0.1,
            "AddColorNoise": 0.1,
            "AddBackgroundNoise": 0.75,
            "Gain": 1.0,
            "RIR": 0.5,
        },
        impulse_paths=list(rir_paths),
        background_paths=[noise_dir],
        background_min_snr_db=-5,
        background_max_snr_db=10,
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )


PARTIAL_KEEP = (0.25, 0.6)  # share of the speech a cut-off clip keeps


def cut_short(a: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """The clip cut off part-way through its speech (a random 25-60% of it kept), as when a phrase is
    started but not finished, or a recording or a wake fragment stops early."""
    rng = rng or np.random.default_rng()
    loud = np.flatnonzero(np.abs(a) > 0.05 * max(float(np.abs(a).max()), 1e-6))
    if len(loud) < 2:
        return a
    start, end = int(loud[0]), int(loud[-1])
    return a[: start + int((end - start) * rng.uniform(*PARTIAL_KEEP))]


def end_at_pause(a: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """The clip up to a random quiet 20 ms frame after its first half second of speech, or whole. Training
    keeps the end of each clip, so a long sentence would otherwise only ever show its last words; cutting
    at a pause (mostly between words) shows the others without cutting a word in half."""
    rng = rng or np.random.default_rng()
    frame = 320
    n = len(a) // frame
    if n < 2:
        return a
    rms = np.sqrt((a[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    loud = np.flatnonzero(rms > 0.1 * max(float(rms.max()), 1e-6))
    if len(loud) < 2:
        return a
    first, last = int(loud[0]), int(loud[-1])
    pauses = [i for i in range(first + 25, last) if rms[i] <= 0.1 * rms.max()]
    i = int(rng.integers(0, len(pauses) + 1))
    return a if i == len(pauses) else a[: (pauses[i] + 1) * frame]


def _shard(args) -> int:
    _set, _split, files, repeat, slide, out = args
    from microwakeword.audio.audio_utils import generate_features_for_clip
    from mmap_ninja.ragged import RaggedMmap

    def gen():
        for _ in range(repeat):
            for f in files:
                audio, sr = sf.read(f, dtype="float32", always_2d=True)
                if sr != 16000:
                    raise ValueError(f"{f}: {sr} Hz (want 16 kHz)")
                a = audio[:, 0]
                if _set == "partial":
                    a = cut_short(a)
                elif _set == "conversational":
                    a = end_at_pause(a)
                spec = generate_features_for_clip(_AUG.augment_clip(a), 10)
                spec = np.round(spec / _FEATURE_SCALE).astype(np.uint16)
                if slide == 1:
                    yield spec
                    continue
                n = spec.shape[0] - slide + 1
                views = np.lib.stride_tricks.sliding_window_view(spec, (n, spec.shape[1]))
                for i in range(slide):
                    yield np.squeeze(views[i]).copy()

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    RaggedMmap.from_generator(out_dir=out, sample_generator=gen(), batch_size=200, verbose=False)
    return len(files) * repeat


def build_features(p: Project, workers: int, log=print) -> None:
    out_root = p.path("mww", "features")
    shutil.rmtree(out_root, ignore_errors=True)
    noise = str(assets.fetch(assets.ASSETS["musan-noise"]))
    rirs = impulse_paths(p, str(assets.fetch(assets.ASSETS["mit-rirs"])))
    if len(rirs) > 1:
        log(f"mww features: room echoes from the MIT set and {len(set(rirs[1:]))} house claps")
    tasks = []
    for set_name, split, files, repeat, slide in feature_jobs(p):
        files = list(files)
        random.Random(1).shuffle(files)
        n = max(1, min(workers, len(files) // 200))  # >= 200 clips per shard
        for i in range(n):
            out = str(out_root / set_name / split / f"shard{i:02d}_mmap")
            tasks.append((set_name, split, files[i::n], repeat, slide, out))
        log(f"mww features {set_name}/{split}: {len(files)} clips x{repeat}, {n} shards")
    from tqdm import tqdm

    from .hardware import worker_start

    with get_context(worker_start()).Pool(
        workers, initializer=_init_worker, initargs=(noise, rirs)
    ) as pool:
        for _ in tqdm(pool.imap_unordered(_shard, tasks), total=len(tasks), desc="mww features"):
            pass


def write_config(p: Project) -> Path:
    neg_root = assets.home() / "mww" / "negative_datasets"
    feats = [
        {
            "features_dir": "features/positive",
            "sampling_weight": 2.0,
            "penalty_weight": 1.0,
            "truth": True,
            "truncation_strategy": "truncate_start",
            "type": "mmap",
        },
        {
            "features_dir": "features/adversarial",
            "sampling_weight": p.microwakeword.adversarial_weight,
            "penalty_weight": 1.0,
            "truth": False,
            "truncation_strategy": "truncate_start",
            "type": "mmap",
        },
    ]
    if p.real.sampling_weight and p.path("mww", "features", "real_positive").exists():
        feats.append(
            {
                "features_dir": "features/real_positive",
                "sampling_weight": p.real.sampling_weight,
                "penalty_weight": 1.0,
                "truth": True,
                "truncation_strategy": "truncate_start",
                "type": "mmap",
            }
        )
    if p.microwakeword.partial_weight and p.path("mww", "features", "partial").exists():
        feats.append(
            {
                "features_dir": "features/partial",
                "sampling_weight": p.microwakeword.partial_weight,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "truncate_start",
                "type": "mmap",
            }
        )
    if p.conversational.clips > 0 and p.path("mww", "features", "conversational").exists():
        feats.append(
            {
                "features_dir": "features/conversational",
                "sampling_weight": p.conversational.weight,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "truncate_start",
                "type": "mmap",
            }
        )
    if p.path("mww", "features", "real_negative").exists():
        # A real false trigger costs more than a synthetic one.
        feats.append(
            {
                "features_dir": "features/real_negative",
                "sampling_weight": 2.0,
                "penalty_weight": 2.0,
                "truth": False,
                "truncation_strategy": "truncate_start",
                "type": "mmap",
            }
        )
    for asset, name, weight, trunc in NEGATIVE_SETS:
        assets.fetch(assets.ASSETS[asset])
        feats.append(
            {
                "features_dir": str(neg_root / name),
                "sampling_weight": weight,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": trunc,
                "type": "mmap",
            }
        )
    phases = 2 if p.microwakeword.finetune_steps else 1
    cfg = {
        "window_step_ms": 10,
        "train_dir": "trained",
        "features": feats,
        "training_steps": [p.microwakeword.steps, p.microwakeword.finetune_steps][:phases],
        "positive_class_weight": [1] * phases,
        "negative_class_weight": [p.microwakeword.negative_class_weight] * phases,
        "learning_rates": [0.001, 0.0002][:phases],
        "batch_size": 128,
        "time_mask_max_size": [0] * phases,
        "time_mask_count": [0] * phases,
        "freq_mask_max_size": [0] * phases,
        "freq_mask_count": [0] * phases,
        "eval_step_interval": EVAL_INTERVAL,
        "clip_duration_ms": 1500,
        "target_minimization": 0.9,
        "minimization_metric": None,
        "maximization_metric": "average_viable_recall",
    }
    path = p.path("mww", "training.yaml")
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def train_device(gpus: list) -> str:
    """Where training runs: the first GPU with room for the validation sets (~5 GB), else the CPU."""
    big = [g for g in gpus if g.memory_gb >= 6]
    return big[0].device if big else "cpu"


def more_open_files() -> None:
    """Every batch worker, and microWakeWord's export step, maps every feature shard (several files each): a
    project with many real recordings passed Ubuntu/WSL's default soft limit of 1024 open files. Raise it toward
    the hard limit (child processes inherit it)."""
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = 65536 if hard == resource.RLIM_INFINITY else min(65536, hard)
    # macOS refuses more than OPEN_MAX (10240) when the hard limit is unlimited
    for n in (want, 10240):
        if soft >= n:
            return
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (n, hard))
            return
        except (ValueError, OSError):
            continue


def train(
    p: Project, log_file: Path, resume: bool = False, device: str = "cpu", seed: int | None = None
) -> Path:
    """Train, then export through microWakeWord. A finished earlier run is moved aside unless resume.

    The trainer is wakesmith's PyTorch port (mww_torch) on `device`, a GPU or the CPU; it always starts
    over (a run takes minutes). WAKESMITH_MWW_TRAINER=tf runs microWakeWord's own trainer (CPU, ~3-6x
    slower), which can resume from its checkpoint."""
    more_open_files()
    cfg = write_config(p)
    run_dir = p.path("mww", "trained")
    if run_dir.exists() and not resume:
        shutil.rmtree(p.path("mww", "trained.previous"), ignore_errors=True)
        run_dir.rename(p.path("mww", "trained.previous"))
    test = [
        "--test_tf_nonstreaming",
        "0",
        "--test_tflite_nonstreaming",
        "0",
        "--test_tflite_nonstreaming_quantized",
        "0",
        "--test_tflite_streaming",
        "0",
        "--test_tflite_streaming_quantized",
        "1",
        "--use_weights",
        "best_weights",
        *MODEL_ARGS,
    ]
    tf_trainer = os.environ.get("WAKESMITH_MWW_TRAINER") == "tf"
    cpus = os.cpu_count() or 1
    # TensorFlow's trainer is mostly one busy thread; past 8, its extra threads only contend (i9-13900K:
    # 51.8 s per 1000 steps at 8 threads, 57.2 at all 32). WAKESMITH_TF_THREADS overrides.
    threads = os.environ.get("WAKESMITH_TF_THREADS") or str(min(8, cpus))
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": os.environ.get("WAKESMITH_TF_GPU", "-1"),
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "TF_NUM_INTRAOP_THREADS": threads,
        "TF_NUM_INTEROP_THREADS": "2",
        "OMP_NUM_THREADS": threads,
    }
    export = [
        sys.executable,
        "-m",
        "microwakeword.model_train_eval",
        f"--training_config={cfg.name}",
    ]
    if tf_trainer:
        runs = [
            (
                [
                    sys.executable,
                    "-m",
                    "wakesmith.mww_runner",
                    f"--training_config={cfg.name}",
                    "--train",
                    "1",
                    "--restore_checkpoint",
                    "1",
                    *test,
                ],
                env,
            )
        ]
    else:
        # batch workers build batches while the model trains: on a GPU they are the CPU's only job
        workers = max(1, min(6, cpus - 2)) if device != "cpu" else max(1, min(4, cpus // 3))
        torch_env = {**env, "OMP_NUM_THREADS": str(max(1, min(8, cpus - workers)))}
        cuda = os.environ.get(
            "WAKESMITH_CUDA_DEVICES", "all"
        )  # what the pipeline hid from TensorFlow
        if cuda == "all":
            torch_env.pop("CUDA_VISIBLE_DEVICES")
        else:
            torch_env["CUDA_VISIBLE_DEVICES"] = cuda
        trainer = [
            sys.executable,
            "-m",
            "wakesmith.mww_torch",
            "--device",
            device,
            "--workers",
            str(workers),
        ]
        if device == "cpu":
            trainer += ["--threads", torch_env["OMP_NUM_THREADS"]]
        if seed is not None:
            trainer += ["--seed", str(seed)]
        runs = [(trainer, torch_env), ([*export, "--train", "0", *test], env)]
    with open(log_file, "a") as log:
        for cmd, cmd_env in runs:
            rc = subprocess.run(
                cmd,
                cwd=p.path("mww"),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=cmd_env,
                check=False,
            ).returncode
            if rc != 0:
                break
    if rc != 0 or not model_path(p).exists():
        raise RuntimeError(f"microWakeWord training failed (exit {rc}): see {log_file}")
    return model_path(p)
