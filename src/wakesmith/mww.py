"""microWakeWord: features, training, streaming model, scoring.

Features are microWakeWord's 40-bin, 10 ms spectrograms of augmented clips, built on every CPU
core and stored as RaggedMmap shards (<project>/mww/features/<set>/<split>/shardNN_mmap). Training
runs microWakeWord's own trainer on the upstream notebook's mixednet; the deliverable is its
streaming int8 model (tflite_stream_state_internal_quant).
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

from . import assets
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
    pos_train += sorted(str(x) for x in p.path("real", "train").glob("*.wav")) * p.real.copies
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
    if "neg" in real.sources:
        items, _ = real.training_items("not_wake")
        neg = [str(real.file(s, f)) for s, f in items]
        if neg:
            jobs.append(("real_negative", "training", neg * p.real.negative_copies, 1, 10))
    return jobs


_AUG = None


def _init_worker(noise_dir: str, rir_dir: str) -> None:
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
        impulse_paths=[rir_dir],
        background_paths=[noise_dir],
        background_min_snr_db=-5,
        background_max_snr_db=10,
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )


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
                spec = generate_features_for_clip(_AUG.augment_clip(audio[:, 0]), 10)
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
    rirs = str(assets.fetch(assets.ASSETS["mit-rirs"]))
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

    with get_context("fork").Pool(
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
            "sampling_weight": 3.0,
            "penalty_weight": 1.0,
            "truth": False,
            "truncation_strategy": "truncate_start",
            "type": "mmap",
        },
    ]
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
    cfg = {
        "window_step_ms": 10,
        "train_dir": "trained",
        "features": feats,
        "training_steps": [p.microwakeword.steps],
        "positive_class_weight": [1],
        "negative_class_weight": [p.microwakeword.negative_class_weight],
        "learning_rates": [0.001],
        "batch_size": 128,
        "time_mask_max_size": [0],
        "time_mask_count": [0],
        "freq_mask_max_size": [0],
        "freq_mask_count": [0],
        "eval_step_interval": 500,
        "clip_duration_ms": 1500,
        "target_minimization": 0.9,
        "minimization_metric": None,
        "maximization_metric": "average_viable_recall",
    }
    path = p.path("mww", "training.yaml")
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def train(p: Project, log_file: Path, resume: bool = False) -> Path:
    """Run microWakeWord's trainer (CPU). A finished earlier run is moved aside unless resume."""
    cfg = write_config(p)
    run_dir = p.path("mww", "trained")
    if run_dir.exists() and not resume:
        shutil.rmtree(p.path("mww", "trained.previous"), ignore_errors=True)
        run_dir.rename(p.path("mww", "trained.previous"))
    cmd = [
        sys.executable,
        "-m",
        "microwakeword.model_train_eval",
        f"--training_config={cfg.name}",
        "--train",
        "1",
        "--restore_checkpoint",
        "1",
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
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": os.environ.get("WAKESMITH_TF_GPU", "-1"),
        "TF_CPP_MIN_LOG_LEVEL": "2",
    }
    with open(log_file, "a") as log:
        rc = subprocess.run(
            cmd, cwd=p.path("mww"), stdout=log, stderr=subprocess.STDOUT, env=env, check=False
        ).returncode
    if rc != 0 or not model_path(p).exists():
        raise RuntimeError(f"microWakeWord training failed (exit {rc}): see {log_file}")
    return model_path(p)
