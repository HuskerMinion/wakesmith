"""Build microWakeWord training features from the clips livekit-wakeword generated plus the real
recordings, on every CPU core (the upstream notebook does one clip at a time). Run in .venv-mww
after train.sh's generate and real stages.

Each set is written as RaggedMmap shards, WORK/mww/features/<set>/<split>/shardNN_mmap, which
microWakeWord's trainer loads as one set (it reads every *_mmap folder in a split dir).
  positive (truth)          synthetic phrase clips + trimmed real positives x MWW_REAL_COPIES
  adversarial (false)       synthetic sound-alikes
  real_negative (false)     real negative clips, full length, x MWW_REAL_NEG_COPIES
Real clips honour review corrections; held-out clips are never used.
Usage: mww_features.py [workers]
"""
import os
import random
import shutil
import sys
from multiprocessing import get_context

import numpy as np
import soundfile as sf

import recipe

OUT = recipe.WORK / "mww/features"
GEN = recipe.OUTPUT
REAL_COPIES = int(recipe.num("MWW_REAL_COPIES", 10))
REAL_NEG_COPIES = int(recipe.num("MWW_REAL_NEG_COPIES", 5))


def originals(d) -> list[str]:
    """Raw TTS clips only (clip_0NNNNN.wav), not augmented rounds or injected real clips."""
    return sorted(str(p) for p in d.glob("clip_0[0-9][0-9][0-9][0-9][0-9].wav"))


def real_negatives() -> list[str]:
    items, _ = recipe.training_items("not_wake")
    return [str(recipe.path(s, f)) for s, f in items]


def trimmed_real_positives() -> list[str]:
    """prep_real.py's end-trimmed clips (phrase ends at the clip end), already correction-aware."""
    return sorted(str(p) for p in (recipe.REAL / "train").glob("*.wav"))


# (set, split, files, repeat, slide_frames)
def jobs() -> list[tuple]:
    pos_train = originals(GEN / "positive_train") + trimmed_real_positives() * REAL_COPIES
    pos_test = originals(GEN / "positive_test")
    adv_train, adv_test = originals(GEN / "negative_train"), originals(GEN / "negative_test")
    random.Random(0).shuffle(pos_test)
    random.Random(0).shuffle(adv_test)
    h, ha = len(pos_test) // 2, len(adv_test) // 2
    out = [
        ("positive", "training", pos_train, 2, 10),
        ("positive", "validation", pos_test[:h], 1, 10),
        ("positive", "testing", pos_test[h:], 1, 1),
        ("adversarial", "training", adv_train, 1, 10),
        ("adversarial", "validation", adv_test[:ha], 1, 10),
        ("adversarial", "testing", adv_test[ha:], 1, 1),
    ]
    neg = real_negatives()
    if neg:
        out.append(("real_negative", "training", neg * REAL_NEG_COPIES, 1, 10))
    return out


_AUG = None


def _init():
    global _AUG
    import warnings

    warnings.filterwarnings("ignore")
    from microwakeword.audio.augmentation import Augmentation

    seed = (os.getpid() * 1_000_003 + int.from_bytes(os.urandom(4), "little")) % 2**32
    random.seed(seed)
    np.random.seed(seed)
    # Settings from microWakeWord's basic_training_notebook, with livekit-wakeword's downloaded
    # MUSAN noise + MIT RIRs as the background/impulse sources.
    _AUG = Augmentation(
        augmentation_duration_s=3.2,
        augmentation_probabilities={
            "SevenBandParametricEQ": 0.1, "TanhDistortion": 0.1, "PitchShift": 0.1,
            "BandStopFilter": 0.1, "AddColorNoise": 0.1, "AddBackgroundNoise": 0.75,
            "Gain": 1.0, "RIR": 0.5,
        },
        impulse_paths=[str(recipe.DATA / "rirs")],
        background_paths=[str(recipe.DATA / "backgrounds")],
        background_min_snr_db=-5,
        background_max_snr_db=10,
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )


def _shard(args):
    set_name, split, files, repeat, slide, idx = args
    from mmap_ninja.ragged import RaggedMmap
    from microwakeword.audio.audio_utils import generate_features_for_clip

    def gen():
        for _ in range(repeat):
            for f in files:
                audio, sr = sf.read(f, dtype="float32", always_2d=True)
                assert sr == recipe.SR, f
                spec = generate_features_for_clip(_AUG.augment_clip(audio[:, 0]), 10)
                # The frontend's float output is uint16 * 0.0390625 exactly; storing uint16 halves the
                # disk use and the trainer rescales uint16 sets itself (as for upstream's negative sets).
                spec = np.round(spec / 0.0390625).astype(np.uint16)
                if slide == 1:
                    yield spec
                    continue
                n = spec.shape[0] - slide + 1
                views = np.lib.stride_tricks.sliding_window_view(spec, (n, spec.shape[1]))
                for i in range(slide):
                    yield np.squeeze(views[i]).copy()

    out = OUT / set_name / split / f"shard{idx:02d}_mmap"
    out.parent.mkdir(parents=True, exist_ok=True)
    RaggedMmap.from_generator(out_dir=str(out), sample_generator=gen(), batch_size=200, verbose=False)
    return set_name, split, len(files) * repeat


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count()
    shutil.rmtree(OUT, ignore_errors=True)
    tasks = []
    for set_name, split, files, repeat, slide in jobs():
        files = list(files)
        random.Random(1).shuffle(files)
        n = max(1, min(workers, len(files) // 200))  # >=200 clips per shard
        for i in range(n):
            tasks.append((set_name, split, files[i::n], repeat, slide, i))
        print(f"{set_name}/{split}: {len(files)} clips x{repeat}, slide {slide}, {n} shards", flush=True)
    from tqdm import tqdm

    with get_context("fork").Pool(workers, initializer=_init) as pool:
        for _ in tqdm(pool.imap_unordered(_shard, tasks), total=len(tasks), desc="shards"):
            pass
    print("done:", OUT)


if __name__ == "__main__":
    main()
