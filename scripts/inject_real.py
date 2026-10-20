"""Copy the prepared real recordings into livekit-wakeword's generated splits, numbered from
clip_100000 so they never collide with the synthetic clip_0NNNNN files. Each copy gets its own random
noise/reverb/pitch in the augment stage.
  WORK/real/train      -> positive_train  x REAL_COPIES      (default 10)
  WORK/real_neg/train  -> negative_train  x REAL_NEG_COPIES  (default 2)
Re-running replaces the previous injection (and its augmented rounds)."""
import shutil
from pathlib import Path

import recipe


def inject(src: Path, split: str, copies: int) -> None:
    dst = recipe.OUTPUT / split
    for old in dst.glob("clip_1[0-9][0-9][0-9][0-9][0-9]*.wav"):
        old.unlink()
    srcs = sorted(src.glob("*.wav"))
    n = 0
    for _ in range(copies):
        for s in srcs:
            shutil.copy(s, dst / f"clip_{100000 + n:06d}.wav")
            n += 1
    print(f"injected {len(srcs)} clips x{copies} = {n} from {src} into {split}")


inject(recipe.REAL / "train", "positive_train", int(recipe.num("REAL_COPIES", 10)))
inject(recipe.REAL_NEG / "train", "negative_train", int(recipe.num("REAL_NEG_COPIES", 2)))
