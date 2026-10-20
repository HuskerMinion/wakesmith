"""Prepare the real recordings for livekit-wakeword (see recipe.py for the folder layout).

Positives (REAL_POS_DIR) are trimmed so the phrase ends at the clip end, as livekit-wakeword
end-aligns its positives: the phrase is taken to be the last speech burst, ignoring the final
TRIM_IGNORE_TAIL_S seconds (a device's confirmation chime). -> WORK/real/{train,test}

Negatives (REAL_NEG_DIR) are cut into 2 s windows ending at 2.0, 2.5, ... s, so whatever made an old
model fire is covered at every position in the window; held-out clips stay whole for scoring.
-> WORK/real_neg/{train,test}
"""
import shutil
import sys

import numpy as np
import soundfile as sf

import recipe

HOP = 160                         # 10 ms energy frames
KEEP_S, TAIL_PAD_S = 1.9, 0.08    # positive clip length; silence kept after the phrase
IGNORE_TAIL_S = recipe.num("TRIM_IGNORE_TAIL_S", 0.3)
WIN = 2 * recipe.SR               # negative window
WIN_HOP_S = 0.5


def read(src: str, f: str) -> np.ndarray:
    a, sr = sf.read(recipe.path(src, f), dtype="float32", always_2d=True)
    if sr != recipe.SR:
        sys.exit(f"{src}/{f}: {sr} Hz - convert to 16 kHz mono first (ffmpeg -i in.wav -ar 16000 -ac 1 out.wav)")
    return a[:, 0]


def trim(a: np.ndarray) -> np.ndarray | None:
    e = np.sqrt(np.convolve(a**2, np.ones(HOP * 3) / (HOP * 3), "same"))[::HOP]
    tail = int(IGNORE_TAIL_S * 100)
    search = e[: len(e) - tail] if tail and len(a) > (KEEP_S + IGNORE_TAIL_S) * recipe.SR else e
    thr = max(search.max() * 0.12, np.median(search) * 3)
    on = search > thr
    # last run of speech >= 150 ms, allowing gaps < 150 ms inside it ("hey" .. "jarvis")
    end = None
    i = len(on) - 1
    while i >= 0:
        if on[i]:
            j, gap = i, 0
            while j > 0 and gap < 15:
                j -= 1
                gap = 0 if on[j] else gap + 1
            if i - j >= 15:
                end = i
                break
            i = j
        else:
            i -= 1
    if end is None:
        return None
    stop = min(len(a), int((end / 100 + TAIL_PAD_S) * recipe.SR))
    return a[max(0, stop - int(KEEP_S * recipe.SR)) : stop]


def prep_positives() -> None:
    _, held, _ = recipe.listing("pos")
    train, notes = recipe.training_items("wake")
    print("\n".join(notes))
    shutil.rmtree(recipe.REAL, ignore_errors=True)
    bad = n = 0
    for split, items in (("test", [("pos", f) for f in sorted(held)]), ("train", train)):
        d = recipe.REAL / split
        d.mkdir(parents=True)
        for src, f in items:
            t = trim(read(src, f))
            if t is None:
                bad += 1
                print("no speech found:", src, f, file=sys.stderr)
                continue
            sf.write(d / recipe.flat_name(src, f), t, recipe.SR)
            n += 1
    print(f"positives: {n} clips written ({len(held)} held-out, {len(train)} train incl. corrections), {bad} skipped")


def prep_negatives() -> None:
    _, held, _ = recipe.listing("neg")
    train, notes = recipe.training_items("not_wake")
    print("\n".join(notes))
    shutil.rmtree(recipe.REAL_NEG, ignore_errors=True)
    for d in ("train", "test"):
        (recipe.REAL_NEG / d).mkdir(parents=True)
    for f in sorted(held):
        shutil.copy(recipe.path("neg", f), recipe.REAL_NEG / "test" / recipe.flat_name("neg", f))
    n_win = 0
    for src, f in train:
        a = read(src, f)
        if len(a) < WIN:
            a = np.concatenate([np.zeros(WIN - len(a), np.float32), a])
        name = recipe.flat_name(src, f).removesuffix(".wav")
        for e in np.arange(WIN, len(a) + 1, WIN_HOP_S * recipe.SR).astype(int):
            sf.write(recipe.REAL_NEG / "train" / f"{name}_e{e / recipe.SR:.1f}.wav", a[e - WIN : e], recipe.SR)
            n_win += 1
    print(f"negatives: {len(train)} train clips incl. corrections -> {n_win} windows, {len(held)} held-out clips")


if __name__ == "__main__":
    if "pos" in recipe.SOURCES:
        prep_positives()
    if "neg" in recipe.SOURCES:
        prep_negatives()
