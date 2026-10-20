"""List the real recordings where trained models disagree with the label, for the review page.

Scores every labeled recording (full original clip, as a live listener hears it) with each model.
A clip is listed when any model disagrees at THRESHOLD: a positive scoring below it, or a negative
scoring at or above it. Writes WORK/review/clips.json, most confident disagreements first.

In practice these lists hold real labeling mistakes. One household's first pass found 11 wrong
training labels among ~1,000 clips, and listening to every disputed held-out clip confirmed that
their test labels were right.

.onnx models run in .venv; for a microWakeWord .tflite run this with .venv-mww (one kind per run).
Usage: build_list.py MODEL [MODEL ...] [--threshold 0.5] [--split all|train|test]"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import recipe  # noqa: E402
from scoring import scorer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("models", nargs="+")
ap.add_argument("--threshold", type=float, default=0.5)
ap.add_argument("--split", choices=("all", "train", "test"), default="all")
args = ap.parse_args()
peaks = {Path(m).stem: scorer(m) for m in args.models}
thr = args.threshold

clips = []
for src, label in (("pos", "wake"), ("neg", "not_wake")):
    files, held, rows = recipe.listing(src)
    for f in files:
        split = "test" if f in held else "train"
        if args.split not in ("all", split):
            continue
        scores = {name: peak(recipe.path(src, f)) for name, peak in peaks.items()}
        if not any((label == "wake") != (p >= thr) for p in scores.values()):
            continue
        extra = {k: v for k, v in rows[f].items() if k != "file" and v}
        clips.append({
            "id": f"{src}/{f}", "src": src, "file": f, "label": label, "split": split,
            "score": round(max(scores.values()) if label == "not_wake" else min(scores.values()), 3),
            "scores": " · ".join(f"{n} {p:.2f}" for n, p in scores.items()) if len(scores) > 1 else None,
            "how": "; ".join(f"{k}: {v}" for k, v in extra.items()),
        })
clips.sort(key=lambda c: -abs(c["score"] - thr))
out = recipe.WORK / "review/clips.json"
out.parent.mkdir(parents=True, exist_ok=True)
json.dump({"threshold": thr, "phrase": recipe.LABEL, "clips": clips}, open(out, "w"), indent=1)
pos = sum(c["label"] == "wake" for c in clips)
print(f"{len(clips)} disagreements at {thr}: {pos} labeled '{recipe.LABEL}' scored low, "
      f"{len(clips) - pos} labeled other scored high; {sum(c['split'] == 'test' for c in clips)} held-out -> {out}")
