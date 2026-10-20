"""Score the held-out real recordings (never trained on) with a trained model:
  WORK/real/test      wake-phrase clips   -> how many it catches
  WORK/real_neg/test  non-phrase clips    -> how many false triggers
Run prep_real.py first. .onnx models run in .venv, microWakeWord .tflite models in .venv-mww.
Usage: test_real.py MODEL [--window 5]"""
import argparse

import recipe
from scoring import scorer

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--window", type=int, default=5, help="microWakeWord sliding window (ESPHome default 5)")
args = ap.parse_args()
peak = scorer(args.model, args.window)

pos = sorted((peak(f), f.name) for f in (recipe.REAL / "test").glob("*.wav"))
neg = sorted((peak(f), f.name) for f in (recipe.REAL_NEG / "test").glob("*.wav"))
print(f"{args.model}\n{len(pos)} held-out '{recipe.LABEL}' clips, {len(neg)} held-out other clips\n")
print("threshold  caught            false triggers")
for t in (0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97):
    tp, fp = sum(p >= t for p, _ in pos), sum(p >= t for p, _ in neg)
    print(f"  {t:.2f}     {tp:3d}/{len(pos)} {tp / max(1, len(pos)):6.1%}    {fp:3d}/{len(neg)}")
print(f"\nlowest-scoring '{recipe.LABEL}' clips:")
for p, n in pos[:8]:
    print(f"  {p:.3f}  {n}")
print("highest-scoring other clips:")
for p, n in neg[-8:][::-1]:
    print(f"  {p:.3f}  {n}")
