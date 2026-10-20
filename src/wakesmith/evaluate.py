"""Score trained models on the held-out real recordings, the way devices would run them.

  microWakeWord: the streaming int8 model on 10 ms features, one probability per 30 ms; a detection
                 is the mean of the last 5 probabilities (ESPHome's sliding window).
  openWakeWord:  a 2 s window every 80 ms through the frozen frontend at int16 scale (as the
                 openWakeWord runtime feeds it); the peak score counts.
Positives are the trimmed held-out clips (real/test); negatives the full held-out clips
(real_neg/test). Each clip gets 1 s of silence before and 0.5 s after.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf

from . import mww, oww
from .project import Project

THRESHOLDS = (0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97)


def mww_scorer(tflite: Path, window: int = 5) -> Callable[[Path], float]:
    from microwakeword.inference import Model

    def peak(path: Path) -> float:
        a, _ = sf.read(str(path), dtype="float32", always_2d=True)
        a = np.concatenate([np.zeros(16000, np.float32), a[:, 0], np.zeros(8000, np.float32)])
        p = np.asarray(Model(str(tflite)).predict_clip(a, step_ms=10), np.float32)
        if len(p) < window:
            return float(p.max(initial=0))
        return float(np.convolve(p, np.ones(window) / window, mode="valid").max())

    return peak


def score(p: Project, kind: str, model: Path | None = None) -> dict:
    if kind == "mww":
        peak = mww_scorer(model or mww.model_path(p))
    else:
        peak = oww.Scorer(model or p.path("oww", "model", "model.tflite")).peak
    pos = {f.name: peak(f) for f in sorted(p.path("real", "test").glob("*.wav"))}
    neg = {f.name: peak(f) for f in sorted(p.path("real_neg", "test").glob("*.wav"))}
    table = [
        {
            "threshold": t,
            "caught": sum(v >= t for v in pos.values()),
            "false": sum(v >= t for v in neg.values()),
        }
        for t in THRESHOLDS
    ]
    result = {
        "kind": kind,
        "model": str(model or ""),
        "positives": len(pos),
        "negatives": len(neg),
        "table": table,
        "pos": pos,
        "neg": neg,
    }
    out = p.path("results", f"{kind}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    return result


def format_table(r: dict) -> str:
    lines = [
        f"{r['kind']}: {r['positives']} held-out positives, {r['negatives']} held-out negatives",
        "  threshold  caught        false triggers",
    ]
    for row in r["table"]:
        lines.append(
            f"  {row['threshold']:.2f}       {row['caught']:3d}/{r['positives']}"
            f" {row['caught'] / max(1, r['positives']):6.1%}   {row['false']:3d}/{r['negatives']}"
        )
    return "\n".join(lines)
