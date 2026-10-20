"""Find labelling mistakes: list the real recordings a trained model disagrees with.

Every labelled recording (full original clip, train and held-out) is scored the way a device would
hear it. A clip is listed when the score disagrees with its label at `threshold`: a positive below
it, or a negative at or above it. The review page plays these and records verdicts in
review/decisions.json, which the next `real` stage applies (held-out clips are never moved).
"""

from __future__ import annotations

import json
from pathlib import Path

from . import evaluate, mww, oww
from .project import Project
from .real import RealData


def build(p: Project, kind: str = "mww", threshold: float | None = None, progress=None) -> dict:
    real = RealData(p)
    if not real:
        raise ValueError("the project has no real recordings to review")
    if kind == "mww":
        peak = evaluate.mww_scorer(mww.model_path(p), frontend=p.microwakeword.frontend)
        threshold = threshold if threshold is not None else p.microwakeword.cutoff
    else:
        peak = oww.Scorer(p.path("oww", "model", "model.tflite")).peak
        threshold = threshold if threshold is not None else p.openwakeword.threshold
    listings = {src: real.listing(src) for src in real.sources}
    todo = [(src, f) for src, (files, _, _) in listings.items() for f in files]
    clips = []
    for i, (src, f) in enumerate(todo):
        _, held, rows = listings[src]
        label = "wake" if src == "pos" else "not_wake"
        score = peak(real.file(src, f))
        if progress:
            progress(i + 1, len(todo))
        if (label == "wake") == (score >= threshold):
            continue
        extra = {k: v for k, v in rows.get(f, {}).items() if k != "file" and v}
        clips.append(
            {
                "id": f"{src}/{f}",
                "src": src,
                "file": f,
                "label": label,
                "score": round(score, 3),
                "split": "test" if f in held else "train",
                "how": "; ".join(f"{k}: {v}" for k, v in extra.items()),
            }
        )
    clips.sort(key=lambda c: -abs(c["score"] - threshold))
    out = {"threshold": threshold, "model": kind, "phrase": p.display, "clips": clips}
    path = p.path("review", "clips.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return out


def load(p: Project) -> dict:
    path = p.path("review", "clips.json")
    return json.loads(path.read_text()) if path.exists() else {"clips": []}


def decisions(p: Project) -> dict:
    path = RealData(p).decisions_path
    return json.loads(path.read_text()) if path.exists() else {}


def mark(p: Project, clip: dict, verdict: str | None, note: str = "") -> None:
    path: Path = RealData(p).decisions_path
    d = decisions(p)
    if verdict is None:
        d.pop(clip["id"], None)
    else:
        d[clip["id"]] = {
            "verdict": verdict,
            "note": note[:500],
            "was_label": clip["label"],
            "score": clip["score"],
            "split": clip["split"],
        }
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1))
    tmp.replace(path)
