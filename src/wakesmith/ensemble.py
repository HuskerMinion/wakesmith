"""Combined model: one microWakeWord file that averages the best-of-N candidates' probabilities.

Candidates trained from different seeds make different mistakes, while real calls score high in all of
them, so their mean fires less on noise without missing calls. On one house's held-out sets, at the same
budget of 5 false triggers, the mean of 3 candidates caught 21 calls / 144 far-away calls / 186 calls over
TV, against 18 / 123 / 163 for the best single candidate. The file is about N times the size of one model
and needs N times its tensor arena, and runs anywhere a single microWakeWord model runs.

<project>/mww/combined/
  tflite_stream_state_internal_quant/stream_state_internal_quant.tflite   the model
  combined.json   the candidates used, their seeds, when, the tensor arena for the manifest
  combine.log     TensorFlow's output of the builds

A new best-of-N run moves it to mww/combined.previous with its candidates (mww/candidates.previous);
combine(previous=True) builds mww/combined from those previous candidates. TensorFlow runs in a child
process (ensemble_build.py) with cwd = <project>/mww, off the GPU; a failed build leaves an earlier
combined model as it was.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import mww
from .project import Project

TFLITE = Path("tflite_stream_state_internal_quant", "stream_state_internal_quant.tflite")
WEIGHTS = "best_weights.weights.h5"
# microWakeWord's usual tensor arena for one model of this architecture (export.py); a combined model
# holds one copy of every candidate's state and activations
ARENA_PER_MODEL = 30000


def folder(p: Project) -> Path:
    return p.path("mww", "combined")


def model_path(p: Project) -> Path:
    return folder(p) / TFLITE


def info(p: Project) -> dict | None:
    """combined.json of the built model, or None when there is none."""
    f = folder(p) / "combined.json"
    if not (f.exists() and model_path(p).exists()):
        return None
    try:
        return json.loads(f.read_text())
    except ValueError:
        return None


def note(i: dict) -> str:
    names = ",".join(f"s{n}" for n in i["candidates"])
    return (
        f"combined from the previous round's {names}"
        if i.get("previous")
        else f"combined from {names}"
    )


def candidates_dir(p: Project, previous: bool = False) -> Path:
    return p.path("mww", "candidates.previous" if previous else "candidates")


def previous(p: Project) -> dict | None:
    """The previous round kept in mww/candidates.previous: {"candidates": N with weights, "picked",
    "combined": whether its combined model was kept too}, or None."""
    have = available(p, previous=True)
    if not have:
        return None
    try:
        picked = json.loads((candidates_dir(p, True) / "pick.json").read_text()).get("picked")
    except (OSError, ValueError):
        picked = None
    return {
        "candidates": len(have),
        "numbers": [c["n"] for c in have],
        "picked": picked,
        "combined": (p.path("mww", "combined.previous") / TFLITE).exists(),
    }


def available(p: Project, previous: bool = False) -> list[dict]:
    """The candidates that finished with weights: {"n", "seed", "weights"} each, by number. pick.json
    lists them when the pick ran; otherwise (a run stopped before the pick) the s<N> folders do.
    previous: the previous round's (mww/candidates.previous)."""
    root = candidates_dir(p, previous)
    pick = root / "pick.json"
    listed: list[dict] = []
    if pick.exists():
        try:
            listed = json.loads(pick.read_text()).get("candidates", [])
        except ValueError:
            listed = []
    if not listed:
        listed = [
            {"n": int(d.name[1:])} for d in root.glob("s*") if d.is_dir() and d.name[1:].isdigit()
        ]
    out = []
    for c in sorted(listed, key=lambda c: c["n"]):
        w = root / f"s{c['n']}" / WEIGHTS
        if not c.get("failed") and w.is_file():
            out.append({"n": c["n"], "seed": c.get("seed"), "weights": w})
    return out


def combine(
    p: Project, candidates: list[int] | None = None, log=print, previous: bool = False
) -> Path:
    """Build the combined model from `candidates` (numbers; default: every one with weights) and return
    its path. previous: from the previous round's candidates (mww/candidates.previous). Never touches
    mww/trained or the candidates."""
    have = {c["n"]: c for c in available(p, previous)}
    which = "the previous round's " if previous else ""
    if candidates:
        missing = [n for n in candidates if n not in have]
        if missing:
            raise ValueError(
                f"no trained weights for {which}candidate(s) {', '.join(f's{n}' for n in missing)}"
                f" (have: {', '.join(f's{n}' for n in have) or 'none'})"
            )
        use = [have[n] for n in dict.fromkeys(candidates)]
    else:
        use = list(have.values())
    if len(use) < 2:
        raise ValueError(
            f"combining needs at least 2 trained candidates; {p.name} has {len(have)}"
            + (
                " in its previous round"
                if previous
                else " (set microwakeword.candidates to 3 and train)"
            )
        )
    cfg = p.path("mww", "training.yaml")
    if not cfg.exists():
        raise ValueError(f"{cfg} is missing: train microWakeWord first")
    out = folder(p)
    out.mkdir(parents=True, exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix=".build-", dir=out))
    log_file = out / "combine.log"
    names = ",".join(f"s{c['n']}" for c in use)
    log(f"mww: combining {which}candidates {names} into one model")
    mww.more_open_files()
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "-1",
        "TF_CPP_MIN_LOG_LEVEL": "2",
    }
    cmd = [sys.executable, "-m", "wakesmith.ensemble_build", str(build)]
    cmd += [str(c["weights"]) for c in use]
    t0 = time.time()
    try:
        with open(log_file, "a") as f:
            f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} combine {which}{names}\n")
            f.flush()
            rc = subprocess.run(
                cmd, cwd=p.path("mww"), stdout=f, stderr=subprocess.STDOUT, env=env, check=False
            ).returncode
        built = build / TFLITE
        if rc != 0 or not built.exists():
            raise RuntimeError(f"combining the candidates failed (exit {rc}): see {log_file}")
        shutil.rmtree(out / TFLITE.parent, ignore_errors=True)
        (out / TFLITE.parent).mkdir()
        os.replace(built, out / TFLITE)
        meta = {
            "candidates": [c["n"] for c in use],
            "seeds": [c["seed"] for c in use],
            "created": time.time(),
            "seconds": round(time.time() - t0, 1),
            "bytes": (out / TFLITE).stat().st_size,
            "tensor_arena_size": ARENA_PER_MODEL * len(use),
            "previous": previous,
        }
        (out / "combined.json").write_text(json.dumps(meta, indent=1))
    finally:
        shutil.rmtree(build, ignore_errors=True)
    log(f"mww: combined model of {names}: {meta['bytes'] / 1024:.0f} KB in {meta['seconds']:.0f} s")
    return out / TFLITE


def remove(p: Project) -> None:
    """Drop the combined model (its candidates are being replaced)."""
    shutil.rmtree(folder(p), ignore_errors=True)
