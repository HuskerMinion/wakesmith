"""Best of N: train several microWakeWord candidates and keep the one that does best on held-out real
recordings.

Runs of one recipe on the same data differ as much as a change of data does (in one house's rounds, three
seeds fired on 3, 22 and 18 of the same 124 unseen device clips), so picking among a few is worth more than
most tuning. Each candidate is kept in mww/candidates/s<N>/ with its seed, its own validation score and its
held-out scores; mww/candidates/pick.json says which one was chosen and why, and that one becomes the
project's model (mww/trained/).

The pick, at the project's cutoff: the fewest false triggers on held-out real negatives decides first
(a quarter more, at least one, still counts as tied), then the most held-out positives caught, then fewer
false triggers, then microWakeWord's validation score. Without held-out recordings it falls back to the
validation score alone.

The pick sees only the "pick" half of the held-out recordings (evaluate.split); the test stage reports the
kept model on the other half, so its numbers aren't flattered by the choice. pick.json records every
candidate on both halves and on the whole set. Too few held-out clips to split: the whole set, as before.

A new best-of-N run never deletes the last round: mww/candidates moves to mww/candidates.previous (each
candidate slimmed to KEEP_CANDIDATE, a few hundred KB) and mww/combined to mww/combined.previous, replacing
any older .previous, as mww/trained.previous works for the model. `wakesmith combine --previous` averages
the previous round's candidates.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
from pathlib import Path

from . import evaluate, mww
from .project import Project
from .real import RealData

KEEP_CANDIDATE = {  # a few hundred KB per candidate: enough to average or rescore it later
    "best_weights.weights.h5",
    "last_weights.weights.h5",
    "stream_state_internal",
    "tflite_stream_state_internal_quant",
}
_BEST = re.compile(r"^best: step (\d+), (\S+) = ([\d.]+)", re.MULTILINE)


def _validation(log_file: Path, start: int) -> float | None:
    """The validation score the trainer logged after byte offset `start` (progress bars are multibyte)."""
    raw = log_file.read_bytes()[start:] if log_file.exists() else b""
    m = _BEST.findall(raw.decode("utf-8", "replace"))
    return float(m[-1][2]) if m else None


def choose(cands: list[dict], on: str = "") -> tuple[dict, str]:
    """The candidate to keep and a one-line reason. on: which held-out clips the numbers are from."""
    ok = [c for c in cands if not c.get("failed")]
    if not ok:
        raise RuntimeError("every microWakeWord candidate failed to train")
    val = lambda c: c.get("validation") or 0.0
    if all("false" in c for c in ok):
        fmin = min(c["false"] for c in ok)
        slack = max(1, round(0.25 * fmin))
        tied = [c for c in ok if c["false"] <= fmin + slack]
        best = max(tied, key=lambda c: (c["caught"], -c["false"], val(c)))
        return best, (
            f"s{best['n']}: {best['caught']}/{best['positives']} held-out positives caught, "
            f"{best['false']}/{best['negatives']} false triggers at {best['cutoff']:.2f}"
            + (f" on the {on}" if on else "")
            + f" (the fewest was {fmin}; candidates within {slack} of it compete on positives caught)"
        )
    best = max(ok, key=val)
    return best, f"s{best['n']}: best validation score ({val(best):.4f}; no held-out recordings)"


def train(p: Project, log_file: Path, device: str, log=print) -> dict | None:
    n = max(1, p.microwakeword.candidates)
    if n == 1:
        mww.train(p, log_file, device=device)
        return None
    out = p.path("mww", "candidates")
    keep_previous(p, log)
    out.mkdir(parents=True)
    held = bool(RealData(p)) and any(p.path("real_neg", "test").glob("*.wav"))
    cutoff = p.microwakeword.cutoff
    cands: list[dict] = []
    halves: dict | None = None
    split_note: str | None = None
    for i in range(1, n + 1):
        seed = random.SystemRandom().randrange(2**31)
        log(f"mww candidate {i} of {n} (seed {seed})")
        start = log_file.stat().st_size if log_file.exists() else 0
        c: dict = {"n": i, "seed": seed}
        try:
            mww.train(p, log_file, device=device, seed=seed)
        except RuntimeError as e:
            log(f"mww candidate {i} failed: {e}")
            cands.append({**c, "failed": str(e)})
            continue
        d = out / f"s{i}"
        shutil.copytree(p.path("mww", "trained"), d)
        c["validation"] = _validation(log_file, start)
        if held:
            model = d / mww.model_path(p).relative_to(p.path("mww", "trained"))
            r = evaluate.score(p, "mww", model=model, tag=f"s{i}")
            if split_note is None:
                halves, split_note = evaluate.split(r["pos"], r["neg"])
                log(f"mww: candidates {split_note}")
            c.update(scored(r, cutoff, halves))
        log(
            f"mww candidate {i}: validation {c['validation']}"
            + (
                f", held-out{' (pick half)' if halves else ''} {c['caught']}/{c['positives']}"
                f" caught, {c['false']} false"
                if held
                else ""
            )
        )
        cands.append(c)
    best, why = choose(cands, "half of the held-out recordings used for choosing" if halves else "")
    shutil.rmtree(p.path("mww", "trained"), ignore_errors=True)
    shutil.copytree(out / f"s{best['n']}", p.path("mww", "trained"))
    pick = {"picked": best["n"], "why": why, "candidates": cands}
    if held:
        pick.update(split=halves, split_note=split_note)
    (out / "pick.json").write_text(json.dumps(pick, indent=1))
    log(f"mww: kept candidate {why}")
    return {"candidates": n, "picked": best["n"], "why": why}


def scored(r: dict, cutoff: float, halves: dict | None) -> dict:
    """A candidate's held-out numbers at the cutoff: caught/false/positives/negatives on what the pick
    may see (the pick half, or everything without a split), plus "report" (the other half) and "full"."""
    full = evaluate.at(r, cutoff)
    if not halves:
        return {"cutoff": cutoff, **full}
    return {
        "cutoff": cutoff,
        **evaluate.at(evaluate.half(r, "pick"), cutoff),
        "report": evaluate.at(evaluate.half(r, "report"), cutoff),
        "full": full,
    }


def _drop(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def keep_previous(p: Project, log=print) -> None:
    """Move the last round's candidates (and its combined model) aside as *.previous, replacing older
    ones, and slim the candidates to their weights and models (KEEP_CANDIDATE)."""
    root = p.path("mww")
    cands, combined = root / "candidates", root / "combined"
    moved = []
    if os.path.lexists(cands):
        prev = root / "candidates.previous"
        _drop(prev)
        cands.rename(prev)
        for d in prev.glob("s*"):
            if d.is_dir() and not d.is_symlink():
                for x in d.iterdir():
                    if x.name not in KEEP_CANDIDATE:
                        _drop(x)
        moved.append("candidates")
        if not os.path.lexists(combined):  # an older combined model belongs to older candidates
            _drop(root / "combined.previous")
    if os.path.lexists(combined):
        _drop(root / "combined.previous")
        combined.rename(root / "combined.previous")
        moved.append("combined model")
    if moved:
        log(f"mww: kept the previous round's {' and '.join(moved)} in mww/*.previous")
