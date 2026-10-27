"""Score trained models on the held-out real recordings, the way devices would run them.

  microWakeWord: the streaming int8 model on 10 ms features, one probability per 30 ms; a detection
                 is the mean of the last 5 probabilities (ESPHome's sliding window). Features come
                 from the project's frontend: "tflm" (ESPHome) or "zserge" (devicefront.py).
                 With "zserge", positives are the full held-out recordings with no padding, so the
                 frontend's noise estimate starts on room sound, as on a listening device.
  openWakeWord:  a 2 s window every 80 ms through the frozen frontend at int16 scale (as the
                 openWakeWord runtime feeds it); the peak score counts.
Positives are the trimmed held-out clips (real/test); negatives the full held-out clips
(real_neg/test). Each clip gets 1 s of silence before and 0.5 s after.

Each result also notes the test set of every clip ("set_of": clip -> "<source>/<top folder>", e.g.
"pos/house", "dneg"; "cv" for Common Voice takes), so the Compare page can break the numbers down.
Older results without it are grouped from the clip names (flat_name keeps the source and folder).

Two halves: best of N (candidates.py) picks the kept model on one half of the held-out recordings, the
"pick" half, and the test reports it on the other, the "report" half, which played no part in choosing
(picking and reporting on the same clips flatters the kept model). A clip's half comes from its prepared
name alone (sha1 parity), so it never changes between runs. With fewer than MIN_HALF phrase clips (or
other clips) in either half there is nothing fair to split: picking and reporting use every clip, as
before, and pick.json says so. A split result keeps the whole set under "full", so its numbers stay
comparable with older results.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf

from . import devicefront, mww, oww
from .project import Project
from .real import KIND, RealData, flat_name

THRESHOLDS = (0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97)
MIN_HALF = 5  # fewer phrase clips (or other clips) than this in a half: no split


def half_of(clip: str) -> str:
    """ "pick" or "report": the half of the held-out recordings a prepared clip belongs to."""
    return "pick" if hashlib.sha1(clip.encode()).digest()[-1] % 2 == 0 else "report"


def split(pos: dict | list, neg: dict | list) -> tuple[dict | None, str]:
    """The two halves' sizes, {"pick": {"positives", "negatives"}, "report": {...}}, or None with the
    reason when a half would be too small to judge on."""
    sizes = {
        h: {
            "positives": sum(half_of(n) == h for n in pos),
            "negatives": sum(half_of(n) == h for n in neg),
        }
        for h in ("pick", "report")
    }
    small = min(s["positives"] for s in sizes.values())
    small_neg = min(s["negatives"] for s in sizes.values())
    if small < MIN_HALF or small_neg < MIN_HALF:
        what = (
            f"{len(pos)} held-out phrase clips" if small < MIN_HALF else f"{len(neg)} other clips"
        )
        return None, (
            f"only {what}: too few to split in two, so the candidates were chosen and are "
            "reported on the same held-out recordings (their numbers flatter the kept one)"
        )
    return sizes, (
        "chosen using half of the held-out recordings "
        f"({sizes['pick']['positives']} phrase clips, {sizes['pick']['negatives']} others); "
        "the test reports the other half "
        f"({sizes['report']['positives']} phrase clips, {sizes['report']['negatives']} others), "
        "which played no part in choosing"
    )


def _table(pos: dict, neg: dict) -> list[dict]:
    return [
        {
            "threshold": t,
            "caught": sum(v >= t for v in pos.values()),
            "false": sum(v >= t for v in neg.values()),
        }
        for t in THRESHOLDS
    ]


def half(r: dict, which: str, positives: bool = True) -> dict:
    """The part of a score() result on one half: positives, negatives, table, pos, neg.
    positives=False keeps every positive (Common Voice takes, never used for picking)."""
    pos = {n: v for n, v in r["pos"].items() if not positives or half_of(n) == which}
    neg = {n: v for n, v in r["neg"].items() if half_of(n) == which}
    return {
        "positives": len(pos),
        "negatives": len(neg),
        "table": _table(pos, neg),
        "pos": pos,
        "neg": neg,
    }


def at(r: dict, cutoff: float) -> dict:
    """Caught and false triggers of a result (or half()) at one cutoff."""
    return {
        "positives": len(r["pos"]),
        "negatives": len(r["neg"]),
        "caught": sum(v >= cutoff for v in r["pos"].values()),
        "false": sum(v >= cutoff for v in r["neg"].values()),
    }


def pick_split(p: Project) -> dict | None:
    """The halves the project's kept microWakeWord model was picked with (pick.json), or None when it
    was picked on every held-out clip (older projects, small sets) or not picked at all (mww/trained
    is no longer the kept candidate, e.g. retrained with one candidate since)."""
    f = p.path("mww", "candidates", "pick.json")
    try:
        pick = json.loads(f.read_text())
        s = pick.get("split")
        if not isinstance(s, dict):
            return None
        kept = (
            f.parent
            / f"s{pick['picked']}"
            / mww.model_path(p).relative_to(p.path("mww", "trained"))
        )
        same = kept.read_bytes() == mww.model_path(p).read_bytes()
    except (OSError, ValueError, KeyError):
        return None
    return s if same else None


def mww_scorer(tflite: Path, window: int = 5, frontend: str = "tflm") -> Callable[[Path], float]:
    from microwakeword.inference import Model

    def peak(path: Path) -> float:
        a, _ = sf.read(str(path), dtype="float32", always_2d=True)
        if frontend == "zserge":
            # the device averages uint8 outputs / 255
            p = Model(str(tflite)).predict_spectrogram(devicefront.features(a[:, 0]))
            p = np.round(np.asarray(p, np.float32) * 255) / 255
        else:
            a = np.concatenate([np.zeros(16000, np.float32), a[:, 0], np.zeros(8000, np.float32)])
            p = np.asarray(Model(str(tflite)).predict_clip(a, step_ms=10), np.float32)
        if len(p) < window:
            return float(p.max(initial=0))
        return float(np.convolve(p, np.ones(window) / window, mode="valid").max())

    return peak


def score(
    p: Project,
    kind: str,
    model: Path | None = None,
    frontend: str | None = None,
    positives: Path | None = None,
    tag: str = "",
    report_half: bool = False,
) -> dict:
    """positives: a folder of positive clips instead of the held-out recordings (e.g. Common Voice
    speakers), saved as results/<kind>-<tag>.json. report_half: when the project's kept model was
    picked on half of the held-out recordings (pick_split), the result is that model on the other half,
    with the whole set under "full"."""
    project_frontend = p.microwakeword.frontend
    frontend = (frontend or project_frontend) if kind == "mww" else None
    if kind == "mww":
        peak = mww_scorer(model or mww.model_path(p), frontend=frontend)
    else:
        peak = oww.Scorer(model or p.path("oww", "model", "model.tflite")).peak
    if positives is not None:
        pos = {f.name: peak(f) for f in sorted(positives.glob("*.wav"))}
    elif frontend == "zserge":
        real = RealData(p)
        pos = {
            flat_name(s, f): peak(real.file(s, f))
            for s in real.of_kind("wake")
            for f in real.heldout(s)
        }
    else:
        pos = {f.name: peak(f) for f in sorted(p.path("real", "test").glob("*.wav"))}
    neg = {f.name: peak(f) for f in sorted(p.path("real_neg", "test").glob("*.wav"))}
    exact = _held_sets(p)
    set_of = {n: "cv" if positives is not None else exact.get(n) or set_name(n) for n in pos}
    set_of.update({n: exact.get(n) or set_name(n) for n in neg})
    result = {
        "kind": kind,
        "frontend": frontend,
        "model": str(model or ""),
        "positives": len(pos),
        "negatives": len(neg),
        "table": _table(pos, neg),
        "pos": pos,
        "neg": neg,
        "set_of": set_of,
    }
    halves = pick_split(p) if report_half and kind == "mww" else None
    if halves:
        full = {k: result[k] for k in ("positives", "negatives", "table", "pos", "neg")}
        result.update(half(result, "report", positives=positives is None))
        result["half"] = "report"
        result["positives_split"] = positives is None  # Common Voice takes: all of them
        result["split"] = halves
        result["full"] = full
    suffix = f"-{frontend}" if frontend and frontend != project_frontend else ""
    suffix += f"-{tag}" if tag else ""
    out = p.path("results", f"{kind}{suffix}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    return result


def brief(r: dict) -> dict:
    """A results file for listings: the table and counts, and for a report-half result which half it is
    and the whole set's table (without the per-clip scores)."""
    out = {k: r[k] for k in ("table", "positives", "negatives")}
    if r.get("half"):
        out["half"] = r["half"]
        out["positives_split"] = r.get("positives_split", True)
        out["full"] = {k: r["full"][k] for k in ("table", "positives", "negatives")}
    return out


def score_combined(p: Project, log: Callable[[str], None] = print) -> dict:
    """Score the combined model (ensemble.py) exactly like the single one, as its own results kind:
    results/mww-combined.json (and mww-combined-cv.json for Common Voice speakers). Results of an
    earlier combined model are removed when there is none now. Returns {kind: table}."""
    from . import ensemble

    files = {k: p.path("results", f"{k}.json") for k in ("mww-combined", "mww-combined-cv")}
    model = ensemble.model_path(p)
    if not (p.microwakeword.enabled and model.exists()):
        for f in files.values():
            f.unlink(missing_ok=True)
        return {}
    out = {}
    if RealData(p):
        r = score(p, "mww", model=model, tag="combined", report_half=True)
        log("combined model: " + format_table(r))
        out["mww-combined"] = r["table"]
    else:
        files["mww-combined"].unlink(missing_ok=True)
    cv_test = p.path("commonvoice", "test")
    if p.commonvoice.enabled and any(cv_test.glob("*.wav")):
        r = score(p, "mww", model=model, positives=cv_test, tag="combined-cv", report_half=True)
        log("combined model, Common Voice speakers: " + format_table(r))
        out["mww-combined-cv"] = r["table"]
    else:
        files["mww-combined-cv"].unlink(missing_ok=True)
    return out


def set_id(src: str, f: str) -> str:
    return f"{src}/{f.split('/')[0]}" if "/" in f else src


def _held_sets(p: Project) -> dict[str, str]:
    """Prepared held-out clip name -> its test set, from the real recordings as they are now."""
    real = RealData(p)
    try:
        return {flat_name(s, f): set_id(s, f) for s in real.sources for f in real.heldout(s)}
    except (OSError, ValueError):  # folders moved since preparing: fall back to the clip names
        return {}


def set_name(clip: str) -> str:
    """A clip's test set from its prepared name alone (results written before "set_of")."""
    src, _, rest = clip.partition("__")
    if src not in KIND or not rest:
        return "other"
    folder, sep, _ = rest.partition("__")
    return f"{src}/{folder}" if sep else src


def by_set(r: dict, cutoffs: list[float]) -> list[dict]:
    """Per test set: how many of its clips reach each cutoff (caught for phrase sets, false
    triggers for the rest)."""
    set_of = r.get("set_of") or {}
    out: dict[tuple[str, str], list[float]] = {}
    for side in ("pos", "neg"):
        for clip, v in (r.get(side) or {}).items():
            out.setdefault((side, set_of.get(clip) or set_name(clip)), []).append(v)
    return [
        {
            "id": sid,
            "phrase": side == "pos",
            "clips": len(vs),
            "hits": [sum(v >= t for v in vs) for t in cutoffs],
        }
        for (side, sid), vs in sorted(out.items(), key=lambda x: (x[0][0] != "pos", x[0][1]))
    ]


def format_table(r: dict) -> str:
    front = f" ({r['frontend']} frontend)" if r.get("frontend") else ""
    n = f"{r['positives']} held-out positives, {r['negatives']} held-out negatives"
    lines = [
        f"{r['kind']}{front}: {n}",
        "  threshold  caught        false triggers",
    ]
    if r.get("half") == "report":
        f = r["full"]
        lines.insert(
            1,
            "  (the half of the held-out recordings that played no part in picking the kept candidate;"
            f' all {f["positives"]} + {f["negatives"]} clips are under "full" in the results file)',
        )
    for row in r["table"]:
        lines.append(
            f"  {row['threshold']:.2f}       {row['caught']:3d}/{r['positives']}"
            f" {row['caught'] / max(1, r['positives']):6.1%}   {row['false']:3d}/{r['negatives']}"
        )
    return "\n".join(lines)
