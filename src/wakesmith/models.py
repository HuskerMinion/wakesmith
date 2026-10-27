"""Model history of a project: every model file it exported, and what became of it.

<project>/models.json is a list of entries, one per distinct model file (by sha256), never removed:
which engine (mww / oww), the file name, when it was made (and the kept run, runs.py), the cutoff
in its manifest and the held-out result at that cutoff, a status with its history, and free-text notes.

Statuses: exported (wakesmith wrote it), offered (handed to someone to try), live (running on the
devices: set by the user, wakesmith doesn't deploy), retired. Only one model per engine is live:
setting a new one live retires the one before it.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path

from . import mww
from .project import Project

STATUSES = ("exported", "offered", "live", "retired")
# export folder -> engine, model file suffix
KINDS = {
    "microwakeword": ("mww", ".tflite"),
    "openwakeword": ("oww", ".tflite"),
}
# engines shown; entries of engines wakesmith no longer trains stay in the file but are left out
ENGINES = {kind for kind, _ in KINDS.values()}
NOTES_MAX = 2000
_RUN_ID = re.compile(r"^\d{8}-\d{6}$")


def _file(p: Project) -> Path:
    return p.path("models.json")


def load(p: Project) -> list[dict]:
    f = _file(p)
    return json.loads(f.read_text()) if f.exists() else []


def shown(p: Project) -> list[dict]:
    """The entries of the engines wakesmith trains."""
    return [e for e in load(p) if e.get("kind") in ENGINES]


@contextmanager
def _editing(p: Project):
    """Read-modify-write under a lock: the server and a training job can both write."""
    f = _file(p)
    with open(p.path(".models.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        entries = load(p)
        yield entries
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(entries, indent=1))
        os.replace(tmp, f)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _model_files(export_dir: Path) -> list[tuple[str, Path]]:
    out = []
    for sub, (kind, suffix) in KINDS.items():
        d = export_dir / sub
        out += [(kind, f) for f in sorted(d.glob(f"*{suffix}"))] if d.is_dir() else []
    return out


def _cutoff(kind: str, model: Path) -> float | None:
    """The cutoff in the manifest next to an exported model."""
    m = model.with_suffix(".json")
    if not m.exists():
        return None
    j = json.loads(m.read_text())
    return j.get("micro", {}).get("probability_cutoff") if kind == "mww" else j.get("threshold")


def _heldout(results: Path, kind: str, cutoff: float | None) -> dict | None:
    """Held-out result at the cutoff from results/<kind>.json (kind: a results kind such as mww or
    mww-combined; None without one, or without a row at that cutoff)."""
    f = results / f"{kind}.json"
    if cutoff is None or not f.exists():
        return None
    r = json.loads(f.read_text())
    row = next((x for x in r["table"] if abs(x["threshold"] - cutoff) < 1e-9), None)
    if not row:
        return None
    return {
        "caught": row["caught"],
        "positives": r["positives"],
        "false": row["false"],
        "negatives": r["negatives"],
    }


def _source(p: Project, kind: str) -> Path:
    """The trained model an export copies."""
    if kind == "mww":
        return mww.model_path(p)
    return p.path("oww", "model", "model.tflite")


def _entry(kind: str, f: Path, digest: str, created: float, run: str | None, cutoff, held) -> dict:
    return {
        "sha256": digest,
        "sha": digest[:8],
        "kind": kind,
        "file": f.name,
        "created": created,
        "run": run,
        "cutoff": cutoff,
        "heldout": held,
        "status": "exported",
        "history": [{"status": "exported", "time": created}],
        "notes": "",
    }


def record_export(p: Project, combined: dict | None = None) -> list[dict]:
    """Note the model files just written to <project>/export (called by export.export). A model
    already listed keeps its entry; when it was exported again at another cutoff, the entry takes
    the new cutoff and the change goes into its history. combined: the combined model's info
    (ensemble.info) when the microWakeWord file is the combined model: its held-out result comes from
    results/mww-combined.json and the entry notes the candidates. Returns the entries of this export."""
    from . import ensemble

    now = time.time()
    out = []
    with _editing(p) as entries:
        by_sha = {e["sha256"]: e for e in entries}
        for kind, f in _model_files(p.path("export")):
            digest, cutoff = sha256(f), _cutoff(kind, f)
            both = kind == "mww" and combined
            src = ensemble.model_path(p) if both else _source(p, kind)
            rkind = "mww-combined" if both else kind
            res = p.path("results", f"{rkind}.json")
            # results older than the trained model belong to an earlier model
            fresh = res.exists() and (
                not src.exists() or res.stat().st_mtime >= src.stat().st_mtime
            )
            held = _heldout(p.path("results"), rkind, cutoff) if fresh else None
            e = by_sha.get(digest)
            if e is None:
                e = _entry(kind, f, digest, now, None, cutoff, held)
                if both:
                    e["combined"] = list(combined["candidates"])
                    e["notes"] = e["history"][0]["note"] = ensemble.note(combined)
                entries.append(e)
                by_sha[digest] = e
            elif e["cutoff"] != cutoff:
                was = e["cutoff"]
                e["cutoff"], e["heldout"] = cutoff, held or e["heldout"]
                e["history"].append(
                    {
                        "status": e["status"],
                        "time": now,
                        "note": f"exported again at cutoff {cutoff} (was {was})",
                    }
                )
            out.append(e)
    return out


def attach_run(p: Project, rid: str) -> None:
    """Link the entries of models kept in run `rid` (runs.snapshot) that have no run yet."""
    shas = {sha256(f) for _, f in _model_files(p.path("runs", rid, "export"))}
    with _editing(p) as entries:
        for e in entries:
            if e["run"] is None and e["sha256"] in shas:
                e["run"] = rid


def find(entries: list[dict], sha: str) -> dict:
    """The entry whose sha256 starts with `sha` (at least the 8-character short sha)."""
    sha = sha.lower()
    hits = [e for e in entries if len(sha) >= 8 and e["sha256"].startswith(sha)]
    if len(hits) != 1:
        raise KeyError(sha)
    return hits[0]


def set_status(p: Project, sha: str, status: str) -> dict:
    """Change a model's status. Returns {"entry": ..., "retired": [short shas]}: setting one live
    retires the model of the same engine that was live before."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {', '.join(STATUSES)}")
    now, retired = time.time(), []
    with _editing(p) as entries:
        e = find(entries, sha)
        if status == "live":
            for o in entries:
                if o is not e and o["kind"] == e["kind"] and o["status"] == "live":
                    o["status"] = "retired"
                    o["history"].append(
                        {"status": "retired", "time": now, "note": f"replaced by {e['sha']}"}
                    )
                    retired.append(o["sha"])
        if e["status"] != status:
            e["status"] = status
            e["history"].append({"status": status, "time": now})
    return {"entry": e, "retired": retired}


def set_notes(p: Project, sha: str, notes: str) -> dict:
    with _editing(p) as entries:
        e = find(entries, sha)
        e["notes"] = notes.strip()[:NOTES_MAX]
    return e


def backfill(p: Project) -> list[dict]:
    """Add entries for model files exported before the history existed: every kept run's export
    (runs/*/export, dated by the run) and the current export folder. Returns the added entries."""
    added = []
    with _editing(p) as entries:
        known = {e["sha256"] for e in entries}
        places = []
        runs_dir = p.path("runs")
        for d in sorted(runs_dir.iterdir()) if runs_dir.is_dir() else []:
            if not (_RUN_ID.match(d.name) and (d / "export").is_dir()):
                continue
            try:
                when = json.loads((d / "run.json").read_text())["time"]
            except (OSError, ValueError, KeyError):
                when = time.mktime(time.strptime(d.name, "%Y%m%d-%H%M%S"))
            places.append((d / "export", d / "results", when, d.name))
        if p.path("export").is_dir():
            places.append((p.path("export"), p.path("results"), None, None))
        for export_dir, results, when, rid in places:
            for kind, f in _model_files(export_dir):
                digest = sha256(f)
                if digest in known:
                    continue
                cutoff = _cutoff(kind, f)
                res = results / f"{kind}.json"
                # the current export's results may belong to a model trained after it
                fresh = rid or (res.exists() and res.stat().st_mtime <= f.stat().st_mtime)
                held = _heldout(results, kind, cutoff) if fresh else None
                e = _entry(kind, f, digest, when or f.stat().st_mtime, rid, cutoff, held)
                e["history"][0]["note"] = "found by Find earlier exports"
                entries.append(e)
                known.add(digest)
                added.append(e)
    return added


def live(p: Project) -> dict:
    """The live model of each engine: {kind: short sha}."""
    return {e["kind"]: e["sha"] for e in shown(p) if e["status"] == "live"}
