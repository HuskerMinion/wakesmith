"""The verdict: is this version better than the one on your devices? In one sentence.

After a version's held-out test, the live model of its wake word (another version's, set live on the
Models list) is scored on exactly the same held-out clips. Then the new version's cutoff is chosen
that catches the most while false-triggering no more than the live model does at its own cutoff, and
both are put side by side: "v10 at 0.70 catches 23 of 24 (v7 now: 21) with 3 false triggers in
4,000 clips (v7 now: 9)". Saved as results/verdict-<kind>.json; nothing is changed on the devices.

Device clips held out from the newest days were saved because the live model fired on them, so they
count against it; the verdict says so when they are part of the test.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from . import evaluate, models, rounds
from .project import Project, all_projects
from .real import flat_name

KINDS = ("mww", "oww")


def model_file(p: Project, entry: dict) -> Path | None:
    """Where a Models-list entry's file is: the export folder, a kept run's export, or the project's
    models/ folder (files added by hand, e.g. models made outside wakesmith)."""
    sub = next(s for s, (k, _) in models.KINDS.items() if k == entry["kind"])
    places = [p.path("export", sub, entry["file"]), p.path("models", entry["file"])]
    if entry.get("run"):
        places.insert(1, p.path("runs", entry["run"], "export", sub, entry["file"]))
    places += sorted(p.path("runs").glob(f"*/export/{sub}/{entry['file']}"), reverse=True)
    for f in places:
        if f.is_file() and models.sha256(f) == entry["sha256"]:
            return f
    return None


def live_version(p: Project, kind: str) -> tuple[Project, dict] | None:
    """The version of p's wake word whose model of this engine is live, and its Models-list entry."""
    for q in sorted(all_projects(), key=lambda q: -q.version):
        if q.family != p.family:
            continue
        for e in models.shown(q):
            if e["kind"] == kind and e["status"] == "live":
                return q, e
    return None


def _result(p: Project, kind: str) -> dict | None:
    f = p.path("results", f"{kind}.json")
    return json.loads(f.read_text()) if f.exists() else None


def compare(p: Project, kind: str = "mww", log=print) -> dict | None:
    """Score the live model on p's held-out clips and write the verdict. None when there is nothing to
    compare: no test result yet, no live version, or p is the live version."""
    r = _result(p, kind)
    found = live_version(p, kind)
    if not r or not found:
        _drop(p, kind)  # an earlier verdict no longer holds (no live version now, or not tested)
        return None
    lp, entry = found
    f = model_file(lp, entry)
    if f is None:
        out = {"kind": kind, "error": f"the live model's file ({entry['file']}) isn't in {lp.name}"}
        _save(p, kind, out)
        return out
    if lp.name == p.name and models.sha256(evaluate_model(p, kind)) == entry["sha256"]:
        _drop(p, kind)  # this version is the one on the devices
        return None
    log(
        f"verdict: scoring the live model {lp.title} ({entry['sha']}) on {p.title}'s held-out clips"
    )
    lr = evaluate.score(p, kind, model=f, tag="live")
    names_pos, names_neg = set(r["pos"]), set(r["neg"])  # the same clips p's result counts
    live_pos = {n: v for n, v in lr["pos"].items() if n in names_pos}
    live_neg = {n: v for n, v in lr["neg"].items() if n in names_neg}
    live_cut = entry.get("cutoff") or (
        lp.microwakeword.cutoff if kind == "mww" else lp.openwakeword.threshold
    )
    live_at = _at(live_pos, live_neg, live_cut)
    own_cut = p.microwakeword.cutoff if kind == "mww" else p.openwakeword.threshold
    best = _best(r["pos"], r["neg"], live_at["false"])
    sets = r.get("set_of", {})
    # labeled device clips (dneg), or device clips kept in a folder of their own ("neg/devices_...")
    held_dev = sum(
        sets.get(n, "").startswith("dneg") or "devices" in sets.get(n, "") for n in r["neg"]
    )
    out = {
        "kind": kind,
        "time": time.time(),
        "positives": len(r["pos"]),
        "negatives": len(r["neg"]),
        "device_negatives": held_dev,
        "live": {
            "name": lp.name,
            "title": lp.title,
            "version": lp.version,
            "sha": entry["sha"],
            "cutoff": live_cut,
            **live_at,
        },
        "own": {"cutoff": own_cut, **_at(r["pos"], r["neg"], own_cut)},
        "best": best,
    }
    out["better"] = is_better(best, live_at)
    out["sentence"] = sentence(p, out)
    _save(p, kind, out)
    return out


def is_better(best: dict | None, live: dict) -> bool:
    """Catches at least as many AND (catches more OR false-triggers less). One that catches fewer is a
    trade-off, never "better"."""
    return (
        bool(best)
        and best["caught"] >= live["caught"]
        and (best["caught"] > live["caught"] or best["false"] < live["false"])
    )


def evaluate_model(p: Project, kind: str) -> Path:
    return evaluate.mww.model_path(p) if kind == "mww" else p.path("oww", "model", "model.tflite")


def _at(pos: dict, neg: dict, cut: float) -> dict:
    return {
        "caught": sum(v >= cut for v in pos.values()),
        "false": sum(v >= cut for v in neg.values()),
    }


def _best(pos: dict, neg: dict, budget: int) -> dict | None:
    """The cutoff that catches the most with no more false triggers than `budget` (ties: the higher
    cutoff, the safer one)."""
    rows = [{"cutoff": t, **_at(pos, neg, t)} for t in evaluate.THRESHOLDS]
    ok = [x for x in rows if x["false"] <= budget]
    return max(ok, key=lambda x: (x["caught"], -x["false"], x["cutoff"])) if ok else None


def sentence(p: Project, v: dict) -> str:
    lv, b = v["live"], v["best"]
    tail = (
        (
            f" {v['device_negatives']} of the not-the-phrase clips were saved by your devices when the"
            " live model fired or nearly did, which counts against it."
        )
        if v["device_negatives"]
        else ""
    )
    if not b:
        return (
            f"{p.title} false-triggers more than {lv['title']} at every cutoff on the same "
            f"{v['negatives']:,} not-the-phrase clips; keep {lv['title']}.{tail}"
        )
    name = f"{p.title} at {b['cutoff']:.2f}"
    caught = (
        f"catches {b['caught']} of {v['positives']} test takes ({lv['title']} now: {lv['caught']})"
    )
    false = f"{b['false']} false trigger{'s' if b['false'] != 1 else ''} in {v['negatives']:,} clips ({lv['title']} now: {lv['false']})"
    # its cutoff is picked on these same clips (the live model keeps its own), which flatters it a little
    note = " Its cutoff was picked on these same clips, which flatters it a little."
    if not v["better"]:
        why = "catches fewer" if b["caught"] < lv["caught"] else "no fewer false triggers"
        return f"{name} {caught} with {false}: {why}, so keep {lv['title']}.{tail}{note}"
    return f"{name} {caught} with {false}: better than what's on your devices.{tail}{note}"


def _drop(p: Project, kind: str) -> None:
    p.path("results", f"verdict-{kind}.json").unlink(missing_ok=True)


def _save(p: Project, kind: str, out: dict) -> None:
    f = p.path("results", f"verdict-{kind}.json")
    f.parent.mkdir(exist_ok=True)
    f.write_text(json.dumps(out, indent=1))


def load(p: Project) -> dict:
    out = {}
    for kind in KINDS:
        f = p.path("results", f"verdict-{kind}.json")
        if f.exists():
            out[kind] = json.loads(f.read_text())
    return out


def everyday(p: Project, kind: str = "mww") -> dict | None:
    """Each cutoff in everyday terms: how many held-out takes it catches, and how many false alarms a
    day it would have raised on your devices, from the held-out device clips (their times give the
    days they cover). Without held-out device clips, false triggers are counted per clip instead."""
    r, cv = _result(p, kind), _result(p, f"{kind}-cv")
    if not r and not cv:
        return None
    r = r or cv
    # no takes of the phrase of its own: who says it comes from the Common Voice speakers' test
    pos = r["pos"] if r["pos"] or not cv else cv["pos"]
    neg = r["neg"]
    at = {flat_name("dneg", c["id"]): c["at"] for c in rounds.load(p)}
    dev = {n: v for n, v in neg.items() if n in at}
    days = None
    if dev:
        t = sorted(datetime.fromisoformat(at[n].replace("Z", "+00:00")).timestamp() for n in dev)
        days = max(1.0, (t[-1] - t[0]) / 86400)
    rows = []
    for c in evaluate.THRESHOLDS:
        row = {"cutoff": c, **_at(pos, neg, c)}
        if dev:
            row["device_false"] = sum(v >= c for v in dev.values())
        rows.append(row)
    return {
        "kind": kind,
        "positives": len(pos),
        "negatives": len(neg),
        "device_clips": len(dev),
        "days": days,
        "rows": rows,
        "half": r.get("half"),
    }
