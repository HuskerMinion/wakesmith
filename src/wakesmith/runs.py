"""Past runs of a project, kept so they can be compared and any of them downloaded.

Each run that reaches export is copied to <project>/runs/<id>/: the held-out results
(results/*.json), the exported model files (export/), and run.json (when, the job's title, a label
you can set, and the settings that shape a model). Runs are only comparable on the same held-out
recordings, so run.json also notes how many held-out clips the results counted.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

from .project import Project

_ID = re.compile(r"^\d{8}-\d{6}$")


def _dir(p: Project) -> Path:
    return p.path("runs")


def _settings(p: Project) -> dict:
    mww, oww = p.microwakeword, p.openwakeword
    return {
        "samples": vars(p.samples),
        "sound_alikes": len(p.negatives),
        "microwakeword": {
            k: getattr(mww, k, None)
            for k in ("enabled", "steps", "finetune_steps", "adversarial_weight", "frontend")
        },
        "openwakeword": {
            k: getattr(oww, k, None) for k in ("enabled", "steps", "layer_dim", "blocks")
        },
        "real": {
            k: getattr(p.real, k, None)
            for k in ("positives", "negatives", "copies", "sampling_weight")
        },
    }


def snapshot(p: Project) -> str | None:
    """Keep the run that just exported. Returns its id (None when there is nothing to keep)."""
    if not p.path("export").is_dir():
        return None
    rid = time.strftime("%Y%m%d-%H%M%S")
    d = _dir(p) / rid
    d.mkdir(parents=True, exist_ok=False)
    shutil.copytree(p.path("export"), d / "export")
    if p.path("results").is_dir():
        shutil.copytree(p.path("results"), d / "results")
    job = p.path("job.json")
    title = json.loads(job.read_text()).get("title", "") if job.exists() else ""
    (d / "run.json").write_text(
        json.dumps(
            {"id": rid, "time": time.time(), "title": title, "label": "", "settings": _settings(p)},
            indent=1,
        )
    )
    return rid


def _check(rid: str) -> str:
    if not _ID.match(rid):
        raise ValueError("not a run id")
    return rid


def path(p: Project, rid: str) -> Path:
    d = _dir(p) / _check(rid)
    if not (d / "run.json").exists():
        raise FileNotFoundError(rid)
    return d


def listing(p: Project) -> list[dict]:
    out = []
    for d in sorted(_dir(p).glob("*/run.json"), reverse=True) if _dir(p).exists() else []:
        run = json.loads(d.read_text())
        res = {}
        for kind in ("mww", "oww"):
            f = d.parent / "results" / f"{kind}.json"
            if f.exists():
                r = json.loads(f.read_text())
                res[kind] = {
                    "table": r["table"],
                    "positives": r["positives"],
                    "negatives": r["negatives"],
                }
        run["results"] = res
        run["exported"] = sorted(x.name for x in (d.parent / "export").iterdir() if x.is_dir())
        out.append(run)
    return out


def set_label(p: Project, rid: str, label: str) -> None:
    f = path(p, rid) / "run.json"
    run = json.loads(f.read_text())
    run["label"] = label.strip()[:60]
    f.write_text(json.dumps(run, indent=1))


def delete(p: Project, rid: str) -> None:
    shutil.rmtree(path(p, rid))
