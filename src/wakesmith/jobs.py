"""Background jobs for the web UI: one at a time (a training run uses the whole machine).

A job is `python -m wakesmith <args>` in its own process session, output in <project>/logs/job.log,
pid in <project>/job.json, so it keeps running if the browser closes or the server restarts.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import project as projects
from .project import Project


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # a finished child we started stays a zombie until reaped
        return os.waitpid(pid, os.WNOHANG) == (0, 0)
    except ChildProcessError:
        return True  # not our child (server restarted): kill(0) succeeded, so it runs


def status(p: Project) -> dict:
    f = p.path("job.json")
    if not f.exists():
        return {"running": False}
    job = json.loads(f.read_text())
    job["running"] = _alive(job["pid"])
    return job


def running_anywhere() -> Project | None:
    home = projects.projects_home()
    for f in home.glob("*/job.json") if home.exists() else []:
        p = projects.load(f.parent)
        if status(p)["running"]:
            return p
    return None


def start(p: Project, args: list[str], title: str) -> dict:
    busy = running_anywhere()
    if busy is not None:
        raise RuntimeError(f"a job is already running for {busy.name}")
    log = p.path("logs", "job.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {title}\n")
        f.flush()
        proc = subprocess.Popen(
            [sys.executable, "-m", "wakesmith", *args],
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "2"},
        )
    job = {"pid": proc.pid, "args": args, "title": title, "started": time.time()}
    p.path("job.json").write_text(json.dumps(job))
    return {**job, "running": True}


def stop(p: Project) -> bool:
    job = status(p)
    if not job.get("running"):
        return False
    os.killpg(job["pid"], signal.SIGTERM)  # the whole session: TTS workers, trainers
    return True


_TQDM = re.compile(r"([^\r\n|]*?):\s*(\d+)%\|[^\r\n]*?(\d+)/(\d+)")


def progress(p: Project) -> dict:
    """Best-effort live progress for the running stage, from the logs it writes."""
    out: dict = {}
    for part in ("positive", "negative"):
        f = p.path("logs", f"generate-{part}.json")
        if f.exists():
            try:
                out[f"generate-{part}"] = json.loads(f.read_text())
            except json.JSONDecodeError:
                pass
    mlog = p.path("logs", "mww-train.log")
    if mlog.exists():
        steps = re.findall(r"Step #(\d+):", _tail(mlog, 20000))
        if steps:
            out["mww-train"] = {"done": int(steps[-1]), "total": p.microwakeword.steps}
    tlog = p.path("logs", "train.log")
    if tlog.exists():
        m = re.findall(r"oww phase (\d) step (\d+)/(\d+)", _tail(tlog, 4000))
        if m:
            ph, s, t = m[-1]
            out["oww-train"] = {"phase": int(ph), "done": int(s), "total": int(t)}
    jlog = p.path("logs", "job.log")
    if jlog.exists():
        bars = _TQDM.findall(_tail(jlog, 4000))
        if bars:
            desc, pct, done, total = bars[-1]
            out["bar"] = {
                "desc": desc.strip(),
                "percent": int(pct),
                "done": int(done),
                "total": int(total),
            }
    return out


def _tail(path: Path, n: int) -> str:
    with open(path, "rb") as f:
        f.seek(max(0, path.stat().st_size - n))
        return f.read().decode("utf-8", "replace")
