"""Routine device-clip sync: bring in a word's new clips and transcribe them, so a person only listens
to the unclear ones.

`wakesmith clips sync PROJECT` imports the clips in the project's devices.source folder whose sidecar
"model" starts with devices.model_prefix (a puller folder holds every word's fires), skips the ones
already in the project, and runs Whisper on the new ones only (rounds.py). `clips sync-all` does it for
every project with a source folder; `clips schedule` installs a systemd user timer that runs sync-all
once a day. Each run appends to <projects>/clips-sync.log and leaves its result in
<project>/devices/sync.json for the Device clips page.

Whisper device "auto": the GPU only while no other wakesmith job is running (training can fill a card),
else the CPU. A project whose own job is running (training reads its device clips) is skipped.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from . import jobs, rounds
from . import project as projects
from .project import Project

UNIT = "wakesmith-clips-sync"
LOG_KEEP = 200_000  # bytes of clips-sync.log kept (oldest lines dropped)


class Busy(RuntimeError):
    """The project's own job is running, or another sync of it is."""


def log_file() -> Path:
    return projects.projects_home() / "clips-sync.log"


def log(line: str) -> None:
    f = log_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(f, "a") as fd:
        fd.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")
    if f.stat().st_size > 2 * LOG_KEEP:
        tail = f.read_bytes()[-LOG_KEEP:]
        f.write_bytes(tail[tail.find(b"\n") + 1 :])


def last_result(p: Project) -> dict | None:
    f = p.path("devices", "sync.json")
    try:
        return json.loads(f.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _own_job_running(p: Project) -> bool:
    """The project's job is running, and it isn't this process (a Sync now job's launcher is our
    parent: _jobrun notes its own pid)."""
    st = jobs.status(p)
    return bool(st.get("running")) and st.get("pid") not in (os.getpid(), os.getppid())


def pick_device(p: Project, want: str = "auto") -> str:
    """Whisper's torch device. "auto": the first GPU when no other project's job is running, else cpu."""
    if want != "auto":
        return want or "cpu"
    if [b for b in jobs.running_all() if b.name != p.name]:
        return "cpu"
    from . import hardware, mww

    return mww.train_device(hardware.detect_safe()[0].gpus)


@contextmanager
def _locked(p: Project):
    f = p.path("devices", "sync.lock")
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(f, "a") as fd:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Busy(f"{p.name}: another sync is running") from None
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)


def sync(
    p: Project,
    source: str = "",
    model_prefix: str | None = None,
    device: str = "auto",
    since: str | None = None,
    limit: int = 0,
    whisper=None,
    progress=None,
) -> dict:
    """Import the new clips of this project's word and transcribe them. Returns the counts:
    new, wake / not_wake / unclear (suggestions for the new clips), to_check (waiting for a person)."""
    source = source or p.devices.source
    if not source:
        raise ValueError(f"{p.name}: no source folder (set devices.source or pass --from)")
    prefix = p.devices.model_prefix if model_prefix is None else model_prefix
    since = p.devices.since if since is None else since
    if _own_job_running(p):
        raise Busy(f"{p.name}: a job of this wake word is running; sync it after it ends")
    started = time.time()
    with _locked(p):
        imp = rounds.import_folder(p, source, prefix, since, limit)
        # the new clips, plus any of this word an earlier, interrupted sync left untranscribed
        todo = {
            c["id"]
            for c in rounds.load(p)
            if c["text"] is None and str(c.get("model") or "").startswith(prefix)
        }
        dev = ""
        if todo:
            dev = pick_device(p, device)
            rounds.transcribe(p, dev, progress=progress, ids=todo, whisper=whisper)
        auto = rounds.accept_suggestions(p)  # clear "not the phrase": no one needs to listen
        clips = rounds.load(p)
        new = [c for c in clips if c["id"] in set(imp["ids"]) | todo]
        out = {
            "project": p.name,
            "source": str(source),
            "model_prefix": prefix,
            "new": imp["added"],
            "transcribed": len(todo),
            "left": imp["left"],
            "device": dev,
            **{
                k: sum(c.get("suggestion") == k for c in new)
                for k in ("wake", "not_wake", "unclear")
            },
            "auto_labelled": auto,
            "to_check": rounds.to_check(clips),
            "total": len(clips),
            "at": time.time(),
            "seconds": round(time.time() - started, 1),
            "result": "ok",
        }
        _keep(p, out)
    return out


def _keep(p: Project, out: dict) -> None:
    f = p.path("devices", "sync.json")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out))
    log(f"{p.name}: {describe(out)}")


def failed(p: Project, error: str) -> dict:
    out = {"project": p.name, "result": "failed", "error": error, "at": time.time()}
    f = p.path("devices", "sync.json")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out))
    log(f"{p.name}: failed: {error}")
    return out


def describe(r: dict) -> str:
    if r.get("result") == "failed":
        return f"failed: {r.get('error', '')}"
    if r.get("result") == "skipped":
        return f"skipped: {r.get('error', '')}"
    s = f"{r['new']} new clips"
    if r["transcribed"]:
        s += (
            f"; suggested: {r['wake']} the phrase, {r['not_wake']} not the phrase, "
            f"{r['unclear']} unclear (Whisper on {r['device']})"
        )
    if r.get("left"):
        s += f"; {r['left']} more next time"
    return s + f"; {r['to_check']} waiting for a person"


def sync_all(device: str = "auto", limit: int = 0) -> list[dict]:
    """sync() for every project with devices.source set; one failing doesn't stop the rest."""
    home = projects.projects_home()
    out = []
    for f in projects.project_files(home):
        try:
            p = projects.load(f.parent)
        except (OSError, ValueError, TypeError):
            continue
        if not p.devices.source:
            continue
        try:
            out.append(sync(p, device=device, limit=limit))
        except Busy as e:
            log(f"{p.name}: skipped: {e}")
            out.append({"project": p.name, "result": "skipped", "error": str(e)})
        except Exception as e:  # noqa: BLE001 - note it, carry on with the next project
            out.append(failed(p, f"{type(e).__name__}: {e}"))
    if not out:
        log("no wake word has a source folder for device clips")
    return out


# -- the daily timer (systemd user units) ---------------------------------------------------------
def unit_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "systemd" / "user"


def command() -> list[str]:
    """How the timer runs wakesmith: this interpreter (the install's venv) with -m wakesmith."""
    return [sys.executable, "-m", "wakesmith", "clips", "sync-all"]


_TIME = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


def check_time(at: str) -> str:
    m = _TIME.match(at.strip())
    if not m:
        raise ValueError(f"time must be HH:MM (24 h), not {at!r}")
    return f"{int(m[1]):02d}:{m[2]}"


def unit_files(at: str, cmd: list[str], env: dict[str, str], log_path: Path) -> dict[str, str]:
    """The .service and .timer text: sync-all once a day at `at` (HH:MM, local time); a run missed
    while the machine was off happens at the next boot (Persistent)."""
    at = check_time(at)
    envs = "".join(f'Environment="{k}={v}"\n' for k, v in sorted(env.items()))
    service = (
        "[Unit]\n"
        "Description=wakesmith: bring in new device clips and transcribe them\n\n"
        "[Service]\n"
        "Type=oneshot\n"
        f"{envs}"
        f"ExecStart={' '.join(_quote(c) for c in cmd)}\n"
        f"StandardOutput=append:{log_path}\n"
        f"StandardError=append:{log_path}\n"
        "Nice=10\n"
    )
    timer = (
        "[Unit]\n"
        "Description=wakesmith: daily device-clip sync\n\n"
        "[Timer]\n"
        f"OnCalendar=*-*-* {at}:00\n"
        "Persistent=true\n"
        f"Unit={UNIT}.service\n\n"
        "[Install]\n"
        "WantedBy=timers.target\n"
    )
    return {f"{UNIT}.service": service, f"{UNIT}.timer": timer}


def _quote(s: str) -> str:
    return f'"{s}"' if re.search(r"\s", s) else s


def _env() -> dict[str, str]:
    """Where the projects and data are, when set for this process (the timer has a bare environment)."""
    return {k: os.environ[k] for k in ("WAKESMITH_PROJECTS", "WAKESMITH_HOME") if os.environ.get(k)}


def _systemctl(run, *args: str):
    return run(
        ["systemctl", "--user", *args], check=False, capture_output=True, text=True, timeout=30
    )


def install(at: str = "03:30", run=None) -> dict:
    """Write the units, reload the user manager and enable the timer."""
    run = run or subprocess.run
    files = unit_files(at, command(), _env(), log_file())
    d = unit_dir()
    d.mkdir(parents=True, exist_ok=True)
    log_file().parent.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (d / name).write_text(text)
    for args in (("daemon-reload",), ("enable", "--now", f"{UNIT}.timer")):
        r = _systemctl(run, *args)
        if r.returncode:
            raise RuntimeError(
                f"systemctl --user {' '.join(args)}: {(r.stderr or r.stdout).strip()}"
            )
    return status(run)


def remove(run=None) -> dict:
    """Disable the timer and delete the units."""
    run = run or subprocess.run
    _systemctl(run, "disable", "--now", f"{UNIT}.timer")
    for name in (f"{UNIT}.service", f"{UNIT}.timer"):
        (unit_dir() / name).unlink(missing_ok=True)
    _systemctl(run, "daemon-reload")
    return status(run)


def _show(run, unit: str, props: str) -> dict:
    r = _systemctl(run, "show", unit, "-p", props)
    pairs = (line.partition("=") for line in (r.stdout or "").splitlines())
    return {k: v for k, sep, v in pairs if sep}


def status(run=None) -> dict:
    """{installed, enabled, at, next, last_run, last_result, log}: from the unit files and systemd."""
    run = run or subprocess.run
    timer = unit_dir() / f"{UNIT}.timer"
    out: dict = {"installed": timer.exists(), "enabled": False, "at": "", "next": "", "log": []}
    if timer.exists():
        m = re.search(r"OnCalendar=\S+ (\d\d:\d\d)", timer.read_text())
        out["at"] = m[1] if m else ""
        try:
            t = _show(run, f"{UNIT}.timer", "UnitFileState,NextElapseUSecRealtime,LastTriggerUSec")
            s = _show(run, f"{UNIT}.service", "Result,ExecMainStatus,ActiveState")
        except (OSError, subprocess.SubprocessError):
            t, s = {}, {}
        if t or s:
            out["enabled"] = t.get("UnitFileState") == "enabled"
            out["next"] = t.get("NextElapseUSecRealtime", "")
            out["last_run"] = t.get("LastTriggerUSec", "")
            out["last_result"] = s.get("Result", "")
            out["running"] = s.get("ActiveState") == "activating"
    f = log_file()
    if f.exists():
        out["log"] = f.read_text(errors="replace").splitlines()[-10:]
    return out
