"""Background jobs for the web UI: one at a time (a training run uses the whole machine).

A job is `python -m wakesmith <args>`, output in <project>/logs/job.log, pid in <project>/job.json. It runs
outside the server's process tree and cgroup (a systemd user unit when there is a user manager, else a
detached session), so it keeps running when the browser closes or the server is stopped or restarted,
even by a tool that kills a whole process tree. The kernel's out-of-memory killer picks it first
(_jobrun.OOM_SCORE_ADJ), before the desktop or other services.

More runs wait in a queue (queue.json next to the projects) and start one after another: the launcher
(_jobrun) records how its job ended in jobs-history.json and starts the next queued run itself, so the
queue moves on while the server is stopped. The server also kicks it at startup and when a run is added.
A file lock (queue.lock) keeps two processes from starting jobs at once.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from . import mww
from . import project as projects
from .project import Project

STARTING_S = 30  # a job that hasn't noted its pid yet counts as running for this long
HISTORY_KEEP = 20  # finished jobs remembered for the UI's notices
LAUNCHER = 2  # job.json written for a launcher that records how the job ended (older ones don't)


def max_jobs() -> int:
    """Jobs allowed at once across all projects (WAKESMITH_MAX_JOBS, default 1: a training run can
    fill a GPU and most of the RAM, and two at once can run the machine out of memory)."""
    try:
        return max(1, int(os.environ.get("WAKESMITH_MAX_JOBS", "1")))
    except ValueError:
        return 1


def _file(name: str) -> Path:
    return projects.projects_home() / name


@contextmanager
def _locked():
    f = _file("queue.lock")
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(f, "a") as fd:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)


def _read(f: Path, default):
    try:
        return json.loads(f.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def _write(f: Path, data) -> None:
    tmp = f.with_name(f.name + ".part")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, f)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # a finished child we started stays a zombie until reaped
        if os.waitpid(pid, os.WNOHANG) != (0, 0):
            return False
    except ChildProcessError:
        pass  # not our child (detached, or the server restarted)
    try:  # the pid may have been reused by something else since the job ended
        return b"wakesmith" in Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return True  # no /proc: trust kill(0)


def status(p: Project) -> dict:
    f = p.path("job.json")
    if not f.exists():
        return {"running": False}
    try:
        job = json.loads(f.read_text())
    except json.JSONDecodeError:
        return {"running": False}
    if job.get("finished"):  # the launcher noted how it ended
        job["running"] = False
    elif job.get("pid"):
        job["running"] = _alive(job["pid"])
    else:  # launched, not yet started
        job["running"] = time.time() - job.get("started", 0) < STARTING_S
    return job


STOPPED = (
    "stopped before it finished: the training process ended (stopped, killed for lack of memory, "
    "or the machine restarted). See the log; Train resumes from here."
)


def stages(p: Project, names) -> dict:
    """Stage states for display. A stage still marked running when no job is alive was cut off
    (out of memory, a reboot, Stop): it shows as failed instead of running forever."""
    st = p.state()
    out = {s: st.get(s) for s in names}
    if not status(p)["running"]:
        for s, v in out.items():
            if v and v.get("status") == "running" and not (v.get("pid") and _alive(v["pid"])):
                out[s] = {**v, "status": "failed", "error": STOPPED}
    return out


def outside_run(p: Project) -> bool:
    """A run started outside the page (the command line, a script) is training this project now."""
    return any(
        isinstance(v, dict) and v.get("status") == "running" and v.get("pid") and _alive(v["pid"])
        for v in p.state().values()
    )


def _with_jobs() -> list[Project]:
    home = projects.projects_home()
    out = []
    for f in sorted(home.glob("*/job.json")) if home.exists() else []:
        try:
            out.append(projects.load(f.parent))
        except (OSError, ValueError):
            continue
    return out


def running_all() -> list[Project]:
    """Projects training now: runs started from the page, and runs started outside it (the command
    line, a script), so neither starts a second job on a busy project or past the job limit."""
    ui = [p for p in _with_jobs() if status(p)["running"]]
    names = {p.name for p in ui}
    return ui + [p for p in projects.all_projects() if p.name not in names and outside_run(p)]


def running_anywhere() -> Project | None:
    busy = running_all()
    return busy[0] if busy else None


def _systemd_user() -> bool:
    """A systemd user manager we can start units under (not in a container without one)."""
    if os.environ.get("WAKESMITH_JOB_SYSTEMD", "") == "0" or not shutil.which("systemd-run"):
        return False
    r = subprocess.run(
        ["systemctl", "--user", "is-system-running"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return r.stdout.strip() in ("running", "degraded")


def start(p: Project, args: list[str], title: str) -> dict:
    with _locked():
        return _start(p, args, title)


def _start(p: Project, args: list[str], title: str) -> dict:
    busy = running_all()
    if p.name in [b.name for b in busy]:
        raise RuntimeError(f"a job is already running for {p.name}")
    if len(busy) >= max_jobs():
        raise RuntimeError(f"a job is already running for {busy[0].name}")
    log = p.path("logs", "job.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {title}\n")
    job_file = p.path("job.json")
    job = {
        "pid": None,
        "args": args,
        "title": title,
        "started": time.time(),
        "project": p.name,
        "label": p.title,
        "launcher": LAUNCHER,
    }
    env = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "2"}
    cmd = [sys.executable, "-m", "wakesmith._jobrun", str(job_file), *args]
    if _systemd_user():
        unit = f"wakesmith-{re.sub(r'[^A-Za-z0-9_-]', '-', p.name)}-{int(job['started'])}"
        job["unit"] = unit
        job_file.write_text(json.dumps(job))
        setenv = [f"--setenv={k}={v}" for k, v in env.items() if "\n" not in v and "=" not in k]
        subprocess.run(
            [
                "systemd-run",
                "--user",
                "--quiet",
                "--collect",
                f"--unit={unit}",
                f"--working-directory={Path.cwd()}",
                "--property=KillMode=control-group",
                f"--property=StandardOutput=append:{log}",
                f"--property=StandardError=append:{log}",
                *setenv,
                *cmd,
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
    else:
        job_file.write_text(json.dumps(job))
        with open(
            log, "a"
        ) as f:  # setsid -f: the job is not our child, so no process tree holds it
            subprocess.run(
                ["setsid", "-f", *cmd],
                stdout=f,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
                check=True,
            )
    for _ in range(100):  # the job notes its pid as it starts
        cur = json.loads(job_file.read_text())
        if cur.get("pid"):
            return {**cur, "running": True}
        time.sleep(0.1)
    return {**job, "running": True}


def stop(p: Project) -> bool:
    job = status(p)
    if not job.get("running"):
        return False
    with _locked():  # so the launcher records "stopped", not "failed"
        cur = _read(p.path("job.json"), None)
        if isinstance(cur, dict) and not cur.get("finished"):
            _write(p.path("job.json"), {**cur, "stop_requested": True})
    if job.get("unit"):  # the whole unit: TTS workers, trainers, everything they started
        subprocess.run(
            ["systemctl", "--user", "stop", job["unit"]],
            check=False,
            capture_output=True,
            timeout=60,
        )
    elif job.get("pid"):
        os.killpg(job["pid"], signal.SIGTERM)  # the whole session: TTS workers, trainers
    return True


# -- queue and finished-job history ---------------------------------------------------------------
def queue() -> list[dict]:
    """Waiting runs, next first: [{id, project, label, args, title, queued_at}]."""
    q = _read(_file("queue.json"), [])
    return q if isinstance(q, list) else []


def enqueue(p: Project, args: list[str], title: str) -> dict:
    """Add a run to the end of the queue; it starts now if nothing is running."""
    item = {
        "id": secrets.token_hex(4),
        "project": p.name,
        "label": p.title,
        "args": args,
        "title": title,
        "queued_at": time.time(),
    }
    with _locked():
        _write(_file("queue.json"), [*queue(), item])
    kick()
    return item


def dequeue(item_id: str) -> bool:
    with _locked():
        q = queue()
        left = [i for i in q if i.get("id") != item_id]
        _write(_file("queue.json"), left)
    return len(left) < len(q)


def drop_project(name: str) -> None:
    """Forget the queued runs of a project (it is being deleted)."""
    with _locked():
        _write(_file("queue.json"), [i for i in queue() if i.get("project") != name])


def move(item_id: str, to: int) -> bool:
    with _locked():
        q = queue()
        item = next((i for i in q if i.get("id") == item_id), None)
        if item is None:
            return False
        q.remove(item)
        q.insert(max(0, min(to, len(q))), item)
        _write(_file("queue.json"), q)
    return True


def history() -> list[dict]:
    """The last HISTORY_KEEP finished jobs, oldest first: {project, label, title, result (ok, failed,
    stopped), exit, error, started, finished}."""
    h = _read(_file("jobs-history.json"), [])
    return h if isinstance(h, list) else []


def _record(entry: dict) -> None:
    _write(_file("jobs-history.json"), [*history(), entry][-HISTORY_KEEP:])


def _why(code: int | None) -> str:
    if code is None:
        return "the job ended without reporting back (killed, or the machine restarted)"
    if code == -signal.SIGKILL:
        return "killed: most likely the machine ran out of memory"
    if code < 0:
        return f"ended by signal {-code}"
    return f"ended with an error (exit code {code}); see the log"


def _finish(job_file: Path, job: dict, code: int | None) -> None:
    result = "stopped" if job.get("stop_requested") else "ok" if code == 0 else "failed"
    job = {**job, "finished": time.time(), "exit": code, "result": result}
    _write(job_file, job)
    _record(
        {
            "project": job.get("project") or job_file.parent.name,
            "label": job.get("label") or job.get("project") or job_file.parent.name,
            "title": job.get("title", ""),
            "result": result,
            "exit": code,
            "error": "" if result != "failed" else _why(code),
            "started": job.get("started"),
            "finished": job["finished"],
        }
    )


def finished(job_file: str | Path, code: int) -> None:
    """Called by the launcher when its job's process has exited: note how, then start the next run."""
    job_file = Path(job_file)
    with _locked():
        job = _read(job_file, {})
        if isinstance(job, dict) and not job.get("finished"):
            _finish(job_file, job, code)
    kick()


def _reconcile() -> None:
    """A job whose launcher died with it (SIGKILL, a reboot) never reported: note it as ended."""
    for p in _with_jobs():
        job = _read(p.path("job.json"), None)
        if (
            isinstance(job, dict)
            and job.get("launcher") == LAUNCHER
            and not job.get("finished")
            and not status(p)["running"]
        ):
            _finish(p.path("job.json"), job, None)


def kick() -> list[dict]:
    """Start queued runs while there is room. Safe to call from any process at any time."""
    started = []
    with _locked():
        _reconcile()
        while True:
            q = queue()
            busy = running_all()
            if not q or len(busy) >= max_jobs():
                break
            names = {b.name for b in busy}
            item = next((i for i in q if i.get("project") not in names), None)
            if item is None:
                break
            _write(_file("queue.json"), [i for i in q if i is not item])
            try:
                started.append(_start(projects.load(item["project"]), item["args"], item["title"]))
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as e:
                now = time.time()
                _record(
                    {
                        "project": item.get("project", ""),
                        "label": item.get("label") or item.get("project", ""),
                        "title": item.get("title", ""),
                        "result": "failed",
                        "exit": None,
                        "error": f"could not start: {e}",
                        "started": now,
                        "finished": now,
                    }
                )
    return started


_TQDM = re.compile(r"([^\r\n|]*?):\s*(\d+)%\|[^\r\n]*?(\d+)/(\d+)")


def progress(p: Project) -> dict:
    """Best-effort live progress for the running stage, from the logs it writes."""
    out: dict = {}
    for part in ("positive", "negative", "conversational"):
        f = p.path("logs", f"generate-{part}.json")
        if f.exists():
            try:
                out[f"generate-{part}"] = json.loads(f.read_text())
            except json.JSONDecodeError:
                pass
    mlog = p.path("logs", "mww-train.log")
    if mlog.exists():
        # microWakeWord prints "Validation Batch #k ... Mini-Batch #m" (\r-terminated) every step
        m = re.findall(r"Validation Batch #(\d+):[^\r\n]*Mini-Batch #(\d+)", _tail(mlog, 4000))
        if m:
            k, mini = m[-1]
            done = (int(k) - 1) * mww.EVAL_INTERVAL + int(mini)
            total = p.microwakeword.steps + p.microwakeword.finetune_steps
            jlog = p.path("logs", "job.log")  # best of N: the candidate being trained in this job
            this_job = _tail(jlog, 20000).rsplit("\n===== ", 1)[-1] if jlog.exists() else ""
            c = re.findall(r"mww candidate (\d+) of (\d+)", this_job)
            if c:
                i, n = int(c[-1][0]), int(c[-1][1])
                done, total = (i - 1) * total + done, n * total
            out["mww-train"] = {"done": done, "total": total}
    olog = p.path("logs", "oww-progress.json")
    if olog.exists():
        try:
            out["oww-train"] = json.loads(olog.read_text())
        except json.JSONDecodeError:
            pass
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
