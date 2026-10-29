"""Archive: free a project's disk space by removing only what training rebuilds by itself.

A trained project holds ~20 GB, almost all of it augmented training features (mww/features, rebuilt in a
minute or two) and prepared copies of the real recordings (real/, real_neg/: rebuilt by the "real" stage
from the source folders, which live outside the project and are never touched). Archiving removes:

  mww/features, oww/features     training features            -> mww-features / oww-features rerun
  real/, real_neg/, commonvoice/ prepared recordings           -> real (and commonvoice) rerun
  mww/candidates/s<N>/ extras     best-of-N runs' training logs and checkpoints, except the small files
                                  needed to average or compare candidates later (KEEP_CANDIDATE); the
                                  same in mww/candidates.previous (mww/combined.previous stays)
  mww/trained.previous            the model before the last retrain
  *.tmp, *.part                   leftovers of interrupted writes
  clips/ (only with clips=True)   the project's own synthetic clips -> generate rerun (slow: speech synthesis)

Nothing is removed through a symlink, and nothing another project links to (rounds share clips, some
projects share another's prepared recordings or features).

Everything else stays: project.yaml, state.json, logs, results, runs, export, models.json, devices,
review verdicts, recorded takes, everyday sentences, the trained model (mww/trained, oww/model).

The stages whose output was removed are marked "archived" in state.json; pipeline.run reruns them first
when a later stage needs their output, so Train / Retrain just work. <project>/archive.json says when and
how much was freed.

Auto rule (app setting archive_after_days, 0 = off): a project idle that long, with no model offered or
live in models.json and no job running or queued, is archived (never its clips).

Sizes come from a du-like walk that never follows symlinks, cached per folder by its mtime, so the
Storage page stays quick over many 20 GB projects.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
import threading
import time
from pathlib import Path

from . import assistant, jobs, models
from . import project as projects
from .candidates import KEEP_CANDIDATE
from .pipeline import STAGES, enabled
from .project import Project
from .real import RealData

ARCHIVE_FILE = "archive.json"
DEFAULT_DAYS = 14
AUTO_MIN_BYTES = 100 * 2**20  # the auto rule leaves projects with less than this to free alone
QUIET = {
    "logs",
    ARCHIVE_FILE,
    ARCHIVE_FILE + ".part",
    ".models.lock",
    "job.json",
}  # changes here don't count as activity


# -- sizes ---------------------------------------------------------------------------------------
_cache: dict[str, tuple] = {}  # dir -> ((ino, mtime_ns), own bytes, newest mtime, subdir names)
_lock = threading.RLock()


def _scan(path: str) -> tuple[int, float]:
    """(bytes, newest mtime) below `path`, like du: symlinks are neither followed nor counted. Each
    folder's own files are cached until the folder's mtime changes (files added, removed, replaced)."""
    try:
        st = os.lstat(path)
    except OSError:
        return 0, 0.0
    if stat.S_ISLNK(st.st_mode):
        return 0, 0.0
    if not stat.S_ISDIR(st.st_mode):
        return st.st_size, st.st_mtime
    key = (st.st_ino, st.st_mtime_ns)
    c = _cache.get(path)
    if c is None or c[0] != key:
        own, newest, subs = 0, st.st_mtime, []
        try:
            with os.scandir(path) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            subs.append(e.name)
                        elif not e.is_symlink():
                            s = e.stat(follow_symlinks=False)
                            own += s.st_size
                            newest = max(newest, s.st_mtime)
                    except OSError:
                        continue
        except OSError:
            pass
        if c is not None:  # forget folders that are gone
            for gone in set(c[3]) - set(subs):
                pre = os.path.join(path, gone)
                for k in [k for k in _cache if k == pre or k.startswith(pre + os.sep)]:
                    del _cache[k]
        c = (key, own, newest, subs)
        _cache[path] = c
    total, newest = c[1], c[2]
    for name in c[3]:
        b, m = _scan(os.path.join(path, name))
        total += b
        newest = max(newest, m)
    return total, newest


def size(path: Path | str) -> int:
    with _lock:
        return _scan(str(path))[0]


# -- what goes ----------------------------------------------------------------------------------
def _safe(p: Project, rel: str) -> bool:
    """`rel` exists inside the project and no part of the way there is a symlink."""
    cur = p.dir
    for part in Path(rel).parts:
        cur = cur / part
        try:
            if stat.S_ISLNK(os.lstat(cur).st_mode):
                return False
        except OSError:
            return False
    return True


def _has_files(d: Path) -> bool:
    try:
        with os.scandir(d) as it:
            return any(True for _ in it)
    except OSError:
        return False


def _clips_ok(p: Project) -> bool:
    """The synthetic clips the features are made from are there (through a link is fine)."""
    return _has_files(p.path("clips", "positive_train"))


LINK_DIRS = (
    "",
    "mww",
    "oww",
)  # where projects link to each other's folders (clips, real, features)


def links(home: Path | None = None) -> dict[str, list[str]]:
    """Where projects' symlinks lead: real path -> the projects holding a link to it. Rounds share
    clips this way, and some projects share another's prepared recordings or features."""
    home = home or projects.projects_home()
    out: dict[str, list[str]] = {}
    for f in projects.project_files(home):
        for sub in LINK_DIRS:
            d = f.parent / sub
            if sub and (d.is_symlink() or not d.is_dir()):
                continue
            try:
                with os.scandir(d) as it:
                    found = [e.path for e in it if e.is_symlink()]
            except OSError:
                continue
            for x in found:
                out.setdefault(os.path.realpath(x), []).append(f.parent.name)
    return out


def _users(p: Project, rel: str, links: dict[str, list[str]]) -> list[str]:
    """Other projects with a link to p's `rel`, into it, or to a folder holding it."""
    t = str(p.dir / rel)
    return sorted(
        {
            n
            for k, names in links.items()
            if k == t or k.startswith(t + os.sep) or t.startswith(k + os.sep)
            for n in names
            if n != p.name
        }
    )


def _clip_info(p: Project, links: dict[str, list[str]]) -> dict:
    c = p.path("clips")
    if not os.path.lexists(c):
        return {"own": False, "link_to": "", "shared_with": []}
    if os.path.islink(c):
        real = os.path.realpath(c)
        owner = Path(real).parent
        return {
            "own": False,
            "link_to": owner.name if (owner / "project.yaml").exists() else real,
            "shared_with": [],
        }
    return {"own": True, "link_to": "", "shared_with": _users(p, "clips", links)}


def plan(p: Project, clips: bool = False, found: dict[str, list[str]] | None = None) -> dict:
    """What archive() would remove: {remove: [{path, bytes, why}], stages, bytes, kept: [notes],
    clips: {own, link_to, shared_with}}. Paths are relative to the project folder."""
    found = links() if found is None else found
    remove: list[dict] = []
    stages: set[str] = set()
    kept: list[str] = []

    def add(rel: str, why: str, *marks: str) -> None:
        if not os.path.lexists(p.dir / rel):
            return
        if not _safe(p, rel):
            kept.append(f"{rel}: a link, left alone")
            return
        if users := _users(p, rel, found):
            kept.append(f"{rel}: kept, {', '.join(users)} link to it")
            return
        remove.append({"path": rel, "bytes": size(p.dir / rel), "why": why})
        stages.update(marks)

    ci = _clip_info(p, found)
    drop_clips = clips and ci["own"] and not ci["shared_with"]
    if clips and ci["own"] and ci["shared_with"]:
        kept.append(f"clips: used by {', '.join(ci['shared_with'])}")
    elif clips and not ci["own"] and ci["link_to"]:
        kept.append(f"clips: a link to {ci['link_to']}'s clips, left alone")
    rebuildable = _clips_ok(p) or drop_clips
    for rel, stage in (("mww/features", "mww-features"), ("oww/features", "oww-features")):
        if rebuildable:
            add(rel, "training features", stage)
        elif os.path.lexists(p.dir / rel):
            kept.append(f"{rel}: kept, the synthetic clips needed to rebuild it are missing")

    # prepared recordings: only when the source folders are still there to rebuild them from
    real = RealData(p)
    cv_on = p.uses_commonvoice
    cv_ok = True
    if cv_on:
        from . import commonvoice

        cv_ok = commonvoice.present(commonvoice.dataset(p))
    for rel, kind in (("real", "wake"), ("real_neg", "not_wake")):
        if not os.path.lexists(p.dir / rel):
            continue
        srcs = real.of_kind(kind)
        if not srcs or not all(real.sources[s].is_dir() for s in srcs):
            kept.append(f"{rel}: kept, its source recordings aren't there to rebuild it")
        elif rel == "real" and cv_on and not cv_ok:
            kept.append("real: kept, it holds Common Voice takes and the dataset isn't there")
        else:
            add(
                rel,
                "prepared copies of your recordings",
                "real",
                *(["commonvoice"] if cv_on else []),
            )
    if cv_on and cv_ok and "real" in stages:
        add("commonvoice", "prepared Common Voice takes", "commonvoice")

    cands = p.path("mww", "candidates")
    if _safe(p, "mww/candidates") and cands.is_dir():
        try:
            picked = json.loads((cands / "pick.json").read_text()).get("picked")
        except (OSError, ValueError):
            picked = None
        for d in sorted(cands.glob("s*")):
            if d.name == f"s{picked}" or not d.is_dir() or d.is_symlink():
                continue
            for x in sorted(d.iterdir()):
                if x.name not in KEEP_CANDIDATE:
                    add(str(x.relative_to(p.dir)), "candidate's training logs and checkpoints")
    prev = p.path("mww", "candidates.previous")
    if _safe(p, "mww/candidates.previous") and prev.is_dir():
        for d in sorted(prev.glob("s*")):
            if not d.is_dir() or d.is_symlink():
                continue
            for x in sorted(d.iterdir()):
                if x.name not in KEEP_CANDIDATE:
                    add(str(x.relative_to(p.dir)), "previous round's training logs and checkpoints")
    add("mww/trained.previous", "the model before the last retrain")
    for f in sorted(p.dir.iterdir()):
        if f.is_file() and not f.is_symlink() and f.suffix in (".tmp", ".part"):
            add(f.name, "leftover of an interrupted write")
    if drop_clips:
        add("clips", "your own synthetic clips (slow to make again)", "generate")
    st = p.state()
    marks = [
        s
        for s in STAGES
        if s in stages and (enabled(p, s) or st.get(s)) and (s != "commonvoice" or cv_on)
    ]
    return {
        "project": p.name,
        "remove": remove,
        "bytes": sum(r["bytes"] for r in remove),
        "stages": marks,
        "kept": kept,
        "clips": ci,
    }


# -- doing it -----------------------------------------------------------------------------------
class Busy(RuntimeError):
    """The project has a job running or waiting in the queue."""


def _busy(p: Project) -> str:
    if jobs.status(p)["running"]:
        return "a job is running for it"
    if any(i.get("project") == p.name for i in jobs.queue()):
        return "it has a run waiting in the queue"
    return ""


def info(p: Project) -> dict | None:
    """archive.json, plus `archived` (what it removed hasn't been rebuilt yet) and `to_rebuild`."""
    try:
        a = json.loads(p.path(ARCHIVE_FILE).read_text())
    except (OSError, ValueError):
        return None
    still = [
        s for s, v in p.state().items() if isinstance(v, dict) and v.get("status") == "archived"
    ]
    if a.get("stages"):  # archived until those stages have run again
        archived = bool(still)
    else:  # only spare files went: until the next job
        archived = (jobs.status(p).get("started") or 0) < a.get("archived_at", 0)
    return {**a, "archived": archived, "to_rebuild": [s for s in STAGES if s in still]}


def archive(p: Project, clips: bool = False, by: str = "you") -> dict:
    """Remove what plan() lists, mark those stages for a rerun and note it in archive.json."""
    with jobs._locked():  # no queued run of this project can start meanwhile
        if why := _busy(p):
            raise Busy(f"can't archive {p.name}: {why}")
        pl = plan(p, clips)
        for r in pl["remove"]:
            f = p.dir / r["path"]
            if not _safe(p, r["path"]):
                continue
            if f.is_dir():
                shutil.rmtree(f)
            else:
                f.unlink(missing_ok=True)
        now = time.time()
        st = p.state()
        p.set_state(
            **{
                s: {**(st.get(s) or {}), "status": "archived", "archived_at": now}
                for s in pl["stages"]
            }
        )
        old = info(p) or {}
        rec = {
            "archived_at": now,
            "freed_bytes": pl["bytes"],
            "removed": [r["path"] for r in pl["remove"]],
            "stages": pl["stages"],
            "clips": any(r["path"] == "clips" for r in pl["remove"]),
            "by": by,
            "total_freed_bytes": old.get("total_freed_bytes", 0) + pl["bytes"],
        }
        tmp = p.path(ARCHIVE_FILE + ".part")
        tmp.write_text(json.dumps(rec, indent=1))
        tmp.replace(p.path(ARCHIVE_FILE))
    return {**rec, "kept": pl["kept"], "project": p.name}


# -- overview and the auto rule ---------------------------------------------------------------
def settings() -> dict:
    s = assistant._read_all().get("storage") or {}
    try:
        days = max(0, int(s.get("archive_after_days", DEFAULT_DAYS)))
    except (TypeError, ValueError):
        days = DEFAULT_DAYS
    return {"archive_after_days": days}


def save_settings(changes: dict) -> dict:
    new = {}
    if "archive_after_days" in changes:
        try:
            new["archive_after_days"] = max(0, min(3650, int(changes["archive_after_days"])))
        except (TypeError, ValueError):
            raise ValueError("days must be a whole number (0 turns it off)") from None
    assistant.update_all(
        lambda data: data.__setitem__("storage", {**(data.get("storage") or {}), **new})
    )
    return settings()


def _last_activity(p: Project, newest_files: float) -> float:
    job = jobs.status(p)
    return max(newest_files, job.get("started") or 0, job.get("finished") or 0)


def overview(p: Project, found: dict[str, list[str]], queued: set[str], now: float) -> dict:
    with _lock:
        total, newest = 0, 0.0
        with os.scandir(p.dir) as it:
            for e in it:
                b, m = _scan(e.path)
                total += b
                if e.name not in QUIET and not e.is_symlink():
                    newest = max(newest, m)
        pl = plan(p, found=found)
        pl_clips = plan(p, clips=True, found=found)
    last = _last_activity(p, newest)
    statuses = [e.get("status") for e in models.load(p)] if p.path("models.json").exists() else []
    a = info(p)
    return {
        "name": p.name,
        "label": p.title,
        "bytes": total,
        "freeable": pl["bytes"],
        "freeable_with_clips": pl_clips["bytes"],
        "last_activity": last,
        "idle_days": round(max(0.0, now - last) / 86400, 1),
        "live": statuses.count("live"),
        "offered": statuses.count("offered"),
        "running": bool(jobs.status(p)["running"]),
        "queued": p.name in queued,
        "archived": bool(a and a["archived"]),
        "archived_at": a.get("archived_at") if a else None,
        "freed_bytes": a.get("total_freed_bytes") if a else 0,
        "clips": pl["clips"],
        "kept": pl["kept"],
    }


def storage(home: Path | None = None) -> list[dict]:
    """One row per project (see overview), largest first."""
    home = home or projects.projects_home()
    found = links(home)
    queued = {str(i.get("project", "")) for i in jobs.queue()}
    now = time.time()
    rows = []
    for f in projects.project_files(home):
        try:
            p = projects.load(f.parent)
        except (OSError, ValueError):
            continue
        rows.append(overview(p, found, queued, now))
    return sorted(rows, key=lambda r: -r["bytes"])


def eligible(row: dict, days: int) -> str:
    """Why the auto rule leaves this project alone ("" = archive it)."""
    if days <= 0:
        return "the auto rule is off"
    if row["live"] or row["offered"]:
        return "a model of it is offered or live"
    if row["running"] or row["queued"]:
        return "it has a job running or queued"
    if row["idle_days"] < days:
        return f"used {row['idle_days']:.0f} days ago"
    if row["freeable"] < AUTO_MIN_BYTES:
        return "nothing much to free"
    return ""


def _log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(f"wakesmith: {msg}", file=sys.stderr)
    try:
        with open(projects.projects_home() / "archive.log", "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def archive_idle(days: int | None = None, by: str = "auto") -> list[dict]:
    """Archive every project the rule picks (never its clips). Returns what was done."""
    days = settings()["archive_after_days"] if days is None else days
    rows = storage()  # also fills the size cache, so the Storage page opens quickly
    done: list[dict] = []
    if days <= 0:
        return done
    for row in rows:
        if eligible(row, days):
            continue
        try:
            r = archive(projects.load(row["name"]), clips=False, by=by)
        except (Busy, OSError, ValueError) as e:
            _log(f"archive of {row['name']} skipped: {e}")
            continue
        _log(
            f"archived {row['name']} ({by}, idle {row['idle_days']:.0f} days): freed "
            f"{r['freed_bytes'] / 1e9:.1f} GB ({', '.join(r['removed']) or 'nothing'})"
        )
        done.append(r)
    return done
