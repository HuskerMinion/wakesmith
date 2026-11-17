"""New version: the next round of a wake word in one step.

A round by hand would be: start from the latest version's recipe and data, bring in the device clips
that came since, label the clear false alarms (Whisper heard none of the phrase's words; the rest wait
for a person), hold out the newest day so the test has false alarms no version trained on, and train
from the recordings on. This does it in one step. The synthetic clips are shared, never made again, so a
version costs the training only. The version before stays as it was; compare them on Results or by the
verdict.

  create(p)   make <word>_v<N> next to p (settings copied, clips linked, device clips carried over)
  prepare(q)  sync new device clips, label the clear false alarms, hold out the newest day
"""

from __future__ import annotations

import fcntl
import json
import os
import secrets
import shutil
import threading
import time
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import clipsync, recordings, rounds
from .project import Project, all_projects, assign_versions, projects_home

HOLD_DAYS = 1.0
MIN_HELD_SHARE = 0.15  # hold out at least this share of the labeled device clips


def _name(word: str, n: int) -> str:
    taken = {q.name for q in all_projects()}
    name, k = f"{word}_v{n}", 1
    while name in taken or (projects_home() / name).exists():
        k += 1
        name = f"{word}_v{n}_{k}"
    return name


def _link_audio(src: str, dst: str) -> None:
    """Audio is hard-linked (the same takes, no extra disk); every other file is copied, so a label
    file written in one version never changes another. On another disk it is copied too."""
    if src.endswith(".wav"):
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


_creating = threading.Lock()  # one New version at a time in this process (the numbering) ...
STALE_S = 86400  # a .<name>.making.* folder this old was left by a killed New version


def create(p: Project, note: str = "") -> Project:
    """A new version of p's wake word from p: same settings and recordings, the same synthetic clips
    (linked), p's takes recorded on the page and its device clips with their labels (both hard-linked, so
    they stay if p is deleted). Built in a temporary folder of its own and moved into place at the end, so
    a failure leaves nothing behind. Nothing is trained yet."""
    home = projects_home()
    home.mkdir(parents=True, exist_ok=True)
    if not p.dir.is_dir():
        raise FileNotFoundError(f"{p.name} was deleted")
    with _creating, open(home / ".versions.lock", "a") as fd:  # ... and across processes
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            held = True
        except OSError:  # a share without flock: this process's lock only, as the other locks do
            held = False
        try:
            _sweep(home)
            return _create(p, note)
        finally:
            if held:
                fcntl.flock(fd, fcntl.LOCK_UN)


def _sweep(home: Path) -> None:
    """Remove what a New version killed mid-way (SIGKILL, power) left behind."""
    for d in home.glob(".*.making.*"):
        try:
            if d.is_dir() and time.time() - d.stat().st_mtime > STALE_S:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


def _create(p: Project, note: str) -> Project:
    kin = [q for q in all_projects() if q.family == p.family]
    assign_versions(kin)
    p = next((q for q in kin if q.name == p.name), p)  # as numbered and named just now
    n = max((q.version for q in kin), default=0) + 1
    q = replace(
        p,
        name=_name(p.family, n),
        version=n,
        note=note or f"v{p.version} + newer device clips",
        created=time.time(),
    )
    final = projects_home() / q.name
    if final.exists():  # made by hand, or by an older wakesmith meanwhile
        raise FileExistsError(f"{q.name} exists already; try again")
    while True:  # a folder of its own; a plain mkdir, so it gets the usual permissions (umask)
        q.dir = projects_home() / f".{q.name}.making.{secrets.token_hex(4)}"
        try:
            q.dir.mkdir()
            break
        except FileExistsError:
            continue
    try:
        done = {
            k: v
            for k, v in p.state().items()
            if k in ("generate", "check-clips", "conversational") and v.get("status") == "done"
        }
        if done:
            q.set_state(**done)
        # under p's locks (index, then recordings, as an import takes them): a sync or the page
        # adding a clip or take meanwhile lands either wholly in this copy or wholly after it
        with rounds.index_lock(p), recordings.locked(p):
            if p.path(
                "recordings"
            ).is_dir():  # takes from the page and devices: q's own (hard links)
                shutil.copytree(
                    p.path("recordings"), q.path("recordings"), copy_function=_link_audio
                )
                for kind in ("pos", "neg"):
                    mine = p.path("recordings", kind).resolve()
                    attr = "positives" if kind == "pos" else "negatives"
                    v = getattr(p.real, attr)
                    if v and Path(v).expanduser().resolve() == mine:
                        setattr(q.real, attr, str(final / "recordings" / kind))
            if p.path(
                "devices", "clips.json"
            ).exists():  # the clips and their labels; files hard-linked
                q.path("devices").mkdir(exist_ok=True)
                if p.path("devices", "inbox").is_dir():
                    shutil.copytree(
                        p.path("devices", "inbox"),
                        q.path("devices", "inbox"),
                        copy_function=_link_audio,
                    )
                shutil.copy2(p.path("devices", "clips.json"), q.path("devices", "clips.json"))
            # deleted clips stay deleted, and device takes already in recordings/ (copied above) don't
            # come in a second time
            for keep in ("deleted.json", "takes.json", "take_runs.json", "take_sessions.json"):
                if p.path("devices", keep).exists():
                    q.path("devices").mkdir(exist_ok=True)
                    shutil.copy2(p.path("devices", keep), q.path("devices", keep))
            # made once and shared (generate never writes into them again): linked last, right
            # before the move, so p being deleted meanwhile fails here instead of leaving dangling links
            if not p.dir.is_dir():
                raise FileNotFoundError(f"{p.name} was deleted while its new version was made")
            for sub in ("clips", "conversational"):
                src = p.path(sub)
                if src.exists():
                    os.symlink(src.resolve(), q.path(sub))
        q.save()
        try:
            q.dir.rename(final)
        except OSError:
            if final.exists():
                raise FileExistsError(f"{q.name} exists already; try again") from None
            raise
    except BaseException:
        shutil.rmtree(q.dir, ignore_errors=True)
        raise
    q.dir = final
    return q


def hold_from(clips: list[dict], days: float = HOLD_DAYS) -> str | None:
    """Where the held-out device clips start: the newest `days`, or more when that's under
    MIN_HELD_SHARE of the labeled clips."""
    times = sorted(c["at"] for c in clips if c.get("label") in ("wake", "not_wake"))
    if not times:
        return None
    newest = datetime.fromisoformat(times[-1].replace("Z", "+00:00"))
    by_day = (newest - timedelta(days=days)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M")
    by_share = times[int(len(times) * (1 - MIN_HELD_SHARE))][:16]
    return min(by_day, by_share)


def prepare(q: Project, log=print, device: str = "auto") -> dict:
    """Bring in new device clips, label the clear ones, hold out the newest day. Returns what it did."""
    out: dict = {}
    if q.devices.source:
        try:
            s = clipsync.sync(q, device=device)
            out["synced"] = {k: s[k] for k in ("new", "not_wake", "wake", "unclear", "to_check")}
            log(clipsync.describe(s))
        except (OSError, ValueError, clipsync.Busy) as e:
            out["sync_error"] = str(e)
            log(f"device clips not synced: {e}")
    out["accepted"] = rounds.accept_suggestions(q) if rounds.load(q) else 0
    held = hold_from(rounds.load(q))
    if held:
        out["holdout_from"] = held
        out["devices"] = rounds.apply(q, held)
        log(f"device clips from {held} UTC on are held out: {json.dumps(out['devices'])}")
    q.path("devices").mkdir(exist_ok=True)
    q.path("devices", "version.json").write_text(json.dumps(out, indent=1))
    return out


def preview(p: Project) -> dict:
    """What a new version from p would start with, for the confirm dialog."""
    clips = rounds.load(p)
    kin = [q for q in all_projects() if q.family == p.family]
    return {
        "from": p.title,
        "version": max((q.version for q in kin), default=0) + 1,
        "device_clips": len(clips),
        "to_check": rounds.to_check(clips),
        "source": p.devices.source,
        "settings": {k: v for k, v in asdict(p).items() if k in ("tts_models",)},
    }
