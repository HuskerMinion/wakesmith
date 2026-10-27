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

import json
import os
import shutil
import time
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import clipsync, rounds
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


def create(p: Project, note: str = "") -> Project:
    """A new version of p's wake word from p: same settings and recordings, the same synthetic clips
    (linked), p's takes recorded on the page and its device clips with their labels (both hard-linked, so
    they stay if p is deleted). Built in a temporary folder and moved into place at the end, so a failure
    leaves nothing behind. Nothing is trained yet."""
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
    q.dir = projects_home() / f".{q.name}.making"
    shutil.rmtree(q.dir, ignore_errors=True)
    try:
        q.dir.mkdir(parents=True)
        # made once and shared: generate never writes into them again
        for sub in ("clips", "conversational"):
            src = p.path(sub)
            if src.exists():
                os.symlink(src.resolve(), q.path(sub))
        done = {
            k: v
            for k, v in p.state().items()
            if k in ("generate", "conversational") and v.get("status") == "done"
        }
        if done:
            q.set_state(**done)
        if p.path("recordings").is_dir():  # takes recorded on the page: q's own (hard links)
            shutil.copytree(p.path("recordings"), q.path("recordings"), copy_function=_link_audio)
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
        q.save()
        q.dir.rename(final)
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
