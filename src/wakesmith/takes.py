"""Takes recorded on a device: a person starts a counted run on the device (N captures, a beep before
each), and the device sends each capture through the device upload as a clip of kind "take". They are
takes of the phrase by definition, so no one needs to listen: the import (rounds.import_folder) puts them
straight into the wake word's recordings (recordings/pos, as if recorded on the page), where they are
trained on and can be held out, renamed to a speaker or deleted like any other take.

A take's sidecar carries "word" (the wake word / model id; matched against devices.model_prefix like a
clip's "model"), "run" (one id per run), "index" (the capture's number in it) and "frame"/"front" (which
audio the device recorded). devices/takes.json remembers which ones came in, so each comes in once.

A run is trained on or held out as a test set as a whole (devices.takes): a run recorded on purpose
keeps every capture with speech in it (silent ones are refused), so a held-out run counts misses
honestly, which clips saved when a model fired cannot. Each run's role is set when its first take comes in (devices/take_runs.json) and stays.
"""

from __future__ import annotations

import io
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from . import recordings
from .project import Project
from .real import SR, read_16k_mono

KIND = "take"


def _index(p: Project) -> Path:
    return p.path("devices", "takes.json")


def done(p: Project) -> dict[str, str]:
    """{take id (device/name.wav): the recordings file it became, "" for one refused}."""
    f = _index(p)
    try:
        d = json.loads(f.read_text()) if f.exists() else {}
        return d if isinstance(d, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


MODES = ("alternate", "train", "test")


def _runs_path(p: Project) -> Path:
    return p.path("devices", "take_runs.json")


def runs(p: Project) -> dict[str, str]:
    """{run key: "train" | "test"}, in the order the runs came in."""
    f = _runs_path(p)
    try:
        d = json.loads(f.read_text()) if f.exists() else {}
        return d if isinstance(d, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


SESSION_GAP_S = (
    600  # takes without a run id: a new run after this long since the device's last take
)


def _sessions_path(p: Project) -> Path:
    return p.path("devices", "take_sessions.json")


def _load(f: Path) -> dict:
    try:
        d = json.loads(f.read_text()) if f.exists() else {}
        return d if isinstance(d, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _secs(at: str) -> float | None:
    try:
        return datetime.fromisoformat(at.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


KEEP_SPANS = 20  # recent runs without an id remembered per device (a late take finds its own)


def _spans(sessions: dict, device: str) -> list[dict]:
    s = sessions.get(device)
    if isinstance(s, dict):  # one span, as the first 0.4.13 builds wrote it
        s = [s]
    return [x for x in s if isinstance(x, dict)] if isinstance(s, list) else []


def run_key(device: str, meta: dict, at: str = "", sessions: dict | None = None) -> str:
    """A run's key: the device's run id; with none sent, the device and the time its run began, a run
    going on while its takes come no more than SESSION_GAP_S apart (sessions: {device: [{"key",
    "first", "last"}, ...]}, the device's recent runs; read only: note() records a take once it is in).
    A take that arrives late or out of order joins the run whose span it falls in."""
    run = str(meta.get("run") or "").strip()[:64]  # a device's id; never an unbounded key
    if run:
        return f"{device}/{run}"
    at = at or str(meta.get("at") or "")
    now = _secs(at)
    if now is not None:
        for s in reversed(_spans(sessions or {}, device)):  # the newest run first
            first, last = _secs(str(s.get("first") or s.get("last"))), _secs(str(s.get("last")))
            if (
                first is not None
                and last is not None
                and first - SESSION_GAP_S <= now <= last + SESSION_GAP_S
            ):
                return str(s.get("key"))
    return f"{device}/{at[:16]}"


def note(sessions: dict, device: str, key: str, at: str) -> None:
    """Record a take without a run id that came in (added or refused): its run's span grows to it, a
    new run is added; other runs stay as they were."""
    spans = _spans(sessions, device)
    s = next((x for x in spans if x.get("key") == key), None)
    if s is None:
        s = {"key": key, "first": at, "last": at}
        spans.append(s)
    else:
        now = _secs(at)
        if now is not None:
            if now < (_secs(str(s.get("first") or s.get("last"))) or now):
                s["first"] = at
            if now > (_secs(str(s.get("last"))) or now):
                s["last"] = at
    # the most recent runs, and always the one just noted (a late, older run stays whole)
    spans.sort(key=lambda x: _secs(str(x.get("last"))) or 0.0)
    keep = [x for x in spans if x is not s][-(KEEP_SPANS - 1) :] + [s]
    sessions[device] = sorted(keep, key=lambda x: _secs(str(x.get("last"))) or 0.0)


def next_role(p: Project, known: dict[str, str]) -> str:
    """The role a new run gets: in "alternate", the first run is the test and then the opposite of the
    last run given a role; otherwise the mode itself."""
    mode = p.devices.takes if p.devices.takes in MODES else "alternate"
    if mode != "alternate":
        return mode
    last = next(reversed(known.values()), None) if known else None
    return "train" if last == "test" else "test"


def take_time(wav: Path, meta: dict) -> str:
    """When the take was recorded: its sidecar's "at", else the file's time (as a clip's import does)."""
    at = str(meta.get("at") or "")
    if at:
        return at
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(wav.stat().st_mtime))
    except OSError:
        return ""


def is_take(meta: dict, rel: Path) -> bool:
    """A take by its sidecar, or (no sidecar kind) by the inbox's <device>/take/ folder."""
    kind = meta.get("kind")
    return kind == KIND if kind else KIND in rel.parts[:-1]


def word(meta: dict) -> str:
    return str(meta.get("model") or meta.get("word") or "")


def condition(device: str, meta: dict) -> str:
    """What the recordings list shows for the take: the device, and run and capture when given."""
    bits = [f"device take: {device}"]
    if meta.get("run"):
        bits.append(f"run {str(meta['run'])[:24]}")
    if isinstance(meta.get("index"), int) and not isinstance(meta.get("index"), bool):
        bits.append(f"#{meta['index']}")
    frame = meta.get("frame") or meta.get("front")
    if frame:
        bits.append(str(frame)[:12])
    return ", ".join(bits)


def add(p: Project, items: list[tuple[str, str, Path, dict]]) -> tuple[int, int, int]:
    """Bring in takes, [(id, device, wav, meta)], oldest first. Returns (added, unreadable, refused): a
    WAV that can't be read is counted and tried again next time; one recordings refuses (silent, too
    short or long) is counted once and remembered (as "") so it isn't tried again."""
    seen, roles, sessions = done(p), runs(p), _load(_sessions_path(p))
    added = unreadable = refused = 0
    for cid, device, wav, meta in items:
        if cid in seen:
            continue
        at = take_time(wav, meta)
        key = run_key(device, meta, at, sessions)
        role = roles.get(key) or next_role(p, roles)  # a run gets its role with its first take in
        try:
            a = read_16k_mono(wav)
            buf = io.BytesIO()
            sf.write(
                buf,
                np.clip(np.round(a * 32768), -32768, 32767).astype(np.int16),  # 16 kHz: exact
                SR,
                format="WAV",
                subtype="PCM_16",
            )
            r = recordings.add(
                p,
                "pos",
                buf.getvalue(),
                condition=condition(device, meta),
                heldout=role == "test",
                choose=False,
            )
        except recordings.LabelsError:  # recordings' labels.csv needs a person: try again after
            unreadable += 1
            continue
        except ValueError:  # silent, too short or long, an odd rate: the same next time
            seen[cid] = ""
            refused += 1
            _in_run(p, sessions, device, key, at, meta)
        except (RuntimeError, OSError):  # unreadable, maybe still being written: next time
            unreadable += 1
            continue
        else:
            seen[cid] = r["file"]
            added += 1
            _in_run(p, sessions, device, key, at, meta)
            if key not in roles:
                roles[key] = role
                _write_json(_runs_path(p), roles)
        # after each one: interrupted, at most the take being added comes in twice
        _write_json(_index(p), seen)
    return added, unreadable, refused


def _in_run(p: Project, sessions: dict, device: str, key: str, at: str, meta: dict) -> None:
    if not str(meta.get("run") or "").strip():  # only runs without an id are tracked by time
        note(sessions, device, key, at)
        _write_json(_sessions_path(p), sessions)


def _write_json(f: Path, data: dict) -> None:
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".part")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(f)
