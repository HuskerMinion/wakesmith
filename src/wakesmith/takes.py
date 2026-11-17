"""Takes recorded on a device: a person starts a counted run on the device (N captures, a beep before
each), and the device sends each capture through the device upload as a clip of kind "take". They are
takes of the phrase by definition, so no one needs to listen: the import (rounds.import_folder) puts them
straight into the wake word's recordings (recordings/pos, as if recorded on the page), where they are
trained on and can be held out, renamed to a speaker or deleted like any other take.

A take's sidecar carries "word" (the wake word / model id; matched against devices.model_prefix like a
clip's "model"), "run" (one id per run), "index" (the capture's number in it) and "frame"/"front" (which
audio the device recorded). devices/takes.json remembers which ones came in, so each comes in once.
"""

from __future__ import annotations

import io
import json
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
    seen = done(p)
    added = unreadable = refused = 0
    for cid, device, wav, meta in items:
        if cid in seen:
            continue
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
                p, "pos", buf.getvalue(), condition=condition(device, meta), choose=False
            )
        except recordings.LabelsError:  # recordings' labels.csv needs a person: try again after
            unreadable += 1
            continue
        except ValueError:  # silent, too short or long, an odd rate: the same next time
            seen[cid] = ""
            refused += 1
        except (RuntimeError, OSError):  # unreadable, maybe still being written: next time
            unreadable += 1
            continue
        else:
            seen[cid] = r["file"]
            added += 1
        f = _index(p)  # after each one: interrupted, at most the take being added comes in twice
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".part")
        tmp.write_text(json.dumps(seen, indent=1))
        tmp.replace(f)
    return added, unreadable, refused
