"""Takes recorded in the browser, kept in the project's own folders.

<project>/recordings/pos  takes of the phrase       (the project's real.positives once used)
<project>/recordings/neg  room sound, not-the-phrase (the project's real.negatives once used)

Each folder has the labels.csv and HELDOUT.txt that real.py reads, so recorded takes go through the
same held-out split as folders you bring yourself. Files are 16 kHz mono 16-bit WAV.
"""

from __future__ import annotations

import csv
import io
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .project import Project

SR = 16000
KINDS = ("pos", "neg")
FIELDS = ["file", "speaker", "condition", "recorded"]
MIN_S, MAX_S = 0.3, 30.0
_NAME = re.compile(r"^take_\d{3,}\.wav$")


def folder(p: Project, kind: str) -> Path:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    return p.path("recordings", kind)


def _setting(p: Project, kind: str) -> str:
    return p.real.positives if kind == "pos" else p.real.negatives


def active(p: Project, kind: str) -> bool:
    """Is the project training on this recordings folder?"""
    s = _setting(p, kind)
    return bool(s) and Path(s).expanduser().resolve() == folder(p, kind).resolve()


def use(p: Project, kind: str) -> None:
    """Make the recordings folder the project's chosen folder (replacing one chosen before). The page no
    longer offers this: takes recorded here are used next to any folder anyway."""
    d = folder(p, kind)
    d.mkdir(parents=True, exist_ok=True)
    if kind == "pos":
        p.real.positives = str(d)
    else:
        p.real.negatives = str(d)
    p.save()


def _rows(d: Path) -> list[dict]:
    f = d / "labels.csv"
    if not f.exists():
        return []
    with open(f, newline="") as fh:
        return list(csv.DictReader(fh))


def _held(d: Path) -> set[str]:
    f = d / "HELDOUT.txt"
    if not f.exists():
        return set()
    return {
        ln.strip() for ln in f.read_text().splitlines() if ln.strip() and not ln.startswith("#")
    }


def _write(d: Path, rows: list[dict], held: set[str]) -> None:
    # each file is written new and moved over the old one, never edited in place: a version made by
    # versions.create may share the old file through a hard link, and must not change it
    with open(d / "labels.csv.tmp", "w", newline="") as fh:
        extra = [
            k for r in rows for k in r if k not in FIELDS
        ]  # columns someone added by hand stay
        w = csv.DictWriter(fh, fieldnames=FIELDS + list(dict.fromkeys(extra)), restval="")
        w.writeheader()
        w.writerows(rows)
    (d / "labels.csv.tmp").replace(d / "labels.csv")
    names = {r["file"] for r in rows}
    (d / "HELDOUT.txt.tmp").write_text(
        "# kept aside for testing, never trained on (wakesmith recorder)\n"
        + "".join(f"{n}\n" for n in sorted(held & names))
    )
    (d / "HELDOUT.txt.tmp").replace(d / "HELDOUT.txt")


def listing(p: Project, kind: str) -> dict:
    d = folder(p, kind)
    held = _held(d)
    takes = []
    for r in _rows(d):
        path = d / r["file"]
        if path.exists():
            takes.append(
                {**r, "held": r["file"] in held, "seconds": round(sf.info(str(path)).duration, 2)}
            )
    return {
        "folder": str(d),
        "active": active(p, kind),
        "setting": _setting(p, kind),
        "takes": takes,
    }


def add(
    p: Project,
    kind: str,
    wav: bytes,
    condition: str = "",
    heldout: bool = False,
    speaker: str = "",
    keep_unused: bool = False,
) -> dict:
    """Store one take (16 kHz mono WAV bytes). Takes recorded here are always trained on and tested
    with, next to any folder the project was given (real.py adds them as their own source), so recording
    only ever adds. A project with no folder yet gets this one as its folder. keep_unused is accepted for
    older callers and changes nothing."""
    a, sr = sf.read(io.BytesIO(wav), dtype="int16", always_2d=True)
    if sr != SR:
        raise ValueError(f"takes must be {SR} Hz, got {sr}")
    a = a[:, 0]
    if not MIN_S <= len(a) / SR <= MAX_S:
        raise ValueError(f"a take must be {MIN_S}-{MAX_S:.0f} s long, got {len(a) / SR:.1f} s")
    if not np.abs(a).max():
        raise ValueError("the take is silent: check the microphone")
    d = folder(p, kind)
    d.mkdir(parents=True, exist_ok=True)
    rows, held = _rows(d), _held(d)
    n = max([int(r["file"][5:-4]) for r in rows if _NAME.match(r["file"])] + [0]) + 1
    while (
        d / f"take_{n:03d}.wav"
    ).exists():  # a file without a row: never write into it (it may be
        n += 1  # hard-linked from another version)
    name = f"take_{n:03d}.wav"
    sf.write(d / f"{name}.tmp", a, SR, subtype="PCM_16", format="WAV")
    (d / f"{name}.tmp").replace(d / name)
    rows.append(
        {
            "file": name,
            "speaker": speaker.strip()[:40],
            "condition": condition.strip()[:80],
            "recorded": time.strftime("%Y-%m-%d %H:%M"),
        }
    )
    if heldout:
        held.add(name)
    _write(d, rows, held)
    if not _setting(p, kind):
        use(p, kind)
    return {"file": name, "held": heldout, "seconds": round(len(a) / SR, 2), "unused": False}


def _check(name: str) -> str:
    if not _NAME.match(name):
        raise ValueError("not a recorded take")
    return name


def path(p: Project, kind: str, name: str) -> Path:
    f = folder(p, kind) / _check(name)
    if not f.exists():
        raise FileNotFoundError(name)
    return f


def delete(p: Project, kind: str, name: str) -> None:
    d = folder(p, kind)
    path(p, kind, name).unlink()
    _write(d, [r for r in _rows(d) if r["file"] != name], _held(d) - {name})


def set_speaker(p: Project, kind: str, name: str, speaker: str) -> None:
    d = folder(p, kind)
    path(p, kind, name)
    rows = [{**r, "speaker": speaker.strip()[:40]} if r["file"] == name else r for r in _rows(d)]
    _write(d, rows, _held(d))


def set_heldout(p: Project, kind: str, name: str, held: bool) -> None:
    d = folder(p, kind)
    path(p, kind, name)
    h = _held(d)
    _write(d, _rows(d), (h | {name}) if held else (h - {name}))
