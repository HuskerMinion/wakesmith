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
FIELDS = ["file", "condition", "recorded"]
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
    """Train on the recordings made here (replaces a folder chosen before)."""
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
    with open(d / "labels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    names = {r["file"] for r in rows}
    (d / "HELDOUT.txt").write_text(
        "# kept aside for testing, never trained on (wakesmith recorder)\n"
        + "".join(f"{n}\n" for n in sorted(held & names))
    )


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


def add(p: Project, kind: str, wav: bytes, condition: str = "", heldout: bool = False) -> dict:
    """Store one take (16 kHz mono WAV bytes) and switch the project to this folder if it has none."""
    a, sr = sf.read(io.BytesIO(wav), dtype="int16", always_2d=True)
    if sr != SR:
        raise ValueError(f"takes must be {SR} Hz, got {sr}")
    a = a[:, 0]
    if not MIN_S <= len(a) / SR <= MAX_S:
        raise ValueError(f"a take must be {MIN_S}-{MAX_S:.0f} s long, got {len(a) / SR:.1f} s")
    if not np.abs(a).max():
        raise ValueError("the take is silent: check the microphone")
    if _setting(p, kind) and not active(p, kind):
        raise PermissionError(
            f"this project trains on {_setting(p, kind)}; switch it to the recordings made here first"
        )
    d = folder(p, kind)
    d.mkdir(parents=True, exist_ok=True)
    rows, held = _rows(d), _held(d)
    n = max([int(r["file"][5:-4]) for r in rows if _NAME.match(r["file"])] + [0]) + 1
    name = f"take_{n:03d}.wav"
    sf.write(d / name, a, SR, subtype="PCM_16")
    rows.append(
        {
            "file": name,
            "condition": condition.strip()[:80],
            "recorded": time.strftime("%Y-%m-%d %H:%M"),
        }
    )
    if heldout:
        held.add(name)
    _write(d, rows, held)
    if not _setting(p, kind):
        use(p, kind)
    return {"file": name, "held": heldout, "seconds": round(len(a) / SR, 2)}


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


def set_heldout(p: Project, kind: str, name: str, held: bool) -> None:
    d = folder(p, kind)
    path(p, kind, name)
    h = _held(d)
    _write(d, _rows(d), (h | {name}) if held else (h - {name}))
