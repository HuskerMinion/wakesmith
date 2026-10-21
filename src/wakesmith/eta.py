"""Time estimates for pipeline stages, from this machine's own measured rates.

Each finished stage records seconds per unit (clips for generation and features, steps for
training, held-out clips for the test) in <data home>/timings.json, as a running average.
Until a stage has run once, the defaults below (measured on 12 cores + 2 x Arc) stand in.
"""

from __future__ import annotations

import json

from . import assets
from .project import Project
from .real import RealData

DEFAULT_SECONDS_PER_UNIT = {
    "generate": 0.0072,
    "real": 0.01,
    "mww-features": 0.0009,
    "mww-train": 0.042,
    "oww-features": 0.024,
    "oww-train": 0.01,
    "test": 0.03,
    "export": 1.0,
}


def _path():
    return assets.home() / "timings.json"


def _load() -> dict:
    try:
        return json.loads(_path().read_text())
    except (OSError, ValueError):
        return {}


def units(p: Project, stage: str) -> float:
    s = p.samples
    clips = s.positive + s.negative + s.positive_test + s.negative_test
    if stage in ("generate", "mww-features", "oww-features"):
        return clips
    if stage == "mww-train":
        return p.microwakeword.steps
    if stage == "oww-train":
        return p.openwakeword.steps * 1.2  # phases 2 and 3 add a tenth each
    if stage in ("real", "test"):
        real = RealData(p)
        try:
            listed = [real.listing(src) for src in real.sources]
        except ValueError:
            return 0
        if stage == "real":
            return sum(len(files) for files, _, _ in listed)
        return sum(len(held) for _, held, _ in listed) * (
            p.microwakeword.enabled + p.openwakeword.enabled
        )
    return 1


def record(p: Project, stage: str, seconds: float, info: dict | None) -> None:
    """Fold a finished stage's rate into the machine's timings (skipped stages don't count)."""
    n = units(p, stage)
    if n <= 0 or (info and "skipped" in info):
        return
    t = _load()
    rate = seconds / n
    old = t.get(stage)
    t[stage] = rate if old is None else 0.5 * old + 0.5 * rate
    try:
        _path().write_text(json.dumps(t, indent=1))
    except OSError:
        pass


def estimates(p: Project) -> dict[str, float]:
    """Expected seconds for each stage of a full run of this project."""
    t = _load()
    return {
        stage: units(p, stage) * t.get(stage, default)
        for stage, default in DEFAULT_SECONDS_PER_UNIT.items()
    }
