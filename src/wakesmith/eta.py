"""Time estimates for pipeline stages, from this machine's own measured rates.

Each finished stage records seconds per unit (clips for generation and features, steps for
training, held-out clips for the test) in <data home>/timings.json, as a running average.
Until a stage has run once, the defaults below (measured on 12 cores + 2 x Arc) stand in.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import sys

from . import assets, hardware, setup_env
from .project import Project
from .real import RealData

DEFAULT_SECONDS_PER_UNIT = {
    "generate": 0.0072,
    "check-clips": 0.07,  # Whisper turbo, one clip per window: ~14 clips/s on an RTX 5060 Ti
    "real": 0.01,
    "mww-features": 0.0009,
    "mww-train": 0.01,  # PyTorch port: ~0.007 on a GPU, ~0.02 on 12 CPU threads
    "oww-features": 0.024,
    "oww-train": 0.01,
    "test": 0.03,
    "commonvoice": 5.0,
    "conversational": 0.1,  # a few seconds of speech per clip, plus the assistant writing it
    "export": 1.0,
}


def _path():
    return assets.home() / "timings.json"


def _load() -> dict:
    try:
        return json.loads(_path().read_text())
    except (OSError, ValueError):
        return {}


@functools.lru_cache(maxsize=1)
def _can_listen() -> bool:
    """A GPU (from PCI ids, not opening its driver) and Whisper: without them the check is skipped."""
    if hardware.forced_cpu() or importlib.util.find_spec("transformers") is None:
        return False
    return sys.platform == "darwin" or bool(setup_env.find_gpus())


def units(p: Project, stage: str) -> float:
    s = p.samples
    clips = s.positive + s.negative + s.positive_test + s.negative_test
    conv = max(0, p.conversational.clips)
    if stage == "conversational":
        return conv
    if stage == "generate":
        return clips
    if stage == "check-clips":
        # the sound-alike clips Whisper hasn't heard yet (one line per clip heard; roughly)
        if not p.samples.check_negatives or not _can_listen() or p.hardware.get("gpus") == []:
            return 0
        heard = 0
        for f in p.path("clips", "whisper").glob("negatives-*.jsonl"):
            with open(f, "rb") as fh:
                heard += sum(1 for _ in fh)
        return max(0, s.negative + s.negative_test - heard)
    if stage in ("mww-features", "oww-features"):
        return clips + conv
    if stage == "mww-train":
        return (p.microwakeword.steps + p.microwakeword.finetune_steps) * max(
            1, p.microwakeword.candidates
        )
    if stage == "oww-train":
        return p.openwakeword.steps * 1.2  # phases 2 and 3 add a tenth each
    if stage in ("real", "test"):
        real = RealData(p)
        try:
            listed = [real.listing(src) for src in real.sources]
        except (OSError, ValueError):
            return 0
        if stage == "real":
            return sum(len(files) for files, _, _ in listed)
        return sum(len(held) for _, held, _ in listed) * (
            p.microwakeword.enabled + p.openwakeword.enabled
        )
    return 1


def _key(p: Project, stage: str) -> str:
    """Speech is made by other voices in another language, at another speed: its own timing."""
    if stage in ("generate", "conversational") and p.language != "en":
        return f"{stage}:{p.language}"
    return stage


def record(p: Project, stage: str, seconds: float, info: dict | None) -> None:
    """Fold a finished stage's rate into the machine's timings (skipped stages don't count)."""
    # the listening check's units are the clips still to hear: count what this run heard instead
    n = (info or {}).get("listened", 0) if stage == "check-clips" else units(p, stage)
    if n <= 0 or (info and "skipped" in info):
        return
    t = _load()
    rate = seconds / n
    key = _key(p, stage)
    old = t.get(key)
    t[key] = rate if old is None else 0.5 * old + 0.5 * rate
    try:
        _path().write_text(json.dumps(t, indent=1))
    except OSError:
        pass


def estimates(p: Project) -> dict[str, float]:
    """Expected seconds for each stage of a full run of this project."""
    t = _load()
    return {
        stage: units(p, stage) * t.get(_key(p, stage), default)
        for stage, default in DEFAULT_SECONDS_PER_UNIT.items()
    }
