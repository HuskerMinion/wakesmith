"""A wakesmith project: one wake word, one folder.

<project>/project.yaml holds every setting; everything else in the folder is generated:
  clips/<split>/clip_NNNNNN.wav   synthetic speech (positive_train, positive_test, negative_train,
                                  negative_test)
  real/{train,test}, real_neg/{train,test}   prepared real recordings
  review/                          review-page list and verdicts
  mww/, oww/                       features, training runs, trained models
  export/                          what you copy to your devices
  state.json, logs/                stage status and logs
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from . import assets


def projects_home() -> Path:
    return Path(os.environ.get("WAKESMITH_PROJECTS", assets.home() / "projects"))


@dataclass
class Samples:
    positive: int = 25000  # synthetic clips of the wake phrase (training)
    negative: int = 25000  # synthetic sound-alike clips (training)
    positive_test: int = 2500
    negative_test: int = 2500


@dataclass
class Real:
    positives: str = ""  # folder of wake-phrase recordings (optional labels.csv, HELDOUT.txt)
    negatives: str = ""  # folder of not-the-phrase recordings, ideally real false triggers
    copies: int = 10  # copies of each real positive in training (each augmented differently)
    negative_copies: int = 5
    # microWakeWord: 0 mixes real positives into the synthetic set; > 0 trains them as their own
    # set sampled at this weight (synthetic positives: 2.0), so a few dozen takes aren't drowned.
    sampling_weight: float = 0.0
    ignore_tail_s: float = 0.3  # ignore a device chime at the end of positive clips


@dataclass
class CommonVoice:
    enabled: bool = False  # real volunteer voices from Mozilla Common Voice (commonvoice.py)
    dataset: str = ""  # Common Voice 17.0 English folder; empty: <data>/common_voice_17
    train: bool = False  # also train on the non-test speakers' takes (test only otherwise)
    test_share: float = 0.4  # share of speakers kept aside for the "Common Voice speakers" test
    max_train: int = 1000  # cap on training takes, so real voices don't drown the synthetic set


@dataclass
class MicroWakeWord:
    enabled: bool = True
    steps: int = 20000
    # a second phase at a fifth of the learning rate (e.g. 30k + 10k steps)
    finetune_steps: int = 0
    negative_class_weight: float = 20.0
    # sampling weight of the synthetic sound-alikes (synthetic positives: 2.0); raise it when
    # near-copies of the phrase ("I needed help" for "I need help") still fire
    adversarial_weight: float = 3.0
    cutoff: float = 0.9  # probability cutoff in the exported manifest (pick from test results)
    # Feature frontend of the devices the model is for, used by `wakesmith test` and review:
    # "tflm" (ESPHome, microWakeWord's own runtime) or "zserge" (zserge/microwakeword, pure Go).
    frontend: str = "tflm"


@dataclass
class OpenWakeWord:
    enabled: bool = True
    steps: int = 50000
    layer_dim: int = 128
    blocks: int = 2
    # Starting weight of negatives (doubled after a phase while the false-trigger rate on the
    # validation set is above target). Tuned on real held-out clips: 3000, LiveKit's setting,
    # suppressed recall to ~14% once the features use the runtime's int16 scale; 100 -> 400 worked.
    max_negative_weight: float = 100.0
    target_fp_per_hour: float = 0.1
    threshold: float = 0.5


@dataclass
class Project:
    name: str
    phrase: str
    label: str = ""
    negatives: list[str] = field(default_factory=list)  # sound-alike phrases to reject
    positive_spellings: list[str] = field(default_factory=list)  # extra spellings of the phrase
    tts_models: list[str] = field(default_factory=lambda: ["libritts-high", "libritts-r-medium"])
    samples: Samples = field(default_factory=Samples)
    real: Real = field(default_factory=Real)
    commonvoice: CommonVoice = field(default_factory=CommonVoice)
    microwakeword: MicroWakeWord = field(default_factory=MicroWakeWord)
    openwakeword: OpenWakeWord = field(default_factory=OpenWakeWord)
    data_profile: str = "personal"
    hardware: dict[str, Any] = field(default_factory=dict)  # overrides for auto-detected settings
    dir: Path = field(default=Path(), repr=False, compare=False)

    @property
    def display(self) -> str:
        return self.label or self.phrase.title()

    def path(self, *parts: str) -> Path:
        return self.dir.joinpath(*parts)

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        data = {k: v for k, v in asdict(self).items() if k != "dir"}
        (self.dir / "project.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

    # -- stage state -----------------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        p = self.path("state.json")
        return json.loads(p.read_text()) if p.exists() else {}

    def set_state(self, **kv: Any) -> None:
        s = self.state()
        s.update(kv)
        tmp = self.path("state.json.tmp")
        tmp.write_text(json.dumps(s, indent=1))
        tmp.replace(self.path("state.json"))


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "wake_word"


def _build(cls, data: dict):
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        v = data[f.name]
        default = f.default_factory() if callable(f.default_factory) else None  # type: ignore[misc]
        kwargs[f.name] = _build(type(default), v) if is_dataclass(default) else v
    return cls(**kwargs)


def load(where: str | Path) -> Project:
    """A project by folder, project.yaml path, or name under projects_home()."""
    p = Path(where).expanduser()
    if p.name == "project.yaml":
        p = p.parent
    if not (p / "project.yaml").exists() and not p.is_absolute():
        p = projects_home() / str(where)
    data = yaml.safe_load((p / "project.yaml").read_text())
    proj = _build(Project, data)
    proj.dir = p.resolve()
    return proj


def create(phrase: str, name: str | None = None, where: Path | None = None, **kw: Any) -> Project:
    name = name or slug(phrase)
    proj = Project(name=name, phrase=phrase.strip(), **kw)
    proj.dir = (where or projects_home() / name).resolve()
    if (proj.dir / "project.yaml").exists():
        raise FileExistsError(f"{proj.dir} already has a project")
    proj.save()
    return proj


def size_bytes(p: Project) -> int:
    return sum(f.stat().st_size for f in p.dir.rglob("*") if f.is_file() and not f.is_symlink())


def delete(p: Project) -> int:
    """Remove a project's folder (generated clips, features, recorded takes, models) and return the
    bytes freed. Folders of recordings the project only points to are not touched."""
    d = p.dir.resolve()
    if not (d / "project.yaml").is_file() or d == projects_home().resolve() or len(d.parts) < 4:
        raise ValueError(f"{d} doesn't look like a wakesmith project folder")
    freed = size_bytes(p)
    shutil.rmtree(d)
    return freed
