"""A wakesmith project: one wake word, one folder.

<project>/project.yaml holds every setting; everything else in the folder is generated:
  clips/<split>/clip_NNNNNN.wav   synthetic speech (positive_train, positive_test, negative_train,
                                  negative_test)
  real/{train,test}, real_neg/{train,test}   prepared real recordings
  conversational/                  everyday sentences (sentences.txt) and their clips (experimental)
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
import time
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
    # train this many and keep the one that does best on held-out real recordings (candidates.py);
    # runs of one recipe differ a lot, so 3 is worth its extra training time
    candidates: int = 1
    # with candidates > 1, also build one model that averages them all (ensemble.py): fewer false
    # triggers at the same calls caught, N times the size; scored as "mww-combined", and exported
    # instead of the kept candidate
    combine: bool = False
    # sampling weight of cut-off copies of the phrase and the sound-alikes, as negatives (0: off). An older
    # house model whose sound-alike clips were accidentally cut short fired less on look-alikes and caught
    # more far-away calls; this makes such clips on purpose (mww.cut_short)
    partial_weight: float = 0.0
    negative_class_weight: float = 20.0
    # also use the house's own room echoes (claps from "Record a room", <data>/rooms/*/claps) for the
    # room-echo augmentation, as often as the MIT impulse responses together (sessions.house_rirs)
    house_rirs: bool = False
    # sampling weight of the synthetic sound-alikes (synthetic positives: 2.0); raise it when
    # near-copies of the phrase ("hey jarvis is" for "hey jarvis") still fire
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
class Conversational:
    # Experimental: everyday sentences that brush against the phrase ("can you help him" for "help
    # help"), written by an AI assistant on an OpenAI-compatible server, spoken by the synthetic voices
    # and trained as not-the-phrase clips (conversational.py). Number of clips; 0: off.
    clips: int = 0
    llm_url: str = ""  # empty: $WAKESMITH_LLM_URL, else http://127.0.0.1:8080
    model: str = ""  # empty: $WAKESMITH_LLM_MODEL, else whatever the server runs
    weight: float = 1.0  # microWakeWord sampling weight of these clips (synthetic positives: 2.0)


@dataclass
class Devices:
    # Device clips brought in by `wakesmith clips sync` (clipsync.py): the folder a puller fills with
    # <device>/.../<clip>.wav + <clip>.json, and the start of the sidecar "model" the clips must have
    # (the word's model on the devices, e.g. hey_jarvis; empty: every clip). Empty source: no sync.
    source: str = ""
    model_prefix: str = ""
    since: str = ""  # ISO time: only clips from then on (e.g. when the current model went live)


@dataclass
class Project:
    name: str
    phrase: str
    label: str = ""
    # Projects of one phrase are versions of one wake word (see family): v1, v2, ... in the order they
    # were made, with a short note on what that version tried. The folder name stays as it was.
    version: int = 0
    note: str = ""
    created: float = 0.0  # when the project was made (seconds since the epoch); 0: older projects
    negatives: list[str] = field(default_factory=list)  # sound-alike phrases to reject
    positive_spellings: list[str] = field(default_factory=list)  # extra spellings of the phrase
    tts_models: list[str] = field(default_factory=lambda: ["libritts-high", "libritts-r-medium"])
    samples: Samples = field(default_factory=Samples)
    real: Real = field(default_factory=Real)
    commonvoice: CommonVoice = field(default_factory=CommonVoice)
    microwakeword: MicroWakeWord = field(default_factory=MicroWakeWord)
    openwakeword: OpenWakeWord = field(default_factory=OpenWakeWord)
    conversational: Conversational = field(default_factory=Conversational)
    devices: Devices = field(default_factory=Devices)
    data_profile: str = "personal"
    hardware: dict[str, Any] = field(default_factory=dict)  # overrides for auto-detected settings
    dir: Path = field(default=Path(), repr=False, compare=False)

    @property
    def display(self) -> str:
        return self.label or self.phrase.title()

    @property
    def title(self) -> str:
        """The name the page shows: the wake word plus its version ("Hey Jarvis v3"). Exported models
        carry `display` only: a device names the phrase, not the version."""
        return f"{self.display} v{self.version}" if self.version else self.display

    @property
    def family(self) -> str:
        """The wake word this project is a version of: every project of the same phrase."""
        return slug(self.phrase)

    def path(self, *parts: str) -> Path:
        return self.dir.joinpath(*parts)

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        data = {k: v for k, v in asdict(self).items() if k != "dir"}
        tmp = self.dir / "project.yaml.tmp"  # replaced in one step: a reader never sees half a file
        tmp.write_text(yaml.safe_dump(data, sort_keys=False))
        tmp.replace(self.dir / "project.yaml")

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
    proj = Project(name=name or slug(phrase), phrase=phrase.strip(), **kw)
    if (
        where is None
    ):  # a new version of a wake word that already has some: the next number, its name
        kin = [p for p in all_projects() if p.family == proj.family]
        if kin:
            assign_versions(kin)
            proj.version = max(p.version for p in kin) + 1
            if not name and (projects_home() / proj.name).exists():
                base, k = f"{proj.family}_v{proj.version}", 1
                proj.name = base
                while (projects_home() / proj.name).exists():
                    k += 1
                    proj.name = f"{base}_{k}"
            m = re.fullmatch(r"(.*?)\s*\((.*)\)", proj.label)
            if m and not proj.note and " " in m.group(2):
                proj.note = m.group(2)
            proj.label = kin[-1].label  # the name belongs to the wake word, not to one version
    proj.created = proj.created or time.time()
    proj.dir = (where or projects_home() / proj.name).resolve()
    if (proj.dir / "project.yaml").exists():
        raise FileExistsError(f"{proj.dir} already has a project")
    proj.save()
    return proj


def project_files(home: Path | None = None) -> list[Path]:
    """Every project.yaml under the projects folder, except folders starting with "." (a version still
    being made, versions.create). Every listing of projects goes through here."""
    home = home or projects_home()
    found = sorted(home.glob("*/project.yaml")) if home.exists() else []
    return [f for f in found if not f.parent.name.startswith(".")]


def all_projects() -> list[Project]:
    out = []
    for f in project_files():
        try:
            out.append(load(f.parent))
        except (OSError, ValueError, TypeError):
            continue
    return out


def created(p: Project) -> float:
    """When the project was made: as saved at creation, else (older projects) its oldest top-level
    entry (project.yaml itself is rewritten)."""
    if p.created:
        return p.created
    times = [
        e.lstat().st_mtime for e in p.dir.iterdir() if e.name not in ("project.yaml", "state.json")
    ]
    return min(times, default=p.path("project.yaml").stat().st_mtime)


def _job_note(p: Project) -> str:
    """The note in the last run's title, "train real -> export (hh3: + soft takes)" -> "+ soft takes"."""
    try:
        title = json.loads(p.path("job.json").read_text()).get("title", "")
    except (OSError, ValueError):
        return ""
    m = re.search(r" \((.*)\)$", title)
    return re.sub(r"^[\w.-]+:\s*", "", m.group(1)) if m else ""


def assign_versions(projects: list[Project]) -> list[Project]:
    """Give every project without a version one, per wake word: v1, v2, ... in the order they were
    made, after any already numbered. Older labels like "Hey Jarvis (round 3 + everyday sentences)" become
    the word's own name plus that note; a bare code ("(c3)") gives way to the last run's description.
    Saved, so numbers never shift when a version is deleted. Returns the projects that changed."""
    changed = []
    by_word: dict[str, list[Project]] = {}
    for p in projects:
        by_word.setdefault(p.family, []).append(p)
    for kin in by_word.values():
        kin.sort(key=created)
        plain = [p.label for p in kin if p.label and "(" not in p.label and "_" not in p.label]
        name = plain[0] if plain else kin[0].phrase.title()
        n = max((p.version for p in kin), default=0)
        for p in kin:
            if p.version:
                continue
            n += 1
            m = re.fullmatch(r"(.*?)\s*\((.*)\)", p.label or "")
            note = ""
            if m and m.group(1) == name and " " in m.group(2):
                note = m.group(
                    2
                )  # "(c3 + everyday sentences)"; a bare code like "(c3)" says nothing
            p.version, p.note, p.label = n, p.note or note or _job_note(p), name
            p.save()
            changed.append(p)
    return changed


def size_bytes(p: Project) -> int:
    return sum(f.stat().st_size for f in p.dir.rglob("*") if f.is_file() and not f.is_symlink())


def dependents(p: Project) -> list[str]:
    """Other projects that use files inside p's folder: synthetic clips linked from it (versions share
    them), or a recordings folder in it chosen as their real recordings."""
    from . import archive  # every symlink a project holds (clips, features, prepared recordings)

    d = p.dir.resolve()
    inside = lambda x: x == d or d in x.parents
    linked = {n for k, names in archive.links().items() if inside(Path(k).resolve()) for n in names}
    out = []
    for q in all_projects():
        if q.dir.resolve() == d:
            continue
        links = [q.path(s) for s in ("clips", "conversational") if q.path(s).is_symlink()]
        folders = [Path(v).expanduser() for v in (q.real.positives, q.real.negatives) if v]
        if q.dir.name in linked or any(inside(x.resolve()) for x in links + folders):
            out.append(q.title)
    return out


def delete(p: Project) -> int:
    """Remove a project's folder (generated clips, features, recorded takes, models) and return the
    bytes freed. Folders of recordings the project only points to are not touched."""
    d = p.dir.resolve()
    if not (d / "project.yaml").is_file() or d == projects_home().resolve() or len(d.parts) < 4:
        raise ValueError(f"{d} doesn't look like a wakesmith project folder")
    users = dependents(p)
    if users:
        raise ValueError(
            f"{', '.join(users)} use{'s' if len(users) == 1 else ''} files in {p.title} (its synthetic voices "
            "or recordings): delete those first, or archive this one instead"
        )
    freed = size_bytes(p)
    shutil.rmtree(d)
    return freed
