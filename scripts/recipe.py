"""Settings and real-recording bookkeeping shared by the recipe's Python scripts.

Settings come from recipe.env (environment variables override it) plus the livekit-wakeword YAML it
names. Real recordings live in two folders the recipe only ever reads:
  REAL_POS_DIR  clips of the wake phrase          (class "wake")
  REAL_NEG_DIR  clips that are not the wake phrase (class "not_wake"), ideally real false triggers
Each may hold labels.csv (column "file": paths relative to the folder; other columns are shown on the
review page) and HELDOUT.txt (relative paths, one per line, "#" comments): held-out clips are never
trained on and are what test_real.py scores. Without labels.csv every *.wav below the folder counts.

Label corrections from the review page (WORK/review/decisions.json) apply to training clips only.
Held-out clips are never moved, so scores stay comparable between models; corrections to held-out
clips are only reported. Clip ids are "pos/<file>" and "neg/<file>".
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _settings() -> dict[str, str]:
    s: dict[str, str] = {}
    env_file = ROOT / "recipe.env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            key, sep, val = line.partition("=")
            if sep and key.strip().isidentifier() and not line.lstrip().startswith("#"):
                s[key.strip()] = val.split("#", 1)[0].strip().strip('"')
    s.update({k: v for k, v in os.environ.items() if k in s or k.isupper()})
    return s


S = _settings()
WORK = ROOT / S.get("WORK", "work")
CONFIG = yaml.safe_load(open(ROOT / S["WAKE_CONFIG"]))
MODEL = CONFIG["model_name"]
PHRASE = CONFIG["target_phrases"][0]
LABEL = S.get("WAKE_LABEL") or PHRASE.title()  # display name, e.g. "Hey Jarvis"
DATA = ROOT / CONFIG.get("data_dir", "./data")
OUTPUT = ROOT / CONFIG.get("output_dir", "./output") / MODEL  # livekit-wakeword's model_output_dir
REAL = WORK / "real"          # prepared positives: train/ (trimmed) and test/ (trimmed held-out)
REAL_NEG = WORK / "real_neg"  # prepared negatives: train/ (2 s windows) and test/ (full held-out clips)
DECISIONS = WORK / "review/decisions.json"
SR = 16000

SOURCES = {"pos": S.get("REAL_POS_DIR", ""), "neg": S.get("REAL_NEG_DIR", "")}
SOURCES = {k: Path(v).expanduser() for k, v in SOURCES.items() if v}
NATIVE = {"wake": "pos", "not_wake": "neg"}


def num(key: str, default: float) -> float:
    return float(S.get(key) or default)


def listing(src: str) -> tuple[list[str], set[str], dict[str, dict]]:
    """(all files, held-out files, labels.csv row per file) for source "pos" or "neg"."""
    d = SOURCES.get(src)
    if d is None:
        return [], set(), {}
    if (d / "labels.csv").exists():
        rows = {r["file"]: r for r in csv.DictReader(open(d / "labels.csv"))}
    else:
        rows = {str(p.relative_to(d)): {} for p in sorted(d.rglob("*.wav"))}
    held = set()
    if (d / "HELDOUT.txt").exists():
        held = {l.strip() for l in open(d / "HELDOUT.txt") if l.strip() and not l.lstrip().startswith("#")}
        missing = held - rows.keys()
        assert not missing, f"{d}/HELDOUT.txt lists {len(missing)} files not in the folder/labels.csv, e.g. {next(iter(missing))}"
    return list(rows), held, rows


def path(src: str, f: str) -> Path:
    return SOURCES[src] / f


def verdicts() -> dict[str, str]:
    if not DECISIONS.exists():
        return {}
    return {k: v["verdict"] for k, v in json.load(open(DECISIONS)).items()}


def training_items(want: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Training clips for class `want` ("wake" or "not_wake") as (src, file) pairs, review corrections
    applied, held-out clips excluded. Also returns human-readable notes about what moved."""
    v_by_id = verdicts()
    native = NATIVE[want]
    items, notes = [], []
    for src in SOURCES:
        files, held, _ = listing(src)
        for f in files:
            v = v_by_id.get(f"{src}/{f}")
            if f in held:
                if src == native and v and v != want:
                    notes.append(f"held-out {src}/{f} reviewed as {v} - kept in the test set unchanged")
                continue
            if src == native and v in (None, want):
                items.append((src, f))
            elif src != native and v == want:
                items.append((src, f))
                notes.append(f"moved in: {src}/{f}")
            elif src == native:
                notes.append(f"dropped ({v}): {src}/{f}")
    return items, notes


def flat_name(src: str, f: str) -> str:
    """File name for a prepared copy of a source clip: unique, no subfolders."""
    return f"{src}__{f.replace('/', '__')}"
