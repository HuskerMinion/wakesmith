"""Real recordings: listing, held-out split, review corrections, and preparation for training.

Two source folders, which wakesmith only ever reads:
  positives  clips of the wake phrase          (class "wake")
  negatives  clips that are not the phrase      (class "not_wake"), ideally real false triggers
Each may hold labels.csv (column "file": paths relative to the folder; other columns are shown on
the review page) and HELDOUT.txt (relative paths, one per line, "#" comments). Held-out clips are
never trained on; they are what `wakesmith test` scores. Without labels.csv every *.wav below the
folder counts.

Review-page verdicts (<project>/review/decisions.json) correct training labels only. Held-out clips
are never moved, so scores stay comparable between models; corrections to them are only reported.
Clip ids are "pos/<file>" and "neg/<file>".
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from .project import Project

SR = 16000
NATIVE = {"wake": "pos", "not_wake": "neg"}
VERDICTS = ("wake", "not_wake", "unsure", "bad_audio")


class RealData:
    def __init__(self, project: Project):
        self.p = project
        self.sources = {
            k: Path(v).expanduser()
            for k, v in (("pos", project.real.positives), ("neg", project.real.negatives))
            if v
        }

    def __bool__(self) -> bool:
        return bool(self.sources)

    @property
    def decisions_path(self) -> Path:
        return self.p.path("review", "decisions.json")

    def listing(self, src: str) -> tuple[list[str], set[str], dict[str, dict]]:
        """(all files, held-out files, labels.csv row per file) for source "pos" or "neg"."""
        d = self.sources.get(src)
        if d is None:
            return [], set(), {}
        if (d / "labels.csv").exists():
            with open(d / "labels.csv", newline="") as f:
                rows = {r["file"]: r for r in csv.DictReader(f)}
        else:
            rows = {str(p.relative_to(d)): {} for p in sorted(d.rglob("*.wav"))}
        held: set[str] = set()
        if (d / "HELDOUT.txt").exists():
            held = {
                line.strip()
                for line in (d / "HELDOUT.txt").read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            }
            missing = held - rows.keys()
            if missing:
                raise ValueError(
                    f"{d}/HELDOUT.txt lists {len(missing)} files that aren't in the folder "
                    f"(or labels.csv), e.g. {next(iter(missing))}"
                )
        return list(rows), held, rows

    def file(self, src: str, f: str) -> Path:
        return self.sources[src] / f

    def verdicts(self) -> dict[str, str]:
        if not self.decisions_path.exists():
            return {}
        return {k: v["verdict"] for k, v in json.loads(self.decisions_path.read_text()).items()}

    def training_items(self, want: str) -> tuple[list[tuple[str, str]], list[str]]:
        """Training clips of class `want` as (src, file), corrections applied, held-out excluded."""
        v_by_id = self.verdicts()
        native = NATIVE[want]
        items, notes = [], []
        for src in self.sources:
            files, held, _ = self.listing(src)
            for f in files:
                v = v_by_id.get(f"{src}/{f}")
                if f in held:
                    if src == native and v and v != want:
                        notes.append(f"held-out {src}/{f} reviewed as {v}: kept in the test set")
                    continue
                if src == native and v in (None, want):
                    items.append((src, f))
                elif src != native and v == want:
                    items.append((src, f))
                    notes.append(f"moved in: {src}/{f}")
                elif src == native:
                    notes.append(f"dropped ({v}): {src}/{f}")
        return items, notes

    def heldout(self, src: str) -> list[str]:
        return sorted(self.listing(src)[1])

    # -- preparation -----------------------------------------------------------------------
    def read(self, src: str, f: str) -> np.ndarray:
        a, sr = sf.read(self.file(src, f), dtype="float32", always_2d=True)
        if sr != SR:
            raise ValueError(
                f"{src}/{f}: {sr} Hz. Convert recordings to 16 kHz mono first, e.g. "
                "ffmpeg -i in.wav -ar 16000 -ac 1 out.wav"
            )
        return a[:, 0]

    def prepare(self) -> dict[str, int]:
        """Write real/{train,test} (trimmed positives) and real_neg/{train,test}."""
        stats: dict[str, int] = {}
        if "pos" in self.sources:
            stats.update(self._prep_positives())
        if "neg" in self.sources:
            stats.update(self._prep_negatives())
        return stats

    def _prep_positives(self) -> dict[str, int]:
        out = self.p.path("real")
        shutil.rmtree(out, ignore_errors=True)
        train, _ = self.training_items("wake")
        held = [("pos", f) for f in self.heldout("pos")]
        n = skipped = 0
        for split, items in (("test", held), ("train", train)):
            d = out / split
            d.mkdir(parents=True)
            for src, f in items:
                t = trim_to_phrase(self.read(src, f), self.p.real.ignore_tail_s)
                if t is None:
                    skipped += 1
                    continue
                sf.write(d / flat_name(src, f), t, SR)
                n += 1
        return {
            "real_positive_train": len(train),
            "real_positive_test": len(held),
            "real_positive_skipped": skipped,
        }

    def _prep_negatives(self) -> dict[str, int]:
        out = self.p.path("real_neg")
        shutil.rmtree(out, ignore_errors=True)
        for d in ("train", "test"):
            (out / d).mkdir(parents=True)
        held = self.heldout("neg")
        for f in held:
            shutil.copy(self.file("neg", f), out / "test" / flat_name("neg", f))
        train, _ = self.training_items("not_wake")
        n_win = 0
        for src, f in train:
            for i, w in enumerate(windows(self.read(src, f))):
                sf.write(out / "train" / flat_name(src, f).replace(".wav", f"_w{i}.wav"), w, SR)
                n_win += 1
        return {
            "real_negative_train": len(train),
            "real_negative_windows": n_win,
            "real_negative_test": len(held),
        }


def flat_name(src: str, f: str) -> str:
    """File name for a prepared copy of a source clip: unique, no subfolders."""
    return f"{src}__{f.replace('/', '__')}"


def trim_to_phrase(
    a: np.ndarray, ignore_tail_s: float = 0.3, keep_s: float = 1.9, tail_pad_s: float = 0.08
) -> np.ndarray | None:
    """Cut a recording so the phrase (its last speech burst) ends at the clip end.

    Speech = 10 ms frames above an energy threshold; the burst may contain gaps < 150 ms ("hey" ..
    "jarvis") and must last >= 150 ms. The final `ignore_tail_s` (a device chime) is not searched.
    """
    hop = SR // 100
    e = np.sqrt(np.convolve(a**2, np.ones(hop * 3) / (hop * 3), "same"))[::hop]
    tail = int(ignore_tail_s * 100)
    search = e[: len(e) - tail] if tail and len(a) > 2.5 * SR else e  # short clips: no chime
    if not len(search):
        return None
    end = _last_burst(search > max(search.max() * 0.12, np.median(search) * 3))
    if (
        end is None
    ):  # a clip that's nearly all speech: its median is speech, so floor on the quietest 10%
        end = _last_burst(search > max(search.max() * 0.12, np.percentile(search, 10) * 3))
    if end is None:
        return None
    stop = min(len(a), int((end / 100 + tail_pad_s) * SR))
    return a[max(0, stop - int(keep_s * SR)) : stop]


def _last_burst(on: np.ndarray) -> int | None:
    """End frame of the last run of `on` frames lasting >= 150 ms, allowing gaps < 150 ms."""
    i = len(on) - 1
    while i >= 0:
        if on[i]:
            j, gap = i, 0
            while j > 0 and gap < 15:
                j -= 1
                gap = 0 if on[j] else gap + 1
            if i - j >= 15:
                return i
            i = j
        else:
            i -= 1
    return None


def windows(a: np.ndarray, win_s: float = 2.0, hop_s: float = 0.5) -> list[np.ndarray]:
    """2 s windows ending every 0.5 s, so a false trigger is covered at every window position."""
    win = int(win_s * SR)
    if len(a) < win:
        a = np.concatenate([np.zeros(win - len(a), np.float32), a])
    return [a[e - win : e] for e in range(win, len(a) + 1, int(hop_s * SR))]
