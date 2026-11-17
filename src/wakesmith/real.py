"""Real recordings: listing, held-out split, review corrections, and preparation for training.

Two source folders, which wakesmith only ever reads:
  positives  clips of the wake phrase          (class "wake")
  negatives  clips that are not the phrase      (class "not_wake"), ideally real false triggers
plus, once device clips have been labeled (rounds.py), the project's own <project>/devices/pos and
devices/neg ("dpos", "dneg").
Each may hold labels.csv (column "file": paths relative to the folder; other columns are shown on
the review page) and HELDOUT.txt (relative paths, one per line, "#" comments). Held-out clips are
never trained on; they are what `wakesmith test` scores. Without labels.csv every *.wav below the
folder counts.

Review-page verdicts (<project>/review/decisions.json) correct training labels only. Held-out clips
are never moved, so scores stay comparable between models; corrections to them are only reported.
Clip ids are "<source>/<file>", e.g. "pos/<file>", "dneg/<file>".
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
# each source's class: the chosen folders, labeled device clips (rounds.py), takes recorded on the page
KIND = {
    "pos": "wake",
    "neg": "not_wake",
    "dpos": "wake",
    "dneg": "not_wake",
    "rpos": "wake",
    "rneg": "not_wake",
}
VERDICTS = ("wake", "not_wake", "unsure", "bad_audio")


MIN_RATE, MAX_RATE = 4000, 768_000  # sample rates converted; others are refused
MAX_SECONDS = 7200.0  # longest clip converted (an evening of TV as not-the-phrase clips)
MAX_SAMPLES = 200_000_000  # samples read at most, all channels (800 MB as float32)


def read_16k_mono(path: str | Path) -> np.ndarray:
    """A WAV as mono float32 in [-1, 1] at 16 kHz: channels averaged, other rates (44.1 kHz phone or
    satellite recordings, say) resampled, so folders need no converting first. ValueError for a
    rate outside 4-768 kHz, or one that needs resampling and is over 2 hours or 200 million samples:
    those could need more memory than the machine has. 16 kHz mono clips read as before, any length."""
    i = sf.info(str(path))
    if i.channels > 1 and i.frames * i.channels > MAX_SAMPLES:  # averaging needs it all in memory
        raise ValueError(f"{path}: {i.channels} channels x {i.frames} samples, too long to convert")
    if i.samplerate != SR:
        if not MIN_RATE <= i.samplerate <= MAX_RATE:
            raise ValueError(f"{path}: {i.samplerate} Hz (wakesmith converts 4-768 kHz)")
        if i.frames > MAX_SECONDS * i.samplerate or i.frames * max(1, i.channels) > MAX_SAMPLES:
            raise ValueError(
                f"{path}: {i.frames / i.samplerate:.0f} s at {i.samplerate} Hz, too long to convert "
                f"(at most {MAX_SECONDS / 3600:.0f} h and {MAX_SAMPLES // 1_000_000} million "
                "samples; 16 kHz mono clips have no limit)"
            )
    a, sr = sf.read(str(path), dtype="float32", always_2d=True)
    a = a.mean(axis=1) if a.shape[1] > 1 else a[:, 0]
    if sr != SR:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(SR, sr)
        a = resample_poly(a, SR // g, sr // g).astype(np.float32)
    # resampling (or a float WAV) can overshoot full scale; int16 consumers would wrap around
    return np.clip(a, -1.0, 1.0, out=a)


def copy_as_16k_mono(src: str | Path, dst: str | Path) -> None:
    """Copy a WAV, converting it to 16 kHz mono 16-bit PCM unless it already is that."""
    i = sf.info(str(src))
    if (i.samplerate, i.channels, i.subtype) == (SR, 1, "PCM_16"):
        shutil.copy2(src, dst)
        return
    a = np.clip(read_16k_mono(src), -1.0, 1.0)
    import os
    import tempfile

    dst = Path(dst)
    # written whole under a name of its own (two imports at once don't share it), then renamed
    fd, name = tempfile.mkstemp(dir=dst.parent, prefix=f".{dst.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(name)
    try:
        sf.write(str(tmp), a, SR, subtype="PCM_16", format="WAV")
        # then the clip's own permission bits (the temp file's are owner-only; set after writing,
        # so a read-only source still converts)
        os.chmod(tmp, os.stat(src).st_mode & 0o777)
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)


def filled_row(row: dict) -> bool:
    """Not a blank row: a spreadsheet saves formatted empty rows below the data as ",,,"."""
    return any(
        any(c.strip() for c in v) if k is None else (v or "").strip()  # None: cells past the header
        for k, v in row.items()
    )


class RealData:
    def __init__(self, project: Project):
        self.p = project
        self.sources = {
            k: Path(v).expanduser()
            for k, v in (("pos", project.real.positives), ("neg", project.real.negatives))
            if v
        }
        for k, sub in (("dpos", "pos"), ("dneg", "neg")):  # labeled device clips (rounds.py)
            d = project.path("devices", sub)
            if (d / "labels.csv").exists():
                self.sources[k] = d
        # takes recorded on the page (and sent from Record a room): always used, next to a chosen folder;
        # when that folder is the recordings folder itself, it already counts as "pos" / "neg"
        for k, sub in (("rpos", "pos"), ("rneg", "neg")):
            d = project.path("recordings", sub)
            chosen = [v.resolve() for key, v in self.sources.items() if key in ("pos", "neg")]
            if (d / "labels.csv").exists() and d.resolve() not in chosen:
                self.sources[k] = d

    def of_kind(self, kind: str) -> list[str]:
        """The sources holding clips of class `kind` ("wake" or "not_wake")."""
        return [s for s in self.sources if KIND[s] == kind]

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
            # utf-8-sig: a spreadsheet's "file" header often comes with a byte-order mark
            with open(d / "labels.csv", newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                try:
                    has_file = "file" in (reader.fieldnames or [])  # reads the header row
                    rows = {r["file"]: r for r in reader if filled_row(r)} if has_file else {}
                    if any(not (f or "").strip() for f in rows):  # a short row or an empty cell
                        raise ValueError(f"{d}/labels.csv has a row without a file name")
                except (csv.Error, UnicodeDecodeError) as e:  # e.g. saved as Windows-1252
                    raise ValueError(f"{d}/labels.csv: {e}") from None
                if not has_file:
                    raise ValueError(
                        f'{d}/labels.csv needs a "file" column (paths relative to the folder)'
                    )
        else:
            rows = {str(p.relative_to(d)): {} for p in sorted(d.rglob("*.wav"))}
        held: set[str] = set()
        if (d / "HELDOUT.txt").exists():
            try:
                text = (d / "HELDOUT.txt").read_text(encoding="utf-8-sig")
            except UnicodeDecodeError as e:
                raise ValueError(f"{d}/HELDOUT.txt: {e}") from None
            held = {
                line.strip()
                for line in text.splitlines()
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
        items, notes = [], []
        for src in self.sources:
            files, held, _ = self.listing(src)
            for f in files:
                v = v_by_id.get(f"{src}/{f}")
                native = KIND[src] == want
                if f in held:
                    if native and v and v != want:
                        notes.append(f"held-out {src}/{f} reviewed as {v}: kept in the test set")
                    continue
                if native and v in (None, want):
                    items.append((src, f))
                elif not native and v == want:
                    items.append((src, f))
                    notes.append(f"moved in: {src}/{f}")
                elif native:
                    notes.append(f"dropped ({v}): {src}/{f}")
        return items, notes

    def heldout(self, src: str) -> list[str]:
        return sorted(self.listing(src)[1])

    # -- preparation -----------------------------------------------------------------------
    def read(self, src: str, f: str) -> np.ndarray:
        """Mono float32 at 16 kHz: channels are averaged and other rates (44.1 kHz phone or
        satellite recordings, say) resampled, so folders need no converting first."""
        return read_16k_mono(self.file(src, f))

    def prepare(self) -> dict[str, int]:
        """Write real/{train,test} (trimmed positives) and real_neg/{train,test}."""
        stats: dict[str, int] = {}
        if self.of_kind("wake"):
            stats.update(self._prep_positives())
        if self.of_kind("not_wake"):
            stats.update(self._prep_negatives())
        return stats

    def _prep_positives(self) -> dict[str, int]:
        out = self.p.path("real")
        shutil.rmtree(out, ignore_errors=True)
        train, _ = self.training_items("wake")
        held = [(s, f) for s in self.of_kind("wake") for f in self.heldout(s)]
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
        held = [(s, f) for s in self.of_kind("not_wake") for f in self.heldout(s)]
        for s, f in held:
            copy_as_16k_mono(self.file(s, f), out / "test" / flat_name(s, f))
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
            i = min(j, i - 1)  # a short burst at frame 0 leaves j == i
        else:
            i -= 1
    return None


def windows(a: np.ndarray, win_s: float = 2.0, hop_s: float = 0.5) -> list[np.ndarray]:
    """2 s windows ending every 0.5 s, so a false trigger is covered at every window position."""
    win = int(win_s * SR)
    if len(a) < win:
        a = np.concatenate([np.zeros(win - len(a), np.float32), a])
    return [a[e - win : e] for e in range(win, len(a) + 1, int(hop_s * SR))]
