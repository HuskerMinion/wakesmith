"""Device clips: the clips your devices saved (false wakes and real calls), labeled into the next round.

The loop that makes a wake word good in a real home: devices save a clip whenever the model fires (or
nearly fires); those clips are the best negatives there are, and the real calls among them the best
positives. This module brings them in and turns them into training data:

1. import_folder: copy clips from a folder (any layout; a puller's <device>/.../clip.wav works) into
   <project>/devices/inbox/<device>/. A sidecar <clip>.json with "at" (ISO time), "model" and "score"
   is read when present (its "device" names the device, else the first folder below the one imported),
   else the file time is used. Copies, because device clip folders are often
   pruned after a while.
2. transcribe: Whisper on every clip not transcribed yet (Hugging Face
   transformers on the GPU, or faster-whisper when installed, the engine Home Assistant uses).
3. suggest: "wake" when the transcript says the phrase, "not_wake" when it has none of the phrase's
   words, "unclear" otherwise. You confirm or change labels (label(), the Device clips page).
4. apply: labeled clips go to <project>/devices/pos and devices/neg with labels.csv, and the newest
   ones (on or after `holdout_from`, a time) into HELDOUT.txt, so the test is on days the model never
   trained on (a random split flatters it: clips of one day share voices and noise). real.py reads both
   folders as extra sources ("dpos", "dneg").
"""

from __future__ import annotations

import csv
import fcntl
import functools
import json
import re
import shutil
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import soundfile as sf

from . import takes
from .project import Project
from .real import SR, copy_as_16k_mono, read_16k_mono

LABELS = ("wake", "not_wake", "skip")
MIN_TRIM_S = 0.3  # the shortest part of a clip a trim may keep


def _inbox(p: Project) -> Path:
    return p.path("devices", "inbox")


_mine = threading.local()  # per thread: the projects whose lock it holds
_fallbacks: dict[str, threading.Lock] = {}
_fallbacks_lock = threading.Lock()


def _fallback(key: str) -> threading.Lock:
    with _fallbacks_lock:
        return _fallbacks.setdefault(key, threading.Lock())


@contextmanager
def index_lock(p: Project):
    """One change to a project's device clip index (clips.json, deleted.json, takes.json, devices/pos
    and neg) at a time, across processes: a sync, the page and the command line each read the index,
    change it and write it back.
    Nested use in one thread takes the lock once."""
    key = str(p.dir.resolve())
    held = getattr(_mine, "keys", set())
    if key in held:
        yield
        return
    with open(p.dir / ".devices.lock", "a") as fd:  # in the project folder: no folder made for it
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            fallback = None
        except OSError:  # a network share without flock: this process's threads, at least
            fallback = _fallback(key)
            fallback.acquire()
        _mine.keys = held | {key}
        try:
            yield
        finally:
            _mine.keys = held
            if fallback is None:
                fcntl.flock(fd, fcntl.LOCK_UN)
            else:
                fallback.release()


def _serial(fn):
    @functools.wraps(fn)
    def run(p: Project, *a, **kw):
        with index_lock(p):
            return fn(p, *a, **kw)

    return run


def _index_path(p: Project) -> Path:
    return p.path("devices", "clips.json")


def load(p: Project) -> list[dict]:
    f = _index_path(p)
    return json.loads(f.read_text()) if f.exists() else []


def _save(p: Project, clips: list[dict]) -> None:
    f = _index_path(p)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".part")
    tmp.write_text(json.dumps(clips, indent=1))
    tmp.replace(f)


def _deleted_path(p: Project) -> Path:
    return p.path("devices", "deleted.json")


def deleted(p: Project) -> set[str]:
    """Ids of clips a person deleted: an import never brings them in again."""
    f = _deleted_path(p)
    try:
        return set(json.loads(f.read_text())) if f.exists() else set()
    except (json.JSONDecodeError, OSError, TypeError):
        return set()


@_serial
def delete(p: Project, ids: list[str]) -> int:
    """Delete clips: the copy here and the index entry go, and the id is remembered so the next import
    or sync skips the clip (the source folder is only read, so it is still there). It also leaves
    devices/pos and devices/neg, so training never meets a clip that is gone. Returns how many."""
    want = set(ids)
    clips = load(p)
    gone = [c for c in clips if c["id"] in want]
    for c in gone:
        file(p, c["id"]).unlink(missing_ok=True)
    f = _deleted_path(p)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".part")
    tmp.write_text(json.dumps(sorted(deleted(p) | {c["id"] for c in gone})))
    tmp.replace(f)
    _save(p, [c for c in clips if c["id"] not in want])
    for c in gone:
        _drop_used(p, c["id"])
    return len(gone)


def _inside(cid: str) -> bool:
    """A clip id that names a file under a folder: no absolute path, no "..", no empty part ("." as
    the folder, from an older import, is the inbox itself and safe)."""
    parts = cid.split("/")
    return len(parts) == 2 and parts[0] not in ("", "..") and parts[1] not in ("", ".", "..")


def _drop_used(p: Project, cid: str) -> None:
    """Take a deleted clip out of devices/pos and devices/neg (its labels.csv row, HELDOUT.txt line and
    link or trimmed copy), and nothing else: the other labels stay as they were last used. The lists
    are replaced whole (a reader never sees half of one) before the file goes."""
    if not _inside(cid):
        return
    for sub in ("pos", "neg"):
        d = p.path("devices", sub)
        lab = d / "labels.csv"
        if lab.exists():
            with open(lab, newline="", encoding="utf-8") as fd:
                rows = list(csv.reader(fd))
            keep = [r for r in rows if not r or r[0] != cid]
            if len(keep) != len(rows):
                tmp = lab.with_suffix(".part")
                with open(tmp, "w", newline="", encoding="utf-8") as fd:
                    csv.writer(fd).writerows(keep)
                tmp.replace(lab)
        held = d / "HELDOUT.txt"
        if held.exists():
            lines = held.read_text(encoding="utf-8").splitlines(keepends=True)
            keep_l = [x for x in lines if x.strip() != cid]
            if len(keep_l) != len(lines):
                tmp = held.with_suffix(".part")
                tmp.write_text("".join(keep_l), encoding="utf-8")
                tmp.replace(held)
        f = d / cid
        if f.is_symlink() or f.exists():
            f.unlink()


@_serial
def set_trim(p: Project, cid: str, start: float | None, end: float | None = None) -> dict | None:
    """Keep only start..end (seconds) of a clip when it goes into training or the test (apply()); the
    clip's own file stays whole. start None clears the trim. Returns the clip's index entry."""
    clips = load(p)
    c = next((c for c in clips if c["id"] == cid), None)
    if c is None:
        raise KeyError(cid)
    if start is None:
        c.pop("trim", None)
    else:
        dur = sf.info(str(file(p, cid))).duration
        s, e = round(float(start), 3), round(float(dur if end is None else end), 3)
        if not (0 <= s < e <= dur + 0.001):
            raise ValueError(f"trim must be within the clip (0-{dur:.2f} s), start before end")
        if e - s < MIN_TRIM_S:
            raise ValueError(f"a trim keeps at least {MIN_TRIM_S} s")
        c["trim"] = [s, min(e, round(dur, 3))]
    _save(p, clips)
    return c


def _write_trimmed(src: Path, dst: Path, trim: list[float]) -> None:
    a = read_16k_mono(src)
    s, e = round(trim[0] * SR), round(trim[1] * SR)
    sf.write(str(dst), a[s:e], SR, subtype="PCM_16", format="WAV")


def _meta(wav: Path) -> dict:
    side = wav.with_suffix(".json")
    if side.exists():
        try:
            meta = json.loads(side.read_text())
            return meta if isinstance(meta, dict) else {}
        except (json.JSONDecodeError, OSError):
            pass
    return {}


@_serial
def import_folder(
    p: Project,
    folder: str | Path,
    model_prefix: str = "",
    since: str = "",
    limit: int = 0,
) -> dict:
    """Copy new clips in; returns {"added", "known", "skipped", "unreadable", "left", "ids", "takes"}.
    Takes recorded on a device (kind "take") go into the word's recordings instead (takes.py), whatever
    their time; "takes" counts them.

    With `model_prefix`, only clips whose sidecar "model" starts with it (the word's model on the
    devices: a puller folder holds every word's fires; slot numbers move, the model name doesn't);
    clips without a sidecar are skipped then. With `since` (ISO time), only clips from then on. With
    `limit`, at most that many new clips, oldest first ("left" counts the rest for the next run).
    A WAV that can't be read or converted is left out and counted in "unreadable"; it is tried
    again next time (it may still have been being written)."""
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise ValueError(f"not a folder: {root}")
    clips = load(p)
    known = {c["id"] for c in clips} | deleted(p)
    took = takes.done(p)
    new, skipped, new_takes = [], 0, []
    for wav in sorted(root.rglob("*.wav")):
        rel = wav.relative_to(root)
        meta = _meta(wav)
        if model_prefix and not takes.word(meta).startswith(model_prefix):
            skipped += 1
            continue
        device = re.sub(
            r"[^A-Za-z0-9_.-]",
            "_",
            str(meta.get("device") or (rel.parts[0] if len(rel.parts) > 1 else root.name)),
        )
        # never the inbox itself, the folder above it, or an absolute id (importing "." or "/"
        # gives an empty name); ".x" stays as 0.4.11 had it
        old_id = f"{device}/{wav.name}"  # the id an older wakesmith gave it
        if device in ("", ".", ".."):
            device = "_" * max(1, len(device))
        cid = f"{device}/{wav.name}"
        if takes.is_take(meta, rel):  # recorded on the device as the phrase: into recordings
            if cid not in took and old_id not in took:
                new_takes.append((str(meta.get("at") or ""), cid, device, wav, meta))
            continue
        if cid in known or old_id in known:  # in already, perhaps under its old id
            continue
        at = meta.get("at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(wav.stat().st_mtime))
        if since and str(at) < since:
            skipped += 1
            continue
        known.add(cid)
        new.append((str(at), cid, device, wav, meta))
    new.sort(key=lambda x: x[0])
    added, unreadable, left = [], 0, 0
    for k, (at, cid, device, wav, meta) in enumerate(new):
        if limit and len(added) >= limit:  # the limit counts clips that came in, not broken ones
            left = len(new) - k
            break
        dst = _inbox(p) / device / wav.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            copy_as_16k_mono(wav, dst)  # a 44.1 kHz or stereo recorder's clips come in converted
        except (RuntimeError, ValueError, OSError):  # a broken or odd WAV: the rest still come in
            dst.unlink(missing_ok=True)  # e.g. half copied when the disk filled
            unreadable += 1
            continue
        added.append(cid)
        clips.append(
            {
                "id": cid,
                "device": device,
                "at": at,
                "model": meta.get("model", ""),
                "score": meta.get("score"),
                "text": None,
                "suggestion": None,
                "label": None,
            }
        )
    clips.sort(key=lambda c: c["at"])
    _save(p, clips)
    new_takes.sort(key=lambda x: (x[0], x[1]))
    took_n, bad_takes, refused = takes.add(
        p, [(cid, dev, wav, meta) for _, cid, dev, wav, meta in new_takes]
    )
    return {
        "takes": took_n,
        "takes_refused": refused,
        "added": len(added),
        "known": len(clips) - len(added),
        "skipped": skipped,
        "unreadable": unreadable + bad_takes,
        "left": left,
        "ids": added,
    }


def file(p: Project, cid: str) -> Path:
    device, name = cid.split("/", 1)
    f = (_inbox(p) / device / name).resolve()
    if _inbox(p).resolve() not in f.parents:
        raise ValueError("bad clip id")
    return f


# -- transcripts and suggestions ------------------------------------------------------------------
_WORD = re.compile(r"[a-zäöüß']+")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def suggest(phrase: str | list[str], text: str) -> str:
    """ "wake" if the transcript says the phrase (its words in order; any of its forms, see
    Project.heard_forms), "not_wake" if it has none of their words, else "unclear" (e.g. one "hey" of
    "hey jarvis")."""
    forms = [_words(f) for f in ([phrase] if isinstance(phrase, str) else phrase)]
    got = _words(text)
    for want in forms:
        if want and any(got[i : i + len(want)] == want for i in range(len(got) - len(want) + 1)):
            return "wake"
    return "unclear" if {w for want in forms for w in want} & set(got) else "not_wake"


class _Whisper:
    """Whisper turbo through faster-whisper (CPU/CUDA, Home Assistant's engine) or transformers (any torch
    device, e.g. an Intel GPU)."""

    def __init__(self, device: str, model: str = "turbo", language: str = "en"):
        self.device, self.language = device, language
        if device.split(":")[0] in ("cpu", "cuda"):
            try:
                from faster_whisper import WhisperModel

                kind = "cuda" if device.startswith("cuda") else "cpu"
                self.fw = WhisperModel(
                    model, device=kind, compute_type="int8" if kind == "cpu" else "float16"
                )
                return
            except ImportError:
                pass
        self.fw = None
        try:
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
        except ImportError:
            raise RuntimeError(
                "transcribing needs Whisper, which comes with wakesmith: reinstall it, "
                "or install faster-whisper"
            ) from None
        name = "openai/whisper-large-v3-turbo"
        dtype = torch.float32 if device == "cpu" else torch.float16
        self.torch, self.dtype = torch, dtype
        self.proc = WhisperProcessor.from_pretrained(name)
        self.model = (
            WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=dtype)
            .to(device)
            .eval()
        )

    def __call__(self, paths: list[Path]) -> list[str]:
        import soundfile as sf

        if self.fw is not None:
            out = []
            for f in paths:
                segs, _ = self.fw.transcribe(
                    str(f), language=self.language, beam_size=5, condition_on_previous_text=False
                )
                out.append(" ".join(s.text.strip() for s in segs))
            return out
        audio = [sf.read(str(f), dtype="float32")[0] for f in paths]
        feats = self.proc(audio, sampling_rate=16000, return_tensors="pt").input_features
        with self.torch.inference_mode():
            ids = self.model.generate(
                feats.to(self.device, self.dtype),
                language=self.language,
                task="transcribe",
                num_beams=5,
                do_sample=False,
            )
        return [t.strip() for t in self.proc.batch_decode(ids, skip_special_tokens=True)]


def transcribe(
    p: Project, device: str = "cpu", progress=None, batch: int = 8, ids=None, whisper=None
) -> dict:
    """Whisper on the clips not transcribed yet (only those in `ids`, when given)."""
    clips = load(p)
    todo = [c for c in clips if c["text"] is None and (ids is None or c["id"] in ids)]
    if not todo:
        return {"transcribed": 0}
    w = whisper or _Whisper(device, language=p.language)
    for i in range(0, len(todo), batch):
        group = todo[i : i + batch]
        heard = {
            c["id"]: text
            for c, text in zip(group, w([file(p, c["id"]) for c in group]), strict=True)
        }
        _merge_text(p, heard)  # progress survives a stop
        if progress:
            progress(min(i + batch, len(todo)), len(todo))
    return {"transcribed": len(todo)}


@_serial
def _merge_text(p: Project, heard: dict[str, str]) -> None:
    """Write transcripts into the index as it is now: labels set (or clips deleted) while Whisper ran
    stay as they are."""
    clips = load(p)
    for c in clips:
        if c["id"] in heard:
            c["text"] = heard[c["id"]]
            c["suggestion"] = suggest(p.heard_forms, c["text"])
    _save(p, clips)


# -- labels and the round ------------------------------------------------------------------------
@_serial
def label(p: Project, labels: dict[str, str | None]) -> int:
    """Set labels by clip id ("wake", "not_wake", "skip", or None to clear)."""
    clips = load(p)
    n = 0
    for c in clips:
        if c["id"] in labels:
            v = labels[c["id"]]
            if v is not None and v not in LABELS:
                raise ValueError(f"label must be one of {LABELS}")
            c["label"] = v
            n += 1
    _save(p, clips)
    return n


@_serial
def accept_suggestions(p: Project, kinds: tuple[str, ...] = ("not_wake",)) -> int:
    """Take the clear suggestions as labels where none is set yet. By default only "not_wake" (Whisper
    heard none of the phrase's words): a clip where it heard the phrase may still be the TV saying it,
    so that one waits for a person unless asked for."""
    clips = load(p)
    n = 0
    for c in clips:
        if c["label"] is None and c.get("suggestion") in kinds:
            c["label"] = c["suggestion"]
            n += 1
    _save(p, clips)
    return n


def newer_than(clips: list[dict], t: float) -> int:
    """Clips the devices saved after time t (seconds since the epoch), left-out ones not counted: what a
    new version made now would add over this one."""
    n = 0
    for c in clips:
        try:
            at = datetime.fromisoformat(c["at"].replace("Z", "+00:00")).timestamp()
        except (KeyError, ValueError, AttributeError):
            continue
        n += at > t and c.get("label") != "skip"
    return n


def to_check(clips: list[dict]) -> int:
    """Clips only a person can label: unlabeled and not a clear "not the phrase" (those are labeled
    by themselves on sync): unclear, not transcribed yet, or where Whisper heard the phrase."""
    return sum(c["label"] is None and c.get("suggestion") != "not_wake" for c in clips)


def summary(p: Project, holdout_from: str | None = None) -> dict:
    clips = load(p)
    out = {
        "total": len(clips),
        "transcribed": sum(c["text"] is not None for c in clips),
        "labels": {k: sum(c["label"] == k for c in clips) for k in LABELS},
        "unlabelled": sum(c["label"] is None for c in clips),
        "unclear": sum(c["label"] is None and c.get("suggestion") == "unclear" for c in clips),
        "to_check": to_check(clips),
        "first": clips[0]["at"] if clips else None,
        "last": clips[-1]["at"] if clips else None,
    }
    inbox = _inbox(p)
    gone = {  # labeled, but the file was removed: Use the labeled clips leaves them out
        c["id"]
        for c in clips
        if c["label"] in ("wake", "not_wake") and not (inbox / c["id"]).is_file()
    }
    if gone:
        out["missing"] = len(gone)
        out["missing_by"] = {
            k: sum(c["label"] == k and c["id"] in gone for c in clips) for k in ("wake", "not_wake")
        }
    if holdout_from:
        out["heldout"] = {
            k: sum(
                c["label"] == k and c["at"] >= holdout_from and c["id"] not in gone for c in clips
            )
            for k in ("wake", "not_wake")
        }
    return out


@_serial
def apply(p: Project, holdout_from: str) -> dict:
    """Write devices/pos and devices/neg (labels.csv, HELDOUT.txt) from the labeled clips. Clips at or
    after `holdout_from` (an ISO time, e.g. "2026-10-23T18:00") are held out."""
    clips = load(p)
    stats = {}
    for sub, want in (("pos", "wake"), ("neg", "not_wake")):
        d = p.path("devices", sub)
        shutil.rmtree(d, ignore_errors=True)
        chosen = [c for c in clips if c["label"] == want]
        if not chosen:
            continue
        d.mkdir(parents=True)
        rows, held, missing = [], [], 0
        for c in chosen:
            rel = c["id"]
            if not _inside(rel):  # an id an older wakesmith made from an empty device name
                missing += 1
                continue
            src = file(p, c["id"])
            if not src.is_file():  # gone from the inbox (removed by hand): left out, not a crash
                missing += 1
                continue
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            if c.get("trim"):  # a trimmed copy; the clip's own file stays whole
                _write_trimmed(src, d / rel, c["trim"])
            else:
                (d / rel).symlink_to(src)
            rows.append(c)
            if c["at"] >= holdout_from:
                held.append(rel)
        with open(d / "labels.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "device", "at", "model", "score", "said"])
            for c in rows:
                w.writerow([c["id"], c["device"], c["at"], c["model"], c["score"], c["text"] or ""])
        (d / "HELDOUT.txt").write_text(
            f"# device clips at or after {holdout_from} (rounds.apply)\n"
            + "".join(f"{h}\n" for h in held),
            encoding="utf-8",
        )
        stats[sub] = {"train": len(rows) - len(held), "heldout": len(held)}
        if missing:
            stats[sub]["missing"] = missing
    meta = p.path("devices", "round.json")
    meta.write_text(
        json.dumps({"holdout_from": holdout_from, "applied": time.time(), "stats": stats})
    )
    return stats
