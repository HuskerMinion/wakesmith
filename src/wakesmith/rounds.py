"""Device clips: the clips your devices saved (false wakes and real calls), labeled into the next round.

The loop that makes a wake word good in a real home: devices save a clip whenever the model fires (or
nearly fires); those clips are the best negatives there are, and the real calls among them the best
positives. This module brings them in and turns them into training data:

1. import_folder: copy clips from a folder (any layout; a puller's <device>/.../clip.wav works) into
   <project>/devices/inbox/<device>/. A sidecar <clip>.json with "at" (ISO time), "model" and "score"
   is read when present (its "device" names the device, else the first folder below the one imported),
   else the file time is used. Copies, because device clip folders are often
   pruned after a while.
2. transcribe: Whisper on every clip not transcribed yet (wakesmith[whisper]: Hugging Face
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
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

from .project import Project

LABELS = ("wake", "not_wake", "skip")


def _inbox(p: Project) -> Path:
    return p.path("devices", "inbox")


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


def _meta(wav: Path) -> dict:
    side = wav.with_suffix(".json")
    if side.exists():
        try:
            meta = json.loads(side.read_text())
            return meta if isinstance(meta, dict) else {}
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def import_folder(
    p: Project,
    folder: str | Path,
    model_prefix: str = "",
    since: str = "",
    limit: int = 0,
) -> dict:
    """Copy new clips in; returns {"added", "known", "skipped", "left", "ids"}.

    With `model_prefix`, only clips whose sidecar "model" starts with it (the word's model on the
    devices: a puller folder holds every word's fires; slot numbers move, the model name doesn't);
    clips without a sidecar are skipped then. With `since` (ISO time), only clips from then on. With
    `limit`, at most that many new clips, oldest first ("left" counts the rest for the next run)."""
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise ValueError(f"not a folder: {root}")
    clips = load(p)
    known = {c["id"] for c in clips}
    new, skipped = [], 0
    for wav in sorted(root.rglob("*.wav")):
        rel = wav.relative_to(root)
        meta = _meta(wav)
        if model_prefix and not str(meta.get("model") or "").startswith(model_prefix):
            skipped += 1
            continue
        device = re.sub(
            r"[^A-Za-z0-9_.-]",
            "_",
            str(meta.get("device") or (rel.parts[0] if len(rel.parts) > 1 else root.name)),
        )
        cid = f"{device}/{wav.name}"
        if cid in known:
            continue
        at = meta.get("at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(wav.stat().st_mtime))
        if since and str(at) < since:
            skipped += 1
            continue
        known.add(cid)
        new.append((str(at), cid, device, wav, meta))
    new.sort(key=lambda x: x[0])
    left = 0
    if limit and len(new) > limit:
        new, left = new[:limit], len(new) - limit
    for at, cid, device, wav, meta in new:
        dst = _inbox(p) / device / wav.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wav, dst)
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
    return {
        "added": len(new),
        "known": len(clips) - len(new),
        "skipped": skipped,
        "left": left,
        "ids": [x[1] for x in new],
    }


def file(p: Project, cid: str) -> Path:
    device, name = cid.split("/", 1)
    f = (_inbox(p) / device / name).resolve()
    if _inbox(p).resolve() not in f.parents:
        raise ValueError("bad clip id")
    return f


# -- transcripts and suggestions ------------------------------------------------------------------
_WORD = re.compile(r"[a-z']+")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def suggest(phrase: str, text: str) -> str:
    """ "wake" if the transcript says the phrase (its words in order), "not_wake" if it has none of the
    phrase's words, else "unclear" (e.g. one "hey" of "hey jarvis")."""
    want, got = _words(phrase), _words(text)
    if want and any(got[i : i + len(want)] == want for i in range(len(got) - len(want) + 1)):
        return "wake"
    return "unclear" if set(want) & set(got) else "not_wake"


class _Whisper:
    """Whisper turbo through faster-whisper (CPU/CUDA, Home Assistant's engine) or transformers (any torch
    device, e.g. an Intel GPU)."""

    def __init__(self, device: str, model: str = "turbo"):
        self.device = device
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
                "transcribing needs Whisper: install wakesmith with the whisper extra "
                "(uv tool install --with transformers ..., or faster-whisper)"
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
                    str(f), language="en", beam_size=5, condition_on_previous_text=False
                )
                out.append(" ".join(s.text.strip() for s in segs))
            return out
        audio = [sf.read(str(f), dtype="float32")[0] for f in paths]
        feats = self.proc(audio, sampling_rate=16000, return_tensors="pt").input_features
        with self.torch.inference_mode():
            ids = self.model.generate(
                feats.to(self.device, self.dtype),
                language="en",
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
    w = whisper or _Whisper(device)
    for i in range(0, len(todo), batch):
        group = todo[i : i + batch]
        for c, text in zip(group, w([file(p, c["id"]) for c in group]), strict=True):
            c["text"] = text
            c["suggestion"] = suggest(p.phrase, text)
        _save(p, clips)  # progress survives a stop
        if progress:
            progress(min(i + batch, len(todo)), len(todo))
    return {"transcribed": len(todo)}


# -- labels and the round ------------------------------------------------------------------------
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
    if holdout_from:
        out["heldout"] = {
            k: sum(c["label"] == k and c["at"] >= holdout_from for c in clips)
            for k in ("wake", "not_wake")
        }
    return out


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
        rows, held = [], []
        for c in chosen:
            rel = c["id"]
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).symlink_to(file(p, c["id"]))
            rows.append(c)
            if c["at"] >= holdout_from:
                held.append(rel)
        with open(d / "labels.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "device", "at", "model", "score", "said"])
            for c in rows:
                w.writerow([c["id"], c["device"], c["at"], c["model"], c["score"], c["text"] or ""])
        (d / "HELDOUT.txt").write_text(
            f"# device clips at or after {holdout_from} (rounds.apply)\n"
            + "".join(f"{h}\n" for h in held)
        )
        stats[sub] = {"train": len(rows) - len(held), "heldout": len(held)}
    meta = p.path("devices", "round.json")
    meta.write_text(
        json.dumps({"holdout_from": holdout_from, "applied": time.time(), "stats": stats})
    )
    return stats
