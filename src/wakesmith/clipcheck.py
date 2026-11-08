"""Whisper listens to the synthetic sound-alike clips; those it hears as the wake phrase itself are left out
of training.

A sound-alike that differs from the phrase by a letter ("hey jarvus", "hee jarvis" for "hey jarvis") is
often spoken exactly like the phrase by the synthetic voices. Trained as "not the phrase", such a clip
teaches the model to ignore its own wake word. In a test on one wake word, Whisper heard 7% of the
sound-alike clips as exactly the phrase; three models trained without them caught 71-76% of real calls at
up to 3 false wakes, against 41-71% for three trained with them, with the same or fewer false wakes in
hours of talk.

Only a clip whose whole transcript is the phrase (any of Project.heard_forms) is left out: a sound-alike
that contains the phrase ("hey jarvis is" for "hey jarvis") is the point of having it. The clips stay where
they are (generation counts them as an unbroken run, and a later version may share the folder); what
Whisper heard is kept in clips/whisper/negatives-<k>.jsonl, one line per clip with its file time, and the
feature stages skip the clips listed there as the phrase (keep()). A clip made again later has another
file time and is listened to again.

Whisper large-v3-turbo through transformers, one clip per 30 s window (packing several clips into one
window makes Whisper hear the phrase in its neighbors too), greedy, in batches: ~14 clips/s on an RTX 5060
Ti, so ~30 minutes for 27,500 clips on one card. With several GPUs, one process per card. It doesn't run on
the CPU (hours).

usage (one card's share, as run() starts it): python -m wakesmith.clipcheck PROJECT_DIR DEVICE K N LIST.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from .project import Project, load
from .rounds import _words

MODEL = "openai/whisper-large-v3-turbo"
SPLITS = ("negative_train", "negative_test")
BATCH = 32  # ~3 GB of GPU memory with turbo in fp16


def _dir(p: Project) -> Path:
    return p.path("clips", "whisper")


def _originals(d: Path) -> list[Path]:
    return sorted(d.glob("clip_[0-9][0-9][0-9][0-9][0-9][0-9].wav"))


def _mtime(f: Path) -> int | None:
    try:
        return f.stat().st_mtime_ns
    except OSError:
        return None


def heard(p: Project) -> dict[tuple[str, str], tuple[int, str]]:
    """(split, file) -> (file time, transcript) for every clip Whisper listened to; for a clip heard more
    than once (made again, heard by another card's process) the record for the file as it is now."""
    out: dict[tuple[str, str], tuple[int, str]] = {}
    for f in sorted(_dir(p).glob("negatives-*.jsonl")):
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
                key, rec = (r["split"], r["file"]), (int(r["mtime"]), str(r["text"]))
            except (ValueError, KeyError, TypeError):
                continue  # a line cut off when a run was stopped
            if key not in out or rec[0] == _mtime(p.path("clips", key[0], key[1])):
                out[key] = rec
    return out


def is_phrase(p: Project, text: str) -> bool:
    """The whole transcript is the phrase (any way it's written), nothing more."""
    got = _words(text)
    return bool(got) and any(got == _words(f) for f in p.heard_forms)


def phrase_clips(p: Project, split: str) -> set[str]:
    """File names in clips/<split> that Whisper heard as the phrase, for the clip as it is now."""
    h = heard(p)
    d = p.path("clips", split)
    return {
        name
        for (s, name), (mt, text) in h.items()
        if s == split and is_phrase(p, text) and _mtime(d / name) == mt
    }


def keep(p: Project, split: str, files: list[str]) -> list[str]:
    """files (paths in clips/<split>) without the clips heard as the phrase; all of them when the check is
    off for the project."""
    if split not in SPLITS or not p.samples.check_negatives:
        return files
    drop = phrase_clips(p, split)
    return [f for f in files if Path(f).name not in drop] if drop else files


def todo(p: Project) -> list[tuple[str, str]]:
    """(split, file) of the sound-alike clips Whisper hasn't listened to yet (as they are now)."""
    h = heard(p)
    out = []
    for split in SPLITS:
        for f in _originals(p.path("clips", split)):
            got = h.get((split, f.name))
            if got is None or got[0] != _mtime(f):
                out.append((split, f.name))
    return out


def summary(p: Project) -> dict:
    """How many sound-alike clips were checked and left out, with the transcripts most often heard."""
    h = heard(p)
    checked = left = 0
    texts: Counter[str] = Counter()
    for split in SPLITS:
        d = p.path("clips", split)
        for f in _originals(d):
            got = h.get((split, f.name))
            if got is None or got[0] != _mtime(f):
                continue
            checked += 1
            if is_phrase(p, got[1]):
                left += 1
                texts[" ".join(_words(got[1]))] += 1
    return {
        "checked": checked,
        "heard_as_phrase": left,
        "share": round(left / checked, 4) if checked else 0.0,
        "heard": [t for t, _ in texts.most_common(3)],
    }


def available() -> str:
    """Why the check can't run here, or "" when it can."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return "Whisper isn't installed (reinstall wakesmith, or pip install transformers)"
    return ""


class CantLoad(RuntimeError):
    """Whisper couldn't be loaded (its first download, or the GPU's memory)."""


CANT_LOAD_EXIT = 75  # a card's process: Whisper couldn't be loaded


class _Listener:
    def __init__(self, device: str, language: str):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from transformers.utils import logging as tf_logging

        # its notes on the attention mask of equal-length windows and on BPE clean-up are noise in a log
        tf_logging.set_verbosity_error()
        tf_logging.disable_progress_bar()
        self.torch, self.device, self.language = torch, device, language
        self.dtype = torch.float32 if device == "cpu" else torch.float16
        self.proc = WhisperProcessor.from_pretrained(MODEL)
        self.model = (
            WhisperForConditionalGeneration.from_pretrained(MODEL, torch_dtype=self.dtype)
            .to(device)
            .eval()
        )

    def __call__(self, paths: list[Path]) -> list[str]:
        import soundfile as sf

        audio = [sf.read(str(f), dtype="float32")[0] for f in paths]
        feats = self.proc.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt", device=self.device
        ).input_features
        with self.torch.inference_mode():
            ids = self.model.generate(
                feats.to(self.device, self.dtype),
                language=self.language,
                task="transcribe",
                num_beams=1,
                do_sample=False,
            )
        texts = self.proc.batch_decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return [t.strip() for t in texts]


def run_share(p: Project, device: str, k: int, n: int, items=None, listen=None) -> int:
    """Listen to every n-th clip of `items` (default: those still to do), from the k-th; append to
    clips/whisper/negatives-<k>.jsonl. Progress in logs/check-clips-<k>.json. Returns the clips done."""
    mine = (todo(p) if items is None else items)[k::n]
    out = _dir(p)
    out.mkdir(parents=True, exist_ok=True)
    progress = p.path("logs", f"check-clips-{k}.json")
    progress.parent.mkdir(parents=True, exist_ok=True)

    def report(done: int) -> None:
        progress.write_text(json.dumps({"done": done, "total": len(mine), "time": time.time()}))

    report(0)
    if not mine:
        return 0
    try:
        listen = listen or _Listener(device, p.language)
    except Exception as e:
        raise CantLoad(str(e)) from e
    with open(out / f"negatives-{k}.jsonl", "a") as fh:
        for i in range(0, len(mine), BATCH):
            group = mine[i : i + BATCH]
            files = [p.path("clips", s, name) for s, name in group]
            for (split, name), f, text in zip(group, files, listen(files), strict=True):
                rec = {"split": split, "file": name, "mtime": _mtime(f), "text": text}
                fh.write(json.dumps(rec) + "\n")
            fh.flush()  # a stopped run keeps what it heard
            report(min(i + BATCH, len(mine)))
    return len(mine)


def _free_gpu() -> None:
    import gc

    import torch

    gc.collect()
    for backend in (torch.cuda, getattr(torch, "xpu", None)):
        if backend is not None and backend.is_available():
            backend.empty_cache()


def run(p: Project, devices: list[str], log=print) -> dict:
    """Listen to the sound-alike clips not checked yet: in this process on one GPU, one process per GPU
    with several. Skipped (with the reason) without a GPU or Whisper, or when Whisper can't be loaded: the
    clips then count as not checked, are trained on as without a GPU, and the next run tries again."""
    for f in p.path("logs").glob("check-clips-*.json"):
        f.unlink()  # a share count from an earlier run would skew the progress
    items = todo(p)
    left = len(items)
    if not left:
        return {"skipped": "every sound-alike clip was checked before", **summary(p)}
    gpus = [d for d in devices if d != "cpu"]
    if not gpus:
        log(
            f"not checking the sound-alike clips: Whisper on the CPU would take hours ({left} clips)"
        )
        return {"skipped": "needs a GPU", "unchecked": left, **summary(p)}
    if why := available():
        log(f"not checking the sound-alike clips: {why}")
        return {"skipped": why, "unchecked": left, **summary(p)}
    log(f"Whisper listens to {len(items)} sound-alike clips on {', '.join(gpus)}")

    def cant_load(why: object) -> dict:
        log(f"not checking the sound-alike clips: Whisper couldn't be loaded: {why}")
        return {
            "skipped": "Whisper couldn't be loaded",
            "error": str(why)[:300],
            "unchecked": left,
            **summary(p),
        }

    if len(gpus) == 1:
        why = None
        try:
            run_share(p, gpus[0], 0, 1, items)
        except CantLoad as e:
            why = str(e)
        _free_gpu()  # out here: the exception above still holds a half-loaded model
        if why is not None:
            return cant_load(why)
        return {"listened": len(items), **summary(p)}
    # the list is made once: a card's process that starts later must not see what another one wrote
    listf = p.path("logs", "check-clips-todo.json")
    listf.write_text(json.dumps(items))
    procs = []
    for k, dev in enumerate(gpus):
        cmd = [
            sys.executable,
            "-m",
            "wakesmith.clipcheck",
            str(p.dir),
            dev,
            str(k),
            str(len(gpus)),
            str(listf),
        ]
        logf = open(p.path("logs", f"check-clips-{k}.log"), "a")  # noqa: SIM115 - closed below
        procs.append((subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT), logf, dev, k))
    failed, unloaded = [], []
    for proc, logf, dev, k in procs:
        rc = proc.wait()
        logf.close()
        if rc == CANT_LOAD_EXIT:
            unloaded.append(f"{dev} (logs/check-clips-{k}.log)")
        elif rc != 0:
            failed.append(f"{dev} (exit {rc}, logs/check-clips-{k}.log)")
    if failed:
        raise RuntimeError(f"checking the sound-alike clips failed on {', '.join(failed)}")
    if len(unloaded) == len(procs):
        return cant_load(f"on {', '.join(unloaded)}")
    if unloaded:  # the other cards heard their share; the rest stays to do for the next run
        log(
            f"Whisper couldn't be loaded on {', '.join(unloaded)}: their clips stay unchecked for now"
        )
        done = 0
        for f in p.path("logs").glob("check-clips-[0-9]*.json"):
            try:
                done += int(json.loads(f.read_text())["done"])
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return {"listened": done, **summary(p)}
    return {"listened": len(items), **summary(p)}


if __name__ == "__main__":
    proj_dir, device, k, n, listf = sys.argv[1:6]
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    items = [tuple(x) for x in json.loads(Path(listf).read_text())]
    try:
        print(json.dumps({"done": run_share(load(Path(proj_dir)), device, int(k), int(n), items)}))
    except CantLoad as e:
        print(f"Whisper couldn't be loaded: {e}")
        sys.exit(CANT_LOAD_EXIT)
