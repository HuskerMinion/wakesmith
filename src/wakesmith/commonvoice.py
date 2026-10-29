"""Real volunteer voices from Mozilla Common Voice (CC0) for phrases made of common words.

The dataset is the Common Voice 17.0 English layout (as on the Hugging Face mirror
fsicoli/common_voice_17_0): transcript/en/{train,dev,test}.tsv and audio/en/<split>/*.tar of MP3s.
`wakesmith commonvoice download` fetches it (about 47 GB); set commonvoice.dataset to use a copy you
already have.

For a phrase, wakesmith finds the takes whose transcript contains it word for word, cuts the phrase
out with forced alignment on the known transcript (torchaudio's MMS aligner; its weights are
CC BY-NC 4.0, fetched by torchaudio on first use), and keeps the cuts per phrase under
<data>/commonvoice/<phrase>/, so later runs and other projects reuse them.

Takes are split by speaker: speakers in the test share are never trained on and give the
"Common Voice speakers" result; the others can be added to training.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import tarfile
import unicodedata
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf

from . import assets
from .project import Project

SR = 16000
PAD_S = 0.15
MIN_ALIGN = 0.5  # mean alignment score below this: probably misread or misaligned, left out
SPLITS = ("train", "dev", "test")
HF_REPO = "fsicoli/common_voice_17_0"
csv.field_size_limit(1 << 30)


def words(text: str) -> list[str]:
    t = unicodedata.normalize("NFKD", text.lower().replace("’", "'"))
    out = (re.sub(r"[^a-z']", "", w).replace("'", "") for w in t.split())
    return [w for w in out if w]


def dataset(p: Project | None = None) -> Path:
    s = p.commonvoice.dataset if p is not None else ""
    return Path(s).expanduser() if s else assets.home() / "common_voice_17"


def present(ds: Path) -> bool:
    return all((ds / "transcript/en" / f"{s}.tsv").exists() for s in SPLITS) and any(
        (ds / "audio/en").glob("*/*.tar")
    )


def _slug(phrase: str) -> str:
    return "_".join(words(phrase)) or "phrase"


def _home(phrase: str) -> Path:
    return assets.home() / "commonvoice" / _slug(phrase)


def _find(ws: list[str], target: list[str]) -> int:
    for i in range(len(ws) - len(target) + 1):
        if ws[i : i + len(target)] == target:
            return i
    return -1


def matches(ds: Path, phrase: str) -> dict[str, dict]:
    """Takes whose transcript contains the phrase: {mp3 name: {split, speaker, sentence}}."""
    target = words(phrase)
    out = {}
    for split in SPLITS:
        with open(ds / "transcript/en" / f"{split}.tsv", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                if _find(words(r["sentence"]), target) >= 0:
                    out[r["path"]] = {
                        "split": split,
                        "speaker": r["client_id"],
                        "sentence": r["sentence"],
                    }
    return out


def count(ds: Path, phrase: str) -> dict:
    """How many takes and speakers Common Voice has for the phrase (cached per dataset and phrase)."""
    cache = assets.home() / "commonvoice" / "counts.json"
    key = f"{ds.resolve()}|{' '.join(words(phrase))}"
    known = json.loads(cache.read_text()) if cache.exists() else {}
    if key not in known:
        m = matches(ds, phrase)
        known[key] = {"takes": len(m), "speakers": len({v["speaker"] for v in m.values()})}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(known, indent=1))
    return {**known[key], "extracted": (_home(phrase) / "index.tsv").exists()}


def extract(ds: Path, phrase: str, log: Callable[[str], None] = print) -> Path:
    """Cut the phrase out of every matching take (once per phrase). Returns the cut folder."""
    home = _home(phrase)
    if (home / "index.tsv").exists():
        return home
    import torch
    import torchaudio
    from scipy.signal import resample_poly

    want = matches(ds, phrase)
    target = words(phrase)
    log(
        f"Common Voice: {len(want)} takes with '{phrase}'; cutting the phrase out (one pass over the audio)"
    )
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star=False).eval()
    tok, aligner = bundle.get_tokenizer(), bundle.get_aligner()
    tmp = home.with_name(home.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "cuts").mkdir(parents=True)
    rows = []
    tars = sorted((ds / "audio/en").glob("*/*.tar"))
    for n, tar in enumerate(tars, 1):
        with tarfile.open(tar) as tf:
            for m in tf:
                name = m.name.rsplit("/", 1)[-1]
                info = want.pop(name, None)
                if info is None:
                    continue
                a, sr = sf.read(
                    io.BytesIO(tf.extractfile(m).read()), dtype="float32", always_2d=True
                )
                a = a[:, 0] if sr == SR else resample_poly(a[:, 0], SR, sr).astype(np.float32)
                ws = words(info["sentence"])
                i = _find(ws, target)
                if i < 0:
                    continue
                with torch.inference_mode():
                    em, _ = model(torch.from_numpy(a)[None])
                    spans = aligner(em[0], tok(ws))
                ratio = len(a) / em.shape[1] / SR
                s, e = spans[i][0].start * ratio, spans[i + len(target) - 1][-1].end * ratio
                score = float(np.mean([t.score for w in spans[i : i + len(target)] for t in w]))
                cut = a[max(0, int((s - PAD_S) * SR)) : int((e + PAD_S) * SR)]
                stem = name.rsplit(".", 1)[0]
                sf.write(tmp / "cuts" / f"{stem}.wav", cut, SR, subtype="PCM_16")
                rows.append(
                    [
                        stem,
                        info["split"],
                        info["speaker"],
                        f"{s:.3f}",
                        f"{e:.3f}",
                        f"{score:.3f}",
                        info["sentence"],
                    ]
                )
        log(f"Common Voice: {n}/{len(tars)} audio files read, {len(rows)} cuts")
        if not want:
            break
    with open(tmp / "index.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["clip", "split", "speaker", "start", "end", "score", "sentence"])
        w.writerows(rows)
    shutil.rmtree(home, ignore_errors=True)
    tmp.rename(home)
    return home


def _is_test(speaker: str, share: float) -> bool:
    return int(hashlib.sha1(speaker.encode()).hexdigest(), 16) % 1000 < share * 1000


def prepare(p: Project, log: Callable[[str], None] = print) -> dict:
    """Common Voice stage: the held-out speakers' cuts to commonvoice/test, and (if training on them)
    the other speakers' cuts into real/train next to your own takes."""
    from .real import trim_to_phrase

    if p.language != "en":
        raise RuntimeError(
            "Common Voice here is the English set: turn it off for this wake word (Real voices page)"
        )
    cv = p.commonvoice
    ds = dataset(p)
    if not present(ds):
        raise RuntimeError(
            f"Common Voice isn't at {ds}: run `wakesmith commonvoice download`, or set commonvoice.dataset"
        )
    home = extract(ds, p.phrase, log)
    with open(home / "index.tsv", newline="") as f:
        rows = [
            r
            for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            if float(r["score"]) >= MIN_ALIGN
        ]
    test_dir, train_dir = p.path("commonvoice", "test"), p.path("real", "train")
    shutil.rmtree(test_dir, ignore_errors=True)
    test_dir.mkdir(parents=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    for old in train_dir.glob("cv__*.wav"):
        old.unlink()
    test = [r for r in rows if _is_test(r["speaker"], cv.test_share)]
    train = sorted(
        (r for r in rows if not _is_test(r["speaker"], cv.test_share)),
        key=lambda r: hashlib.sha1(r["clip"].encode()).hexdigest(),
    )[: cv.max_train]
    for r in test:
        shutil.copy(home / "cuts" / f"{r['clip']}.wav", test_dir / f"{r['clip']}.wav")
    used = 0
    if cv.train:
        for r in train:
            a, _ = sf.read(home / "cuts" / f"{r['clip']}.wav", dtype="float32")
            t = trim_to_phrase(a, 0.0)
            sf.write(
                train_dir / f"cv__{r['clip']}.wav", a if t is None else t, SR, subtype="PCM_16"
            )
            used += 1
    return {
        "cv_test": len(test),
        "cv_test_speakers": len({r["speaker"] for r in test}),
        "cv_train": used,
        "cv_train_speakers": len({r["speaker"] for r in train}) if cv.train else 0,
    }


def download(log: Callable[[str], None] = print) -> Path:
    """Fetch Common Voice 17.0 English (validated splits, about 47 GB) from the Hugging Face mirror."""
    from huggingface_hub import snapshot_download

    ds = assets.home() / "common_voice_17"
    pats = [f"transcript/en/{s}.tsv" for s in SPLITS] + [f"audio/en/{s}/*" for s in SPLITS]
    log(f"downloading Common Voice 17.0 English (CC0, about 47 GB) to {ds}")
    snapshot_download(
        HF_REPO, repo_type="dataset", local_dir=str(ds), allow_patterns=pats, max_workers=4
    )
    return ds
