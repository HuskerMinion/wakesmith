"""Synthetic clip generation for a project: which phrases, how many, on which devices.

Parts: "positive" (positive_train + positive_test) and "negative" (negative_train + negative_test).
Each split is filled from every TTS model in turn (equal shares), so voices from both generators mix.
With several GPUs, each part runs in its own process on its own card (`python -m wakesmith.synth`).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import zlib
from pathlib import Path

from . import assets
from .project import Project, load
from .tts.generate import count_clips, generate

SPLITS = {
    "positive": [("positive_train", "positive"), ("positive_test", "positive_test")],
    "negative": [("negative_train", "negative"), ("negative_test", "negative_test")],
}


def phrases_for(p: Project, part: str) -> list[str]:
    if part == "positive":
        return [p.phrase, *p.positive_spellings]
    if not p.negatives:
        raise ValueError("the project lists no sound-alike phrases (negatives)")
    return list(p.negatives)


def run_part(p: Project, part: str, device: str, batch_size: int) -> dict:
    stats = {"written": 0, "failed": 0}
    texts = phrases_for(p, part)
    models = [assets.ASSETS[assets.TTS_MODELS[m]] for m in p.tts_models]
    for a in models:
        assets.fetch(a)
        assets.fetch(assets.ASSETS[a.name + "-config"])
    progress = p.path("logs", f"generate-{part}.json")
    progress.parent.mkdir(parents=True, exist_ok=True)
    for split, key in SPLITS[part]:
        n_total = getattr(p.samples, key)
        out = p.path("clips", split)
        for i, a in enumerate(models):
            share = n_total * (i + 1) // len(models)  # cumulative target after model i
            seed = zlib.crc32(f"{p.name}/{split}/{a.name}".encode())

            def report(done: int, n: int, split: str = split, total: int = n_total) -> None:
                progress.write_text(
                    json.dumps({"split": split, "done": done, "total": total, "time": time.time()})
                )

            r = generate(
                texts,
                out,
                share,
                a.local,
                device=device,
                batch_size=batch_size,
                seed=seed,
                on_progress=report,
            )
            stats["written"] += r["written"]
            stats["failed"] += r["failed"]
    return stats


def complete(p: Project) -> bool:
    return all(
        count_clips(p.path("clips", split)) >= getattr(p.samples, key)
        for part in SPLITS.values()
        for split, key in part
    )


def _free_gpu() -> None:
    """Hand cached GPU memory back to the driver once the speech models are gone."""
    import gc

    import torch

    gc.collect()
    for backend in (torch.cuda, getattr(torch, "xpu", None)):
        if backend is not None and backend.is_available():
            backend.empty_cache()


def run(p: Project, devices: list[str], batch_size: int, log=print) -> dict:
    """Generate every part, one process per device when there are several."""
    if len(devices) < 2:
        stats = {}
        for part in SPLITS:
            stats[part] = run_part(p, part, devices[0], batch_size)
        _free_gpu()  # in-process: torch's cache would hold the card (12 GB of 16) through training
        return stats
    procs = {}
    for part, dev in zip(SPLITS, devices):
        cmd = [sys.executable, "-m", "wakesmith.synth", str(p.dir), part, dev, str(batch_size)]
        logf = open(p.path("logs", f"generate-{part}.log"), "a")  # noqa: SIM115 - closed below
        procs[part] = (subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT), logf)
        log(f"generating {part} clips on {dev}")
    stats = {}
    for part, (proc, logf) in procs.items():
        rc = proc.wait()
        logf.close()
        if rc != 0:
            raise RuntimeError(f"generation of {part} clips failed: see logs/generate-{part}.log")
        stats[part] = json.loads(p.path("logs", f"generate-{part}.result.json").read_text())
    return stats


if __name__ == "__main__":
    proj_dir, part, device, batch = sys.argv[1:5]
    proj = load(Path(proj_dir))
    result = run_part(proj, part, device, int(batch))
    proj.path("logs", f"generate-{part}.result.json").write_text(json.dumps(result))
    print(json.dumps(result))
