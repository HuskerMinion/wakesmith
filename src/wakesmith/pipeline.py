"""The training pipeline as resumable stages, with status in <project>/state.json.

generate      synthetic clips of the phrase and its sound-alikes (GPU when available)
real          prepare real recordings (held-out split, review corrections, trimming)
mww-features  microWakeWord features (all CPU cores)
mww-train     microWakeWord training (CPU)
oww-features  openWakeWord features (all CPU cores)
oww-train     openWakeWord DNN head (CPU) and .tflite export
test          score held-out real recordings with every trained model
export        device-ready files in <project>/export
"""

from __future__ import annotations

import logging
import os
import time
import traceback
from collections.abc import Callable
from datetime import datetime

from . import commonvoice, eta, evaluate, export, hardware, mww, oww, runs, synth
from .project import Project
from .real import RealData

STAGES = [
    "generate",
    "real",
    "commonvoice",
    "mww-features",
    "mww-train",
    "oww-features",
    "oww-train",
    "test",
    "export",
]


def enabled(p: Project, stage: str) -> bool:
    if stage.startswith("mww"):
        return p.microwakeword.enabled
    if stage.startswith("oww"):
        return p.openwakeword.enabled
    if stage == "commonvoice":
        return p.commonvoice.enabled
    return True


def run(
    p: Project,
    first: str = STAGES[0],
    last: str = STAGES[-1],
    log: Callable[[str], None] | None = None,
) -> None:
    logf = p.path("logs", "train.log")
    logf.parent.mkdir(parents=True, exist_ok=True)

    def say(msg: str) -> None:
        line = f"{datetime.now():%H:%M:%S} {msg}"
        with open(logf, "a") as f:
            f.write(line + "\n")
        (log or print)(line)

    # Only speech generation uses the GPU: other stages must not open the driver at all.
    if "generate" in STAGES[STAGES.index(first) : STAGES.index(last) + 1]:
        hw, problem = hardware.detect_safe()
        if problem:
            say(f"warning: {problem}; generating on the CPU")
    else:
        hw = hardware.detect(gpu=False)
    for key, val in p.hardware.items():  # project overrides
        setattr(hw, key, val)
    stages = STAGES[STAGES.index(first) : STAGES.index(last) + 1]
    if low := hardware.disk_warning(str(p.dir)):
        say(f"warning: {low}")
    if p.microwakeword.enabled and (low := hardware.ram_warning(hw)):
        say(f"warning: {low}")
    say(
        f"hardware: {hw.accel}, {len(hw.gpus)} GPU(s), {hw.cpus} CPUs, "
        f"{hw.ram_free_gb:.0f}/{hw.ram_gb:.0f} GB RAM free; stages: {', '.join(stages)}"
    )
    for stage in stages:
        if not enabled(p, stage):
            continue
        p.set_state(**{stage: {"status": "running", "started": time.time()}})
        say(f"=== {stage}")
        t0 = time.time()
        try:
            info = _run_stage(p, stage, hw, say)
        except Exception as e:
            p.set_state(**{stage: {"status": "failed", "error": str(e), "time": time.time() - t0}})
            say(f"{stage} failed: {e}")
            logging.getLogger(__name__).debug(traceback.format_exc())
            with open(logf, "a") as f:
                f.write(traceback.format_exc())
            raise
        p.set_state(**{stage: {"status": "done", "time": time.time() - t0, "info": info or {}}})
        eta.record(p, stage, time.time() - t0, info)
        say(f"{stage} done in {(time.time() - t0) / 60:.1f} min {info or ''}")
    if "export" in stages and (rid := runs.snapshot(p)):
        say(f"run kept as {rid} (compare it with earlier runs on the Compare page)")


def _tf_off_gpu() -> None:
    """Training runs on the CPU (the tested path; the trainers' own processes already hide the
    GPU). Hide it from this process too: TensorFlow, loaded here for the features, otherwise takes
    a CUDA card's whole memory and keeps it while doing nothing on it (RTX 5060 Ti under WSL2:
    15.8 of 16 GB held at 1% load). WAKESMITH_TF_GPU=<devices> (e.g. 0) lets TensorFlow use them."""
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("WAKESMITH_TF_GPU", "-1")


def _run_stage(p: Project, stage: str, hw: hardware.Hardware, say) -> dict | None:
    if stage != "generate":  # only speech generation uses the GPU
        _tf_off_gpu()
    if stage == "generate":
        if synth.complete(p):
            return {"skipped": "all clips present"}
        return synth.run(p, hw.tts_devices(), hw.tts_batch(), log=say)
    if stage == "real":
        real = RealData(p)
        return real.prepare() if real else {"skipped": "no real recordings"}
    if stage == "commonvoice":
        return commonvoice.prepare(p, log=say)
    if stage == "mww-features":
        mww.build_features(p, hw.workers(), log=say)
        return None
    if stage == "mww-train":
        mww.train(p, p.path("logs", "mww-train.log"))
        return None
    if stage == "oww-features":
        oww.build_features(p, hw.workers(), log=say)
        return None
    if stage == "oww-train":
        oww.train(p, log=say)
        return None
    if stage == "test":
        cv_test = p.path("commonvoice", "test")
        has_cv = p.commonvoice.enabled and any(cv_test.glob("*.wav"))
        if not RealData(p) and not has_cv:
            return {"skipped": "no held-out recordings"}
        out = {}
        for kind, on in (("mww", p.microwakeword.enabled), ("oww", p.openwakeword.enabled)):
            if on and RealData(p):
                r = evaluate.score(p, kind)
                say(evaluate.format_table(r))
                out[kind] = r["table"]
            if on and has_cv:
                r = evaluate.score(p, kind, positives=cv_test, tag="cv")
                say("Common Voice speakers (never trained on): " + evaluate.format_table(r))
                out[f"{kind}-cv"] = r["table"]
        return {"results": "results/*.json"} if out else None
    if stage == "export":
        return {"files": [str(f) for f in export.export(p)]}
    raise ValueError(f"unknown stage {stage}")
