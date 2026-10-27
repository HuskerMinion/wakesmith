"""The training pipeline as resumable stages, with status in <project>/state.json.

generate      synthetic clips of the phrase and its sound-alikes (GPU when available)
real          prepare real recordings (held-out split, review corrections, trimming)
commonvoice   real volunteer voices from Common Voice (only when enabled)
conversational  everyday sentences from an AI assistant, spoken as not-the-phrase clips (experimental;
              only when conversational.clips > 0)
mww-features  microWakeWord features (all CPU cores)
mww-train     microWakeWord training (GPU if found); with candidates > 1, the best of N on held-out
              recordings (candidates.py)
oww-features  openWakeWord features (all CPU cores; the embedding model on the GPU if found)
oww-train     openWakeWord DNN head (GPU if found) and .tflite export
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

from . import (
    candidates,
    commonvoice,
    conversational,
    ensemble,
    eta,
    evaluate,
    export,
    hardware,
    models,
    mww,
    oww,
    runs,
    synth,
    verdict,
)
from .project import Project
from .real import RealData

STAGES = [
    "generate",
    "real",
    "commonvoice",
    "conversational",
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
    if stage == "conversational":
        return p.conversational.clips > 0
    return True


# What each stage reads that an earlier stage made. Archiving (archive.py) removes some of that and marks
# the maker "archived"; such makers rerun first whenever a stage that reads their output runs.
NEEDS = {
    "commonvoice": ("real",),  # it adds takes to real/train
    "mww-features": ("generate", "real", "commonvoice"),
    "mww-train": ("mww-features", "real"),  # best of N picks on real_neg/test
    "oww-features": ("generate", "real", "commonvoice"),
    "oww-train": ("oww-features",),
    "test": ("real", "commonvoice"),
}


def rebuild_first(p: Project, stages: list[str]) -> list[str]:
    """Archived stages outside `stages` whose output these stages read (directly, or through
    another archived stage), in pipeline order."""
    st = p.state()

    def archived(s: str) -> bool:
        return (st.get(s) or {}).get("status") == "archived" and enabled(p, s)

    want: set[str] = set()
    todo = [s for s in stages if enabled(p, s)]
    while todo:
        for n in NEEDS.get(todo.pop(), ()):
            if n not in want and n not in stages and archived(n):
                want.add(n)
                todo.append(n)
    return [s for s in STAGES if s in want]


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

    stages = STAGES[STAGES.index(first) : STAGES.index(last) + 1]
    if extra := rebuild_first(p, stages):
        say(f"rebuilding what archiving removed first: {', '.join(extra)}")
        stages = [s for s in STAGES if s in stages or s in extra]
    # Speech generation, both models' training and openWakeWord features use the GPU: other stages must
    # not open the driver at all.
    gpu_stages = {"generate", "mww-train", "oww-features", "oww-train"}
    if p.conversational.clips > 0:
        gpu_stages.add("conversational")
    if gpu_stages & set(stages):
        hw, problem = hardware.detect_safe()
        if problem:
            say(f"warning: {problem}; using the CPU")
    else:
        hw = hardware.detect(gpu=False)
    for key, val in p.hardware.items():  # project overrides
        setattr(hw, key, val)
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
        p.set_state(**{stage: {"status": "running", "started": time.time(), "pid": os.getpid()}})
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
        models.attach_run(p, rid)
        say(f"run kept as {rid} (compare it with earlier runs on the Compare page)")


def _tf_off_gpu() -> None:
    """Hide CUDA cards from TensorFlow in this process: loaded here for the features, it otherwise takes a
    card's whole memory and keeps it while doing nothing on it (RTX 5060 Ti under WSL2: 15.8 of 16 GB held
    at 1% load). WAKESMITH_TF_GPU=<devices> (e.g. 0) lets TensorFlow use them. The original setting is kept
    in WAKESMITH_CUDA_DEVICES ("all": none was set) for mww.train's PyTorch trainer."""
    os.environ.setdefault("WAKESMITH_CUDA_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "all"))
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("WAKESMITH_TF_GPU", "-1")


def _run_stage(p: Project, stage: str, hw: hardware.Hardware, say) -> dict | None:
    # TensorFlow stays off the GPU (mww-train hands it back to PyTorch); speech generation keeps it
    if stage not in ("generate", "conversational"):
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
    if stage == "conversational":
        info = conversational.run(p, hw.tts_devices()[0], hw.tts_batch(), log=say)
        synth._free_gpu()
        return info
    if stage == "mww-features":
        mww.build_features(p, hw.workers(), log=say)
        return None
    if stage == "mww-train":
        info = candidates.train(
            p, p.path("logs", "mww-train.log"), device=mww.train_device(hw.gpus), log=say
        )
        if p.microwakeword.combine and info:
            try:
                ensemble.combine(p, log=say)
                info["combined"] = ensemble.note(ensemble.info(p) or {"candidates": []})
            except (RuntimeError, ValueError) as e:
                # the kept candidate is a whole model on its own: carry on with it
                say(f"warning: {e}; `wakesmith combine {p.name}` tries again")
                info["combined"] = f"failed: {e}"
        return info
    if stage == "oww-features":
        return oww.build_features(p, hw.workers(), log=say, device=oww.feature_device(hw.gpus))
    if stage == "oww-train":
        oww.train(p, log=say, device=oww.train_device(hw.gpus))
        return None
    if stage == "test":
        cv_test = p.path("commonvoice", "test")
        has_cv = p.commonvoice.enabled and any(cv_test.glob("*.wav"))
        if not RealData(p) and not has_cv:
            return {"skipped": "no held-out recordings"}
        out = {}
        for kind, on in (("mww", p.microwakeword.enabled), ("oww", p.openwakeword.enabled)):
            if on and RealData(p):
                r = evaluate.score(p, kind, report_half=True)
                say(evaluate.format_table(r))
                out[kind] = r["table"]
            if on and has_cv:
                r = evaluate.score(p, kind, positives=cv_test, tag="cv", report_half=True)
                say("Common Voice speakers (never trained on): " + evaluate.format_table(r))
                out[f"{kind}-cv"] = r["table"]
        out.update(evaluate.score_combined(p, log=say))
        for kind in ("mww", "oww"):  # side by side with the version on the devices, in one sentence
            if kind in out:
                try:
                    v = verdict.compare(p, kind, log=say)
                    if v and v.get("sentence"):
                        say(v["sentence"])
                except Exception as e:  # noqa: BLE001 - a verdict is a convenience; the test itself stands
                    say(f"verdict ({kind}) skipped: {e}")
        return {"results": "results/*.json"} if out else None
    if stage == "export":
        return {"files": [str(f) for f in export.export(p)]}
    raise ValueError(f"unknown stage {stage}")
