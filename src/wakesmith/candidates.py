"""Best of N: train several microWakeWord candidates and keep the one that does best on held-out real
recordings.

Runs of one recipe on the same data differ as much as a change of data does (in one house's rounds, three
seeds fired on 3, 22 and 18 of the same 124 unseen device clips), so picking among a few is worth more than
most tuning. Each candidate is kept in mww/candidates/s<N>/ with its seed, its own validation score and its
held-out scores; mww/candidates/pick.json says which one was chosen and why, and that one becomes the
project's model (mww/trained/).

The pick, at the project's cutoff: only candidates that catch at least MIN_CAUGHT_SHARE (70%) as many
held-out positives as the best one compete (a seed that barely fires has few false triggers but is no
model); among them the fewest false triggers on held-out real negatives decides first (a quarter more, at
least one, still counts as tied), then the most held-out positives caught, then fewer false triggers, then
microWakeWord's validation score. Without held-out recordings it falls back to the validation score alone.

The pick sees only the "pick" half of the held-out recordings (evaluate.split); the test stage reports the
kept model on the other half, so its numbers aren't flattered by the choice. pick.json records every
candidate on both halves and on the whole set. Too few held-out clips to split: the whole set, as before.

With several GPUs, candidates train side by side, one per GPU (each in mww/candidates/s<N>.part/, renamed
to s<N> once it has finished, so an s<N> folder is always a whole candidate; with its own log,
logs/mww-train-s<N>.log, appended to logs/mww-train.log when it ends); logs/mww-candidates.json says
which are done and which are training, for the progress bar. One GPU or the CPU: one after another.

A new best-of-N run never deletes the last round: mww/candidates moves to mww/candidates.previous (each
candidate slimmed to KEEP_CANDIDATE, a few hundred KB) and mww/combined to mww/combined.previous, replacing
any older .previous, as mww/trained.previous works for the model. A round stopped before its pick doesn't
replace a finished round kept there (it is dropped, with a combined model built from it), and one with no
finished candidate is dropped too; with no finished round kept, a stopped round's finished candidates are
kept as before. A combined model always stays with the round it was built from (combine --previous builds
one from candidates.previous: it leaves when that round does). `wakesmith combine --previous` averages
the previous round's candidates.
"""

from __future__ import annotations

import json
import os
import queue
import random
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import evaluate, mww
from .project import Project
from .real import RealData

KEEP_CANDIDATE = {  # a few hundred KB per candidate: enough to average or rescore it later
    "best_weights.weights.h5",
    "last_weights.weights.h5",
    "stream_state_internal",
    "tflite_stream_state_internal_quant",
}
# a candidate catching under this share of the most positives any candidate caught can't be kept
MIN_CAUGHT_SHARE = 0.7
_BEST = re.compile(r"^best: step (\d+), (\S+) = ([\d.]+)", re.MULTILINE)


def _validation(text: str) -> float | None:
    """The validation score a candidate's training log ends with."""
    m = _BEST.findall(text)
    return float(m[-1][2]) if m else None


def choose(cands: list[dict], on: str = "") -> tuple[dict, str]:
    """The candidate to keep and a one-line reason. on: which held-out clips the numbers are from."""
    ok = [c for c in cands if not c.get("failed")]
    if not ok:
        raise RuntimeError("every microWakeWord candidate failed to train")
    val = lambda c: c.get("validation") or 0.0
    if all("false" in c for c in ok):
        floor = MIN_CAUGHT_SHARE * max(c["caught"] for c in ok)
        able = [c for c in ok if c["caught"] >= floor]  # never empty: the best one is in it
        fmin = min(c["false"] for c in able)
        slack = max(1, round(0.25 * fmin))
        tied = [c for c in able if c["false"] <= fmin + slack]
        best = max(tied, key=lambda c: (c["caught"], -c["false"], val(c)))
        out = len(ok) - len(able)
        return best, (
            f"s{best['n']}: {best['caught']}/{best['positives']} held-out positives caught, "
            f"{best['false']}/{best['negatives']} false triggers at {best['cutoff']:.2f}"
            + (f" on the {on}" if on else "")
            + f" (the fewest was {fmin}; candidates within {slack} of it compete on positives caught"
            + (
                f"; {out} that caught under {round(100 * MIN_CAUGHT_SHARE)}% of the best one's"
                f" {max(c['caught'] for c in ok)} left out"
                if out
                else ""
            )
            + ")"
        )
    best = max(ok, key=val)
    return best, f"s{best['n']}: best validation score ({val(best):.4f}; no held-out recordings)"


def progress_path(p: Project) -> Path:
    return p.path("logs", "mww-candidates.json")


def train(p: Project, log_file: Path, device: str | list[str], log=print) -> dict | None:
    """Best of N on `device`: one device, or a list of GPUs that each train one candidate at a time."""
    devices = [device] if isinstance(device, str) else list(device) or ["cpu"]
    if os.environ.get("WAKESMITH_MWW_TRAINER") == "tf":  # microWakeWord's own trainer: CPU only
        devices = devices[:1]
    n = max(1, p.microwakeword.candidates)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    progress_path(p).parent.mkdir(parents=True, exist_ok=True)
    for f in (progress_path(p), progress_path(p).with_name(progress_path(p).name + ".part")):
        f.unlink(missing_ok=True)  # left by a run that was killed
    for f in p.path("logs").glob("mww-train-s*.log"):
        f.unlink(missing_ok=True)
    for f in [*p.path("mww").glob("training-s*.yaml"), *p.path("mww").glob(".training*.tmp")]:
        f.unlink(missing_ok=True)
    for d in p.path("mww", "candidates").glob("s*.part"):  # a killed run's half-trained candidates
        shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(p.path("mww", "trained.part"), ignore_errors=True)  # a killed run's half copy
    restore_model(p)  # a killed run moved the model aside: it's the model
    if n == 1:
        try:
            mww.train(p, log_file, device=devices[0])
        except BaseException:  # failed or stopped: the model from before is the model again
            restore_model(p)
            raise
        return None
    out = p.path("mww", "candidates")
    keep_previous(p, log)
    out.mkdir(parents=True)
    trained = p.path(
        "mww", "trained"
    )  # the last model stays as trained.previous, as with one candidate
    if mww.model_path(p).exists():
        shutil.rmtree(p.path("mww", "trained.previous"), ignore_errors=True)
        trained.rename(p.path("mww", "trained.previous"))
    else:  # no model in it (restore_model left nothing better): never put it over the previous one
        shutil.rmtree(trained, ignore_errors=True)
    held = bool(RealData(p)) and any(p.path("real_neg", "test").glob("*.wav"))
    cutoff = p.microwakeword.cutoff
    slots = min(len(devices), n)
    seeds = {i: random.SystemRandom().randrange(2**31) for i in range(1, n + 1)}
    free: queue.Queue[str] = queue.Queue()
    for d in devices[:slots]:
        free.put(d)
    lock = threading.Lock()
    stop = threading.Event()  # set when the run ends early: candidates not started yet don't start
    state = {"n": n, "slots": slots, "done": [], "running": {}}

    def note() -> None:  # under `lock`; after the run ended nothing writes it again
        if stop.is_set():
            return
        state["time"] = time.time()
        path = progress_path(p)
        tmp = path.with_name(path.name + ".part")
        try:
            tmp.write_text(json.dumps(state))
            tmp.replace(path)  # the page never reads half a file
        except OSError:
            pass

    def one(i: int) -> tuple[int, str | None, str]:
        dev = free.get()
        clog = p.path("logs", f"mww-train-s{i}.log")
        err, text = None, ""
        try:  # from here on the device always goes back, whatever fails
            if stop.is_set():
                return i, "not started: the run ended", ""
            clog.unlink(missing_ok=True)
            with lock:
                log(
                    f"mww candidate {i} of {n} (seed {seeds[i]})"
                    + (f" on {dev}" if slots > 1 else "")
                )
                state["running"][str(i)] = clog.name
                note()
            part = out / f"s{i}.part"  # becomes s<N> only once the candidate is whole
            try:
                mww.train(
                    p,
                    clog,
                    device=dev,
                    seed=seeds[i],
                    train_dir=f"candidates/{part.name}",
                    share=slots,
                    stop=stop,
                )
                part.rename(out / f"s{i}")
            except RuntimeError as e:
                err = str(e).replace(str(clog), str(log_file))
                shutil.rmtree(part, ignore_errors=True)
            except BaseException:  # anything else ends the run: no other candidate starts after it
                stop.set()
                raise
            with lock:
                text = clog.read_text("utf-8", "replace") if clog.exists() else ""
                with open(log_file, "a") as f:
                    f.write(f"\n===== candidate s{i}\n{text}")
                state["running"].pop(str(i), None)
                state["done"].append(i)
                note()
            return i, err, text
        finally:
            free.put(dev)

    cands: list[dict] = []
    halves: dict | None = None
    split_note: str | None = None
    pool = ThreadPoolExecutor(slots)
    try:
        for fut in as_completed([pool.submit(one, i) for i in range(1, n + 1)]):
            i, err, text = fut.result()
            c: dict = {"n": i, "seed": seeds[i]}
            if err:
                shutil.rmtree(out / f"s{i}", ignore_errors=True)  # a half-written candidate
                p.path("logs", f"mww-train-s{i}.log").unlink(missing_ok=True)  # in the main log
                log(f"mww candidate {i} failed: {err}")
                cands.append({**c, "failed": err})
                continue
            p.path("logs", f"mww-train-s{i}.log").unlink(missing_ok=True)
            d = out / f"s{i}"
            c["validation"] = _validation(text)
            if held:
                model = d / mww.model_path(p).relative_to(p.path("mww", "trained"))
                r = evaluate.score(p, "mww", model=model, tag=f"s{i}")
                if split_note is None:
                    halves, split_note = evaluate.split(r["pos"], r["neg"])
                    log(f"mww: candidates {split_note}")
                c.update(scored(r, cutoff, halves))
            log(
                f"mww candidate {i}: validation {c['validation']}"
                + (
                    f", held-out{' (pick half)' if halves else ''} {c['caught']}/{c['positives']}"
                    f" caught, {c['false']} false"
                    if held
                    else ""
                )
            )
            cands.append(c)
    except (
        BaseException
    ):  # Ctrl-C, a candidate's OSError, an error while scoring: the waiting ones don't
        # start and the running ones are ended (mww.train watches `stop`, within seconds)
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
        _abandon(p, trained)
        raise
    else:
        pool.shutdown()
    finally:
        with (
            lock
        ):  # a candidate still finishing can't write it back (note() checks stop under the lock)
            stop.set()
            progress_path(p).unlink(missing_ok=True)
            progress_path(p).with_name(progress_path(p).name + ".part").unlink(missing_ok=True)
    cands.sort(key=lambda c: c["n"])
    try:
        best, why = choose(
            cands, "half of the held-out recordings used for choosing" if halves else ""
        )
    except RuntimeError:  # every candidate failed
        _abandon(p, trained)
        raise
    part = p.path("mww", "trained.part")  # copied whole, then put in place: never half a model
    shutil.rmtree(part, ignore_errors=True)
    try:
        shutil.copytree(out / f"s{best['n']}", part)
        shutil.rmtree(trained, ignore_errors=True)
        part.rename(trained)
    except BaseException:  # a full disk, Ctrl-C: the model from before stays the model
        shutil.rmtree(part, ignore_errors=True)
        _abandon(p, trained)
        raise
    pick = {"picked": best["n"], "why": why, "candidates": cands}
    if held:
        pick.update(split=halves, split_note=split_note)
    (out / "pick.json").write_text(json.dumps(pick, indent=1))
    log(f"mww: kept candidate {why}")
    return {"candidates": n, "picked": best["n"], "why": why, "slots": slots}


def restore_model(p: Project) -> None:
    """mww/trained.previous back as mww/trained when mww/trained has no model (missing, or half
    written by a run that was stopped or killed)."""
    trained, prev = p.path("mww", "trained"), p.path("mww", "trained.previous")
    if prev.exists() and not mww.model_path(p).exists():
        shutil.rmtree(trained, ignore_errors=True)
        prev.rename(trained)


def _abandon(p: Project, trained: Path) -> None:
    """A round that didn't finish: the model from before it is the model again, and the candidates'
    own logs go (their text is in the main log), as do the candidates stopped half way; the ones that
    finished stay, as after a stop in older versions (combine can still use them)."""
    for f in p.path("logs").glob("mww-train-s*.log"):
        f.unlink(missing_ok=True)
    for d in p.path("mww", "candidates").glob("s*.part"):  # stopped half way
        shutil.rmtree(d, ignore_errors=True)
    restore_model(p)


def scored(r: dict, cutoff: float, halves: dict | None) -> dict:
    """A candidate's held-out numbers at the cutoff: caught/false/positives/negatives on what the pick
    may see (the pick half, or everything without a split), plus "report" (the other half) and "full"."""
    full = evaluate.at(r, cutoff)
    if not halves:
        return {"cutoff": cutoff, **full}
    return {
        "cutoff": cutoff,
        **evaluate.at(evaluate.half(r, "pick"), cutoff),
        "report": evaluate.at(evaluate.half(r, "report"), cutoff),
        "full": full,
    }


def _drop(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _whole(cands: Path) -> bool:
    """Any finished candidate in the folder (s<N>; one still training is s<N>.part)?"""
    return any(d.is_dir() and d.name[1:].isdigit() for d in cands.glob("s*"))


def keep_previous(p: Project, log=print) -> None:
    """Move the last round's candidates (and its combined model) aside as *.previous, replacing older
    ones, and slim the candidates to their weights and models (KEEP_CANDIDATE)."""
    root = p.path("mww")
    cands, combined = root / "candidates", root / "combined"
    moved = []
    try:  # built with combine --previous: it belongs to the round kept in candidates.previous
        of_previous = bool(json.loads((combined / "combined.json").read_text()).get("previous"))
    except (OSError, ValueError, AttributeError):
        of_previous = False
    stopped = (
        os.path.lexists(cands) and not cands.is_symlink() and not (cands / "pick.json").exists()
    )
    if stopped and (not _whole(cands) or (root / "candidates.previous" / "pick.json").exists()):
        # a round stopped before its pick: it doesn't replace a round that finished (kept as
        # candidates.previous, with its combined model), and with no finished candidate there is
        # nothing in it to keep; otherwise it's kept as before, for combine --previous
        _drop(cands)
        if os.path.lexists(combined) and not of_previous:
            _drop(combined)  # built from the dropped round: it goes with it
    if os.path.lexists(cands):
        prev = root / "candidates.previous"
        _drop(prev)
        cands.rename(prev)
        for d in prev.glob("s*"):
            if d.is_dir() and not d.is_symlink():
                for x in d.iterdir():
                    if x.name not in KEEP_CANDIDATE:
                        _drop(x)
        moved.append("candidates")
        if os.path.lexists(combined) and of_previous:
            _drop(combined)  # built from the round just replaced: it goes with that round
        if not os.path.lexists(combined):  # an older combined model belongs to older candidates
            _drop(root / "combined.previous")
    if os.path.lexists(combined):
        _drop(root / "combined.previous")
        combined.rename(root / "combined.previous")
        moved.append("combined model")
    if moved:
        log(f"mww: kept the previous round's {' and '.join(moved)} in mww/*.previous")
