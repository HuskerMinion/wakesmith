"""Best of N on several GPUs: one candidate per GPU at a time (candidates.train with a list of devices)."""

import json
import os
import threading
import time

import pytest

from wakesmith import candidates, eta, jobs, mww, project


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path / "projects"))
    return project.create("hey jarvis", negatives=["jarvis", "hey harvest"])


@pytest.fixture
def fake_train(proj, monkeypatch):
    """mww.train stand-in: records each call, sleeps a little so runs overlap, writes a model that names it."""
    calls, now, lock = [], {"n": 0, "most": 0}, threading.Lock()

    def train(p, log_file, device, seed=None, train_dir="trained", share=1, stop=None):
        with lock:
            now["n"] += 1
            now["most"] = max(now["most"], now["n"])
            calls.append({"device": device, "dir": train_dir, "share": share, "seed": seed})
        log_file.write_text(
            f"Validation Batch #1: Mini-Batch #10\nbest: step 500, recall = 0.{len(calls)}\n"
        )
        time.sleep(0.2)
        m = p.path("mww", train_dir) / mww.model_path(p).relative_to(p.path("mww", "trained"))
        m.parent.mkdir(parents=True)
        m.write_bytes(train_dir.encode())
        with lock:
            now["n"] -= 1
        cand = train_dir.split("/")[-1].split(".")[0]  # "candidates/s3.part" -> "s3"
        if os.environ.get("FAKE_OSERROR") == cand:
            raise OSError("disk full")
        if os.environ.get("FAKE_FAIL") == cand:
            raise RuntimeError(f"microWakeWord training failed (exit 1): see {log_file}")
        return m

    monkeypatch.setattr(mww, "train", train)
    return calls, now


def test_candidates_train_one_per_gpu_at_once(proj, fake_train):
    calls, now = fake_train
    proj.microwakeword.candidates = 4
    old = mww.model_path(proj)
    old.parent.mkdir(parents=True)
    old.write_bytes(b"last round")
    said = []
    log = proj.path("logs", "mww-train.log")
    info = candidates.train(proj, log, ["cuda:0", "cuda:1"], log=said.append)
    assert now["most"] == 2  # two at once, never more
    assert sorted(c["dir"] for c in calls) == [f"candidates/s{i}.part" for i in range(1, 5)]
    assert {c["device"] for c in calls} == {"cuda:0", "cuda:1"} and {c["share"] for c in calls} == {
        2
    }
    assert len({c["seed"] for c in calls}) == 4
    assert info["slots"] == 2 and info["candidates"] == 4
    pick = json.loads(proj.path("mww", "candidates", "pick.json").read_text())
    assert [c["n"] for c in pick["candidates"]] == [1, 2, 3, 4]
    assert all(c["validation"] is not None for c in pick["candidates"])
    # the kept candidate is the model; the last round's model is kept aside, not a candidate
    kept = proj.path("mww", "candidates", f"s{info['picked']}")
    assert (
        mww.model_path(proj).read_bytes() == f"candidates/s{info['picked']}.part".encode()
    )  # its bytes name where it trained
    assert kept.is_dir()
    prev = proj.path("mww", "trained.previous") / old.relative_to(proj.path("mww", "trained"))
    assert prev.read_bytes() == b"last round"
    # every candidate's log ends up in the main log; the per-candidate logs and progress file are gone
    text = log.read_text()
    assert all(f"===== candidate s{i}" in text for i in range(1, 5))
    assert not list(proj.path("logs").glob("mww-train-s*.log"))
    assert not candidates.progress_path(proj).exists()
    assert any("on cuda:" in s for s in said)


def test_one_device_trains_candidates_one_after_another(proj, fake_train):
    calls, now = fake_train
    proj.microwakeword.candidates = 3
    info = candidates.train(proj, proj.path("logs", "mww-train.log"), "cpu", log=lambda m: None)
    assert now["most"] == 1 and info["slots"] == 1
    assert {c["device"] for c in calls} == {"cpu"} and {c["share"] for c in calls} == {1}


def test_more_gpus_than_candidates_uses_only_what_it_needs(proj, fake_train):
    calls, _ = fake_train
    proj.microwakeword.candidates = 2
    info = candidates.train(
        proj, proj.path("logs", "mww-train.log"), ["xpu:0", "xpu:1", "xpu:2"], log=lambda m: None
    )
    assert info["slots"] == 2 and {c["share"] for c in calls} == {2}
    assert {c["device"] for c in calls} <= {"xpu:0", "xpu:1"}


def test_a_failed_candidate_is_left_out_and_its_folder_removed(proj, fake_train, monkeypatch):
    proj.microwakeword.candidates = 3
    monkeypatch.setenv("FAKE_FAIL", "s3")
    said = []
    info = candidates.train(
        proj, proj.path("logs", "mww-train.log"), ["cuda:0", "cuda:1"], log=said.append
    )
    pick = json.loads(proj.path("mww", "candidates", "pick.json").read_text())
    failed = [c for c in pick["candidates"] if c.get("failed")]
    assert [c["n"] for c in failed] == [3] and info["picked"] != 3
    assert "mww-train-s3" not in failed[0]["failed"]  # points at the main log, which has its output
    assert not proj.path("mww", "candidates", "s3").exists()
    assert any("candidate 3 failed" in s for s in said)


def test_the_tf_trainer_stays_on_one_device(proj, fake_train, monkeypatch):
    calls, now = fake_train
    monkeypatch.setenv("WAKESMITH_MWW_TRAINER", "tf")
    proj.microwakeword.candidates = 2
    candidates.train(
        proj, proj.path("logs", "mww-train.log"), ["cpu", "cuda:1"], log=lambda m: None
    )
    assert now["most"] == 1 and {c["device"] for c in calls} == {"cpu"}


def test_progress_counts_finished_and_running_candidates(proj):
    proj.microwakeword.steps, proj.microwakeword.finetune_steps = 1000, 0
    logs = proj.path("logs")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "mww-train-s3.log").write_text(
        "x\rValidation Batch #2: loss Mini-Batch #100\r"
    )  # 600 steps
    (logs / "mww-train-s4.log").write_text("")  # just started
    candidates.progress_path(proj).write_text(
        json.dumps(
            {
                "n": 4,
                "slots": 2,
                "done": [1, 2],
                "running": {"3": "mww-train-s3.log", "4": "mww-train-s4.log"},
            }
        )
    )
    got = jobs._candidate_progress(proj)
    assert got == {"done": 2600, "total": 4000, "running": 2}
    candidates.progress_path(proj).write_text("{broken")
    assert jobs._candidate_progress(proj) is None
    # a damaged file can't point the reader at the logs folder, a pipe or another file
    import os

    os.mkfifo(logs / "mww-train-s9.log")
    for bad in ("", ".", "..", "mww-train-s9.log", "train.log", "../project.yaml"):
        candidates.progress_path(proj).write_text(
            json.dumps({"n": 2, "done": [1], "running": {"2": bad}})
        )
        assert jobs._candidate_progress(proj) == {"done": 1000, "total": 2000, "running": 1}


def test_time_estimate_counts_rounds_of_candidates(proj, monkeypatch, tmp_path):
    monkeypatch.setattr(eta, "_path", lambda: tmp_path / "timings.json")
    proj.microwakeword.steps, proj.microwakeword.finetune_steps = 1000, 0
    proj.microwakeword.candidates = 8
    assert eta.units(proj, "mww-train") == 8000  # nothing recorded: one at a time
    eta.record(proj, "mww-train", 400.0, {"candidates": 8, "picked": 1, "slots": 2})
    t = json.loads((tmp_path / "timings.json").read_text())
    assert t["mww-train:slots"] == 2 and t["mww-train"] == pytest.approx(400 / 4000)
    assert eta.units(proj, "mww-train") == 4000  # 4 rounds of 2
    proj.microwakeword.candidates = 3
    assert eta.units(proj, "mww-train") == 2000  # 2 then 1


def test_train_devices_lists_every_big_gpu():
    gpu = lambda d, gb: type("G", (), {"device": d, "memory_gb": gb})()
    assert mww.train_devices([gpu("cuda:0", 16), gpu("cuda:1", 16)]) == ["cuda:0", "cuda:1"]
    assert mww.train_devices([gpu("xpu:0", 4), gpu("xpu:1", 12)]) == ["xpu:1"]
    assert mww.train_devices([]) == ["cpu"]
    assert mww.train_device([gpu("cuda:0", 16), gpu("cuda:1", 16)]) == "cuda:0"


def test_write_config_names_its_file_and_folder(proj, monkeypatch):
    import yaml

    from wakesmith import assets

    monkeypatch.setattr(assets, "fetch", lambda a: None)
    proj.path("mww").mkdir(parents=True)
    assert mww.write_config(proj) == proj.path("mww", "training.yaml")
    assert yaml.safe_load(proj.path("mww", "training.yaml").read_text())["train_dir"] == "trained"
    got = mww.write_config(proj, "candidates/s2", "training-s2.yaml")
    assert got == proj.path("mww", "training-s2.yaml")
    assert yaml.safe_load(got.read_text())["train_dir"] == "candidates/s2"
    assert sorted(f.name for f in proj.path("mww").iterdir()) == [
        "training-s2.yaml",
        "training.yaml",
    ]


def test_write_config_from_many_threads_leaves_one_whole_file(proj, monkeypatch):
    import yaml

    from wakesmith import assets

    monkeypatch.setattr(assets, "fetch", lambda a: None)
    proj.path("mww").mkdir(parents=True)
    ts = [threading.Thread(target=mww.write_config, args=(proj,)) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert yaml.safe_load(proj.path("mww", "training.yaml").read_text())["train_dir"] == "trained"
    assert [f.name for f in proj.path("mww").iterdir()] == [
        "training.yaml"
    ]  # no temporary files left


def test_an_error_while_scoring_stops_the_candidates_still_waiting(proj, fake_train):
    calls, _ = fake_train
    proj.microwakeword.candidates = 6

    def log(msg):
        if ": validation" in msg:  # the main thread fails on the first finished candidate
            raise RuntimeError("scoring broke")

    with pytest.raises(RuntimeError, match="scoring broke"):
        candidates.train(proj, proj.path("logs", "mww-train.log"), "cpu", log=log)
    time.sleep(0.5)  # anything still queued would have started by now
    assert len(calls) <= 2  # the one that finished, at most the one already started
    assert not candidates.progress_path(proj).exists()


def test_train_devices_leaves_out_gpus_there_is_no_ram_for():
    gpu = lambda d: type("G", (), {"device": d, "memory_gb": 24})()
    four = [gpu(f"cuda:{i}") for i in range(4)]
    assert mww.train_devices(four, ram_free_gb=40) == [f"cuda:{i}" for i in range(4)]
    assert mww.train_devices(four, ram_free_gb=17) == ["cuda:0", "cuda:1"]
    assert mww.train_devices(four, ram_free_gb=3) == ["cuda:0"]  # always at least one
    assert mww.train_devices(four) == [f"cuda:{i}" for i in range(4)]  # unknown: no cap


def test_a_candidate_failing_some_other_way_stops_the_ones_still_waiting(
    proj, fake_train, monkeypatch
):
    calls, _ = fake_train
    monkeypatch.setenv("FAKE_OSERROR", "s1")
    proj.microwakeword.candidates = 6
    with pytest.raises(OSError, match="disk full"):
        candidates.train(
            proj, proj.path("logs", "mww-train.log"), ["cuda:0", "cuda:1"], log=lambda m: None
        )
    time.sleep(0.5)
    assert len(calls) == 2  # s1, and s2 running beside it: nothing starts after s1 fails
    assert not list(proj.path("logs").glob("mww-candidates.json*"))


def test_a_failing_log_line_gives_the_gpu_back_instead_of_hanging(proj, fake_train):
    proj.microwakeword.candidates = 3
    said = []

    def log(msg):  # e.g. a full disk: train.log can't take the line
        if " of 3 " in msg and len(said) < 2:
            said.append(msg)
            raise OSError("no space left")

    box = {}

    def run():
        try:
            candidates.train(
                proj, proj.path("logs", "mww-train.log"), ["cuda:0", "cuda:1"], log=log
            )
        except OSError as e:
            box["err"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(10)
    assert not t.is_alive(), "candidates.train hung waiting for a GPU that never came back"
    assert "no space left" in str(box.get("err"))


def test_a_failed_config_write_leaves_no_temporary_file(proj, monkeypatch):
    from pathlib import Path

    from wakesmith import assets

    monkeypatch.setattr(assets, "fetch", lambda a: None)
    proj.path("mww").mkdir(parents=True)
    real = Path.replace

    def full(self, target):
        raise OSError("no space left")

    monkeypatch.setattr(Path, "replace", full)
    with pytest.raises(OSError):
        mww.write_config(proj)
    monkeypatch.setattr(Path, "replace", real)
    assert list(proj.path("mww").iterdir()) == []


def test_leftovers_of_a_killed_run_are_cleared_at_the_start(proj, fake_train):
    proj.microwakeword.candidates = 2
    logs = proj.path("logs")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "mww-candidates.json").write_text("{}")
    (logs / "mww-candidates.json.part").write_text("{")
    candidates.train(proj, logs / "mww-train.log", "cpu", log=lambda m: None)
    assert not list(logs.glob("mww-candidates.json*"))


def test_a_set_stop_ends_a_running_step():
    import subprocess
    import sys

    stop = threading.Event()
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    threading.Timer(0.3, stop.set).start()
    t0 = time.time()
    rc = mww._wait(proc, stop)
    assert rc != 0 and time.time() - t0 < 5 and proc.poll() is not None
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    assert mww._wait(done, threading.Event()) == 0 and mww._wait(done, None) == 0


def test_running_candidates_are_told_to_stop_when_the_run_fails(proj, monkeypatch):
    seen = []

    def train(p, log_file, device, seed=None, train_dir="trained", share=1, stop=None):
        seen.append(stop)
        if train_dir.endswith("s1.part"):
            raise OSError("disk full")
        assert stop.wait(5), "the other candidate was never told to stop"
        raise RuntimeError("microWakeWord training failed (exit -15)")

    monkeypatch.setattr(mww, "train", train)
    proj.microwakeword.candidates = 2
    with pytest.raises(OSError):
        candidates.train(
            proj, proj.path("logs", "mww-train.log"), ["cuda:0", "cuda:1"], log=lambda m: None
        )
    time.sleep(0.3)
    assert all(s is not None and s.is_set() for s in seen)


def test_config_files_left_by_a_killed_run_are_removed(proj, fake_train):
    proj.microwakeword.candidates = 2
    proj.path("mww").mkdir(parents=True)
    for name in ("training-s7.yaml", ".training.yaml.123.456.tmp"):
        proj.path("mww", name).write_text("x")
    candidates.train(proj, proj.path("logs", "mww-train.log"), "cpu", log=lambda m: None)
    assert not list(proj.path("mww").glob("*training*"))  # the fake trainer writes no config


def test_a_failed_candidate_leaves_no_log_of_its_own(proj, fake_train, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "s2")
    proj.microwakeword.candidates = 3
    candidates.train(
        proj, proj.path("logs", "mww-train.log"), ["cuda:0", "cuda:1"], log=lambda m: None
    )
    assert not list(proj.path("logs").glob("mww-train-s*.log"))
    assert "===== candidate s2" in proj.path("logs", "mww-train.log").read_text()


def test_when_every_candidate_fails_the_last_model_stays_the_model(proj, monkeypatch):
    def train(p, log_file, device, seed=None, train_dir="trained", share=1, stop=None):
        raise RuntimeError("microWakeWord training failed (exit 1)")

    monkeypatch.setattr(mww, "train", train)
    old = mww.model_path(proj)
    old.parent.mkdir(parents=True)
    old.write_bytes(b"last round")
    proj.microwakeword.candidates = 2
    with pytest.raises(RuntimeError, match="every microWakeWord candidate failed"):
        candidates.train(proj, proj.path("logs", "mww-train.log"), "cpu", log=lambda m: None)
    assert mww.model_path(proj).read_bytes() == b"last round"


def test_a_stopped_round_keeps_only_whole_candidates(proj, monkeypatch):
    """combine uses the s<N> folders of a round stopped before the pick: none may be half trained."""
    from wakesmith import ensemble

    def train(p, log_file, device, seed=None, train_dir="trained", share=1, stop=None):
        d = p.path("mww", train_dir)
        (d / "tflite_stream_state_internal_quant").mkdir(parents=True)
        (d / "best_weights.weights.h5").write_bytes(
            b"w"
        )  # microWakeWord's own trainer saves as it goes
        if train_dir.endswith("s1.part"):
            (
                d / "tflite_stream_state_internal_quant" / "stream_state_internal_quant.tflite"
            ).write_bytes(b"m")
            return
        if train_dir.endswith("s2.part"):
            time.sleep(0.2)
            raise OSError("disk full")  # ends the run
        assert stop.wait(5)
        raise RuntimeError("microWakeWord training failed (exit -15)")

    monkeypatch.setattr(mww, "train", train)
    proj.microwakeword.candidates = 3
    old = mww.model_path(proj)
    old.parent.mkdir(parents=True)
    old.write_bytes(b"last round")
    with pytest.raises(OSError):
        candidates.train(
            proj,
            proj.path("logs", "mww-train.log"),
            ["cuda:0", "cuda:1", "cuda:2"],
            log=lambda m: None,
        )
    cands = proj.path("mww", "candidates")
    assert sorted(d.name for d in cands.iterdir()) == ["s1"]  # no s<N>.part left, s1 finished
    assert [c["n"] for c in ensemble.available(proj)] == [1]
    assert mww.model_path(proj).read_bytes() == b"last round"  # the model from before is the model
    assert not list(proj.path("logs").glob("mww-train-s*.log"))


def test_ctrl_c_during_the_grace_still_ends_the_step(monkeypatch):
    import subprocess
    import sys

    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
        ]
    )
    stop = threading.Event()
    stop.set()
    real = subprocess.Popen.wait
    state = {"n": 0}

    def wait(self, timeout=None):
        state["n"] += 1
        if state["n"] == 2:  # the 10 s grace after terminate
            raise KeyboardInterrupt
        return real(self, timeout=timeout)

    monkeypatch.setattr(subprocess.Popen, "wait", wait)
    with pytest.raises(KeyboardInterrupt):
        mww._wait(proc, stop)
    monkeypatch.setattr(subprocess.Popen, "wait", real)
    assert proc.poll() is not None  # killed, not left running


def test_a_round_stopped_before_any_candidate_finished_keeps_the_previous_round(proj, fake_train):
    proj.microwakeword.candidates = 2
    prev = proj.path("mww", "candidates.previous", "s1")
    prev.mkdir(parents=True)
    (prev / "best_weights.weights.h5").write_bytes(b"good round")
    (proj.path("mww", "candidates", "s3.part")).mkdir(
        parents=True
    )  # a stopped round: nothing finished
    candidates.keep_previous(proj, log=lambda m: None)
    assert (prev / "best_weights.weights.h5").read_bytes() == b"good round"
    assert not proj.path("mww", "candidates").exists()


def test_a_model_a_killed_run_moved_aside_is_the_model_again(proj, monkeypatch):
    proj.microwakeword.candidates = 1
    prev = proj.path("mww", "trained.previous") / mww.model_path(proj).relative_to(
        proj.path("mww", "trained")
    )
    prev.parent.mkdir(parents=True)
    prev.write_bytes(b"last round")
    seen = {}
    monkeypatch.setattr(
        mww, "train", lambda p, *a, **k: seen.setdefault("model", mww.model_path(p).read_bytes())
    )
    candidates.train(proj, proj.path("logs", "mww-train.log"), "cpu", log=lambda m: None)
    assert seen["model"] == b"last round"  # back in mww/trained before training started


def test_archive_keeps_trained_previous_when_it_is_the_only_model(proj):
    from wakesmith import archive

    rel = mww.model_path(proj).relative_to(proj.path("mww", "trained"))
    prev = proj.path("mww", "trained.previous") / rel
    prev.parent.mkdir(parents=True)
    prev.write_bytes(b"the only model")  # a Stop left it here, no mww/trained
    got = archive.plan(proj)
    assert "mww/trained.previous" not in {r["path"] for r in got["remove"]}
    assert any("trained.previous: kept" in k for k in got["kept"])
    m = mww.model_path(proj)
    m.parent.mkdir(parents=True)
    m.write_bytes(b"new")  # with a model in mww/trained it is a spare again
    assert "mww/trained.previous" in {r["path"] for r in archive.plan(proj)["remove"]}


def test_a_half_copied_model_gives_way_to_the_previous_one(proj, monkeypatch):
    rel = mww.model_path(proj).relative_to(proj.path("mww", "trained"))
    prev = proj.path("mww", "trained.previous") / rel
    prev.parent.mkdir(parents=True)
    prev.write_bytes(b"last round")
    (proj.path("mww", "trained") / "best_weights.weights.h5").parent.mkdir(parents=True)
    (proj.path("mww", "trained") / "best_weights.weights.h5").write_bytes(b"half")  # no model file
    proj.path("mww", "trained.part").mkdir()
    seen = {}
    monkeypatch.setattr(
        mww, "train", lambda p, *a, **k: seen.setdefault("model", mww.model_path(p).read_bytes())
    )
    proj.microwakeword.candidates = 1
    candidates.train(proj, proj.path("logs", "mww-train.log"), "cpu", log=lambda m: None)
    assert seen["model"] == b"last round" and not proj.path("mww", "trained.part").exists()


def test_a_half_written_model_never_replaces_the_previous_one(proj, fake_train):
    """A Stop during one-candidate training leaves weights but no model in mww/trained; with no
    previous model to restore, the next run must not keep that half model as the previous one."""
    t = proj.path("mww", "trained")
    t.mkdir(parents=True)
    (t / "best_weights.weights.h5").write_bytes(b"half")
    proj.microwakeword.candidates = 2
    candidates.train(proj, proj.path("logs", "mww-train.log"), "cpu", log=lambda m: None)
    assert not proj.path("mww", "trained.previous").exists()


def test_a_failed_final_copy_keeps_the_model_from_before(proj, fake_train, monkeypatch):
    old = mww.model_path(proj)
    old.parent.mkdir(parents=True)
    old.write_bytes(b"last round")
    proj.microwakeword.candidates = 2

    def full(src, dst, *a, **k):
        raise OSError("no space left")

    monkeypatch.setattr(candidates.shutil, "copytree", full)
    with pytest.raises(OSError):
        candidates.train(proj, proj.path("logs", "mww-train.log"), "cpu", log=lambda m: None)
    assert mww.model_path(proj).read_bytes() == b"last round"
    assert not proj.path("mww", "trained.part").exists()


def test_archive_lists_a_half_trained_candidate_folder_whole(proj):
    from wakesmith import archive

    d = proj.path("mww", "candidates", "s3.part")
    d.mkdir(parents=True)
    (d / "best_weights.weights.h5").write_bytes(b"w")
    got = {r["path"] for r in archive.plan(proj)["remove"]}
    assert "mww/candidates/s3.part" in got and not any(
        g.startswith("mww/candidates/s3.part/") for g in got
    )


def test_one_candidate_that_fails_puts_the_model_back(proj, monkeypatch):
    old = mww.model_path(proj)
    prev = proj.path("mww", "trained.previous") / old.relative_to(proj.path("mww", "trained"))
    prev.parent.mkdir(parents=True)
    prev.write_bytes(b"last round")

    def train(p, *a, **k):  # moved the model aside (as mww.train does), wrote half a run, failed
        import shutil

        shutil.rmtree(p.path("mww", "trained.previous"), ignore_errors=True)
        p.path("mww", "trained").rename(p.path("mww", "trained.previous"))
        (p.path("mww", "trained")).mkdir(parents=True)
        (p.path("mww", "trained", "best_weights.weights.h5")).write_bytes(b"half")
        raise RuntimeError("microWakeWord training failed (exit 1)")

    monkeypatch.setattr(mww, "train", train)
    proj.microwakeword.candidates = 1
    with pytest.raises(RuntimeError):
        candidates.train(proj, proj.path("logs", "mww-train.log"), "cpu", log=lambda m: None)
    assert mww.model_path(proj).read_bytes() == b"last round"


def test_a_round_stopped_before_its_pick_does_not_replace_a_finished_round(proj):
    root = proj.path("mww")
    prev = root / "candidates.previous"
    for i in (1, 2, 3):
        (prev / f"s{i}").mkdir(parents=True)
        (prev / f"s{i}" / "best_weights.weights.h5").write_bytes(b"good")
    (prev / "pick.json").write_text("{}")
    (root / "combined.previous").mkdir()
    (root / "candidates" / "s1").mkdir(parents=True)  # stopped: one finished, no pick
    candidates.keep_previous(proj, log=lambda m: None)
    assert sorted(d.name for d in prev.iterdir()) == ["pick.json", "s1", "s2", "s3"]
    assert (root / "combined.previous").exists() and not (root / "candidates").exists()
    # with no finished round to protect, a stopped round's finished candidates are kept as before
    import shutil

    shutil.rmtree(prev)
    (root / "candidates" / "s1").mkdir(parents=True)
    (root / "candidates" / "s2").mkdir(parents=True)
    candidates.keep_previous(proj, log=lambda m: None)
    assert sorted(d.name for d in prev.iterdir()) == ["s1", "s2"]


def test_a_stopped_or_failed_job_puts_the_model_back_at_once(proj):
    rel = mww.model_path(proj).relative_to(proj.path("mww", "trained"))
    prev = proj.path("mww", "trained.previous") / rel
    prev.parent.mkdir(parents=True)
    prev.write_bytes(b"last round")
    proj.path("job.json").write_text(json.dumps({"started": 1.0}))
    seen = {}
    real = candidates.restore_model

    def check(p):  # the job must still count as running while the model goes back
        seen["finished"] = json.loads(proj.path("job.json").read_text()).get("finished")
        real(p)

    import pytest as _pt

    mp = _pt.MonkeyPatch()
    mp.setattr(candidates, "restore_model", check)
    try:
        jobs.finished(proj.path("job.json"), -15)
    finally:
        mp.undo()
    assert mww.model_path(proj).read_bytes() == b"last round"
    assert seen == {"finished": None}  # restored before the job was noted as ended


def test_a_dropped_stopped_round_takes_its_combined_model_with_it(proj):
    root = proj.path("mww")
    (root / "candidates.previous" / "s1").mkdir(parents=True)
    (root / "candidates.previous" / "pick.json").write_text("{}")
    (root / "combined.previous").mkdir()
    (root / "candidates" / "s1").mkdir(parents=True)  # stopped round, combined by hand
    (root / "combined").mkdir()
    candidates.keep_previous(proj, log=lambda m: None)
    assert not (root / "combined").exists() and (root / "combined.previous").exists()


def test_archive_lists_a_killed_runs_training_leftovers(proj):
    from wakesmith import archive

    proj.path("mww", "trained.part").mkdir(parents=True)
    proj.path("mww", "training-s2.part.yaml").write_text("x")
    got = {r["path"] for r in archive.plan(proj)["remove"]}
    assert {"mww/trained.part", "mww/training-s2.part.yaml"} <= got


def test_a_job_that_died_with_its_launcher_gets_its_model_back(proj):
    rel = mww.model_path(proj).relative_to(proj.path("mww", "trained"))
    prev = proj.path("mww", "trained.previous") / rel
    prev.parent.mkdir(parents=True)
    prev.write_bytes(b"last round")
    proj.path("job.json").write_text(
        json.dumps({"launcher": jobs.LAUNCHER, "started": 1.0, "pid": 2**22 + 12345})
    )
    assert not jobs.status(proj)["running"]
    with jobs._locked():
        jobs._reconcile()
    assert mww.model_path(proj).read_bytes() == b"last round"
    assert json.loads(proj.path("job.json").read_text()).get("finished")


def test_a_combined_model_of_the_finished_round_is_kept_when_a_stopped_round_goes(proj):
    root = proj.path("mww")
    (root / "candidates.previous" / "s1").mkdir(parents=True)
    (root / "candidates.previous" / "pick.json").write_text("{}")
    (root / "candidates" / "s1").mkdir(parents=True)  # stopped round
    (root / "combined").mkdir()
    (root / "combined" / "combined.json").write_text(
        json.dumps({"previous": True})
    )  # combine --previous
    candidates.keep_previous(proj, log=lambda m: None)
    assert (
        json.loads((root / "combined.previous" / "combined.json").read_text())["previous"] is True
    )


def test_a_combine_previous_model_leaves_with_its_round(proj):
    """A finished round replaces candidates.previous: a combined model built from that old round
    (combine --previous) goes with it, instead of sitting next to the new previous round."""
    root = proj.path("mww")
    (root / "candidates.previous" / "s1").mkdir(parents=True)
    (root / "candidates.previous" / "pick.json").write_text("{}")
    (root / "combined.previous").mkdir()
    (root / "candidates" / "s1").mkdir(parents=True)
    (root / "candidates" / "pick.json").write_text("{}")  # this round finished
    (root / "combined").mkdir()
    (root / "combined" / "combined.json").write_text(json.dumps({"previous": True}))
    candidates.keep_previous(proj, log=lambda m: None)
    assert not (root / "combined").exists() and not (root / "combined.previous").exists()
    # one built from the round that finished goes with it, as before
    import shutil

    for d in ("candidates.previous",):
        shutil.rmtree(root / d)
    (root / "candidates" / "s1").mkdir(parents=True)
    (root / "candidates" / "pick.json").write_text("{}")
    (root / "combined").mkdir()
    (root / "combined" / "combined.json").write_text(json.dumps({"previous": False}))
    candidates.keep_previous(proj, log=lambda m: None)
    assert (root / "combined.previous").exists() and (root / "candidates.previous" / "s1").exists()


def test_archive_lists_a_killed_runs_temporary_config_but_not_a_running_ones(proj, monkeypatch):
    from wakesmith import archive

    proj.path("mww").mkdir(parents=True)
    proj.path("mww", ".training.yaml.1.2.tmp").write_text("x")
    assert "mww/.training.yaml.1.2.tmp" in {r["path"] for r in archive.plan(proj)["remove"]}
    monkeypatch.setattr(jobs, "busy", lambda p: True)
    assert "mww/.training.yaml.1.2.tmp" not in {r["path"] for r in archive.plan(proj)["remove"]}


def test_a_page_job_ending_leaves_a_command_line_run_alone(proj, monkeypatch):
    rel = mww.model_path(proj).relative_to(proj.path("mww", "trained"))
    prev = proj.path("mww", "trained.previous") / rel
    prev.parent.mkdir(parents=True)
    prev.write_bytes(b"last round")
    proj.path("mww", "trained").mkdir()  # the command-line run is training into it
    (proj.path("mww", "trained") / "best_weights.weights.h5").write_bytes(b"in progress")
    monkeypatch.setattr(jobs, "outside_run", lambda p: True)
    proj.path("job.json").write_text(json.dumps({"started": 1.0}))
    jobs.finished(proj.path("job.json"), -9)
    assert (proj.path("mww", "trained") / "best_weights.weights.h5").read_bytes() == b"in progress"
    assert prev.read_bytes() == b"last round"


def test_archive_leaves_a_running_rounds_part_folders_alone(proj, monkeypatch):
    from wakesmith import archive

    proj.path("mww", "candidates", "s2.part").mkdir(parents=True)
    monkeypatch.setattr(jobs, "busy", lambda p: True)
    assert "mww/candidates/s2.part" not in {r["path"] for r in archive.plan(proj)["remove"]}
