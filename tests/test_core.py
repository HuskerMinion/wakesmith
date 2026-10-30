"""Fast tests: no training, no downloads beyond the small lexicon (skipped when espeak-ng is missing)."""

from __future__ import annotations

import json
import pickle
import shutil

import numpy as np
import pytest
import soundfile as sf

from wakesmith import export, jobs, project, real
from wakesmith.real import RealData, trim_to_phrase, windows

needs_espeak = pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="espeak-ng missing")


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path / "projects"))
    return project.create("hey jarvis", negatives=["jarvis", "hey harvest"])


# --- phonemes ------------------------------------------------------------------------------------


@needs_espeak
def test_phoneme_ids_layout():
    from wakesmith.tts.phonemize import phoneme_ids, phonemes

    id_map = {
        "^": [1],
        "_": [0],
        "$": [2],
        **{p: [i + 3] for i, p in enumerate(phonemes("hey, jarvis"))},
    }
    ids = phoneme_ids("hey, jarvis", id_map)
    assert ids[:2] == [1, 0] and ids[-1] == 2
    assert "," in phonemes("hey, jarvis") and " " in phonemes("hey, jarvis")  # clause kept


# --- sound-alikes --------------------------------------------------------------------------------


@needs_espeak
def test_suggestions_never_sound_like_the_phrase():
    from wakesmith import phrases

    out = phrases.suggest("hey jarvis", 40)
    assert "jarvis" in out  # the bare name: the classic false trigger
    target = phrases._ipa_key("hey jarvis")
    assert all(phrases._ipa_key(s) != target for s in out)
    assert "hey jarvis" not in out


def test_neighbors_exclude_superstrings():
    from wakesmith import phrases

    try:
        near = phrases.neighbors("computer", 50)
    except (OSError, RuntimeError):  # lexicon unavailable offline
        pytest.skip("cmudict not available")
    assert "computers" not in near and "commuter" in near


# --- real recordings -----------------------------------------------------------------------------


def _tone(seconds: float, start: float, length: float, sr: int = 16000) -> np.ndarray:
    a = np.random.default_rng(0).normal(0, 0.002, int(seconds * sr)).astype(np.float32)
    s, e = int(start * sr), int((start + length) * sr)
    a[s:e] += 0.3 * np.sin(np.arange(e - s) * 2 * np.pi * 220 / sr).astype(np.float32)
    return a


def test_trim_ends_at_phrase_and_ignores_chime():
    a = _tone(4.0, 1.5, 0.8)  # "phrase" 1.5-2.3 s
    a[-int(0.2 * 16000) :] += 0.3  # chime in the last 0.2 s
    t = trim_to_phrase(a, ignore_tail_s=0.3)
    assert t is not None and len(t) == int(1.9 * 16000)
    assert np.abs(t[-int(0.05 * 16000) :]).max() < 0.35  # ends just after the phrase, not the chime


def test_trim_short_burst_at_clip_start_returns():
    # speech from the first frame, shorter than a burst: used to loop forever in _last_burst
    a = _tone(0.7, 0.0, 0.1)
    trim_to_phrase(a, ignore_tail_s=0.3)  # returns at all (None is fine: too short to be a phrase)
    assert trim_to_phrase(_tone(0.7, 0.0, 0.6), ignore_tail_s=0.3) is not None


def test_windows_cover_every_position():
    w = windows(np.zeros(4 * 16000, np.float32))
    assert len(w) == 5 and all(len(x) == 32000 for x in w)
    assert len(windows(np.zeros(8000, np.float32))) == 1  # short clip: left-padded


def test_heldout_and_corrections(proj, tmp_path):
    pos, neg = tmp_path / "pos", tmp_path / "neg"
    for d, names in ((pos, ["a.wav", "b.wav", "day2/c.wav"]), (neg, ["x.wav", "y.wav"])):
        for n in names:
            (d / n).parent.mkdir(parents=True, exist_ok=True)
            sf.write(d / n, _tone(2.5, 1.0, 0.8), 16000)
    (pos / "HELDOUT.txt").write_text("# test day\nday2/c.wav\n")
    proj.real.positives, proj.real.negatives = str(pos), str(neg)
    proj.save()
    r = RealData(proj)
    proj.path("review").mkdir()
    proj.path("review", "decisions.json").write_text(
        json.dumps(
            {
                "neg/y.wav": {"verdict": "wake"},  # mislabeled false trigger: really the phrase
                "pos/b.wav": {"verdict": "unsure"},  # dropped from training
                "pos/day2/c.wav": {"verdict": "not_wake"},  # held out: never moved
            }
        )
    )
    wake, notes = r.training_items("wake")
    assert sorted(wake) == [("neg", "y.wav"), ("pos", "a.wav")]
    assert any("held-out" in n for n in notes)
    stats = r.prepare()
    assert stats["real_positive_test"] == 1 and len(list(proj.path("real", "test").iterdir())) == 1
    assert (proj.path("real", "test") / real.flat_name("pos", "day2/c.wav")).exists()


def test_heldout_must_exist(proj, tmp_path):
    (tmp_path / "pos").mkdir()
    (tmp_path / "pos" / "HELDOUT.txt").write_text("missing.wav\n")
    proj.real.positives = str(tmp_path / "pos")
    with pytest.raises(ValueError, match="HELDOUT"):
        RealData(proj).listing("pos")


# --- project + export ----------------------------------------------------------------------------


def test_project_roundtrip(proj):
    proj.microwakeword.cutoff = 0.95
    proj.real.copies = 3
    proj.save()
    q = project.load("hey_jarvis")
    assert q.microwakeword.cutoff == 0.95 and q.real.copies == 3 and q.negatives == proj.negatives
    with pytest.raises(FileExistsError):  # a name that's taken is refused
        project.create("hey jarvis", name=proj.name)
    again = project.create("hey jarvis")  # no name given: the next version of the word
    assert again.name == "hey_jarvis_v2" and again.version == 2


def test_projects_from_the_vswakeword_days_still_load(proj):
    """Projects made while vsWakeWord was an option keep its settings, stages, results and export
    folder; they load, list and compare with those left out."""
    from fastapi.testclient import TestClient

    from wakesmith.web.server import create_app

    yml = proj.path("project.yaml")
    yml.write_text(yml.read_text() + "vswakeword:\n  enabled: true\n  steps: 20000\n")
    done = {"status": "done", "finished": 1.0}
    proj.set_state(**{s: done for s in ("mww-train", "vww-features", "vww-train", "export")})
    table = [{"threshold": t, "caught": 1, "false": 0} for t in (0.5, 1.0)]
    result = {"table": table, "positives": 1, "negatives": 1, "pos": {"a.wav": 1}, "neg": {}}
    proj.path("results").mkdir()
    for kind in ("mww", "vww"):
        proj.path("results", f"{kind}.json").write_text(json.dumps(result))
    for kind, ext in (("microwakeword", "tflite"), ("vswakeword", "onnx")):
        proj.path("export", kind).mkdir(parents=True)
        proj.path("export", kind, f"hey_jarvis.{ext}").write_bytes(b"model")
    shutil.copytree(proj.path("export"), proj.path("runs", "20261022-120000", "export"))
    shutil.copytree(proj.path("results"), proj.path("runs", "20261022-120000", "results"))
    run = {"id": "20261022-120000", "time": 1.0, "title": "", "label": "", "settings": {}}
    proj.path("runs", "20261022-120000", "run.json").write_text(json.dumps(run))
    entry = {"sha256": "ab" * 32, "sha": "abababab", "kind": "vww", "status": "live", "created": 1}
    proj.path("models.json").write_text(json.dumps([entry]))

    p = project.load("hey_jarvis")
    assert not hasattr(p, "vswakeword")
    p.save()
    assert "vswakeword" not in proj.path("project.yaml").read_text()
    c = TestClient(create_app(), base_url="http://localhost")
    got = c.get(f"/api/projects/{p.name}").json()
    assert "vswakeword" not in got and not any(s.startswith("vww") for s in got["stages"])
    assert got["stages"]["mww-train"]["status"] == "done" and list(got["results"]) == ["mww"]
    assert got["exported"] == ["microwakeword/hey_jarvis.tflite"] and got["live"] == {}
    (card,) = c.get("/api/projects").json()
    assert card["done"] == ["mww-train", "export"] and list(card["brief"]) == ["mww"]
    assert list(card["models"]) == ["mww", "oww"]
    assert list(c.get(f"/api/projects/{p.name}/sets").json()) == ["mww"]
    (r,) = c.get(f"/api/projects/{p.name}/runs").json()
    assert list(r["results"]) == ["mww"] and r["exported"] == ["microwakeword"]
    assert c.get(f"/api/projects/{p.name}/models").json() == []
    assert c.get(f"/api/projects/{p.name}/zip/vswakeword").status_code == 404
    assert c.get(f"/api/projects/{p.name}/zip/microwakeword").status_code == 200


def _tiny_tflite() -> bytes:
    import tensorflow as tf

    m = tf.keras.Sequential([tf.keras.Input((4,)), tf.keras.layers.Dense(1)])
    return tf.lite.TFLiteConverter.from_keras_model(m).convert()


def test_export_manifest(proj, monkeypatch):
    from wakesmith import mww

    model = mww.model_path(proj)
    model.parent.mkdir(parents=True)
    model.write_bytes(_tiny_tflite())
    proj.microwakeword.cutoff = 0.93
    files = export.export(proj)
    exported = proj.path("export", "microwakeword", "hey_jarvis.tflite")
    assert export.model_frontend(model) is None and export.model_frontend(exported) == "tflm"
    export.tag_frontend(exported, exported, "zserge")  # re-tagging replaces the tag
    assert export.model_frontend(exported) == "zserge"
    manifest = json.loads(proj.path("export", "microwakeword", "hey_jarvis.json").read_text())
    assert manifest["type"] == "micro" and manifest["micro"]["probability_cutoff"] == 0.93
    assert manifest["wake_word"] == "Hey Jarvis"
    assert any(f.name == "LICENSE-NOTE.txt" for f in files)  # personal data profile


# --- TTS checkpoint loading ----------------------------------------------------------------------


def test_restricted_unpickler_refuses_foreign_code():
    import io

    from wakesmith.tts.generate import _Unpickler

    class Evil:
        def __reduce__(self):
            return (shutil.rmtree, ("/nonexistent",))

    with pytest.raises(pickle.UnpicklingError, match="refusing"):
        _Unpickler(io.BytesIO(pickle.dumps(Evil()))).load()


def test_device_frontend_matches_zserge_go():
    """Checksum of the Go engine's int8 features (zserge/microwakeword bfaf384) on this signal."""
    import hashlib

    from wakesmith import devicefront

    t = np.arange(32000) / 16000
    noise = np.random.default_rng(7).normal(0, 200, len(t))
    a = (3000 * np.sin(2 * np.pi * 440 * t) * (t > 0.8) + noise).astype(np.int16)
    f = devicefront.features(a)
    assert f.shape == (198, 40)
    assert hashlib.sha256(f.tobytes()).hexdigest() == (
        "ac03e26a32b186a4deb0f33a30c1a6906619adcaf9b70f3b22dbe2bb822d7043"
    )


def test_cpu_only_mode(monkeypatch):
    from wakesmith import hardware

    monkeypatch.setenv("WAKESMITH_CPU", "1")
    hw, problem = hardware.detect_safe()
    assert (hw.accel, hw.gpus, problem) == ("cpu", [], "")
    assert hw.tts_devices() == ["cpu"] and hw.tts_batch() == 16 and hw.workers() >= 1


def test_disk_warning(monkeypatch):
    from collections import namedtuple

    from wakesmith import hardware

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(hardware.shutil, "disk_usage", lambda p: usage(0, 0, 10 * 2**30))
    assert "10 GB free" in hardware.disk_warning("/")
    monkeypatch.setattr(hardware.shutil, "disk_usage", lambda p: usage(0, 0, 100 * 2**30))
    assert hardware.disk_warning("/") == ""


def test_training_progress_between_step_lines(proj):
    from wakesmith import jobs

    logs = proj.path("logs")
    logs.mkdir(parents=True)
    batch = "Validation Batch #{}: Accuracy = 0.9; Loss = 0.1; Mini-Batch #{}\r"
    (logs / "mww-train.log").write_text(
        "INFO:absl:Step #500: rate 0.001\n" + "".join(batch.format(2, i) for i in range(1, 140))
    )
    (logs / "oww-progress.json").write_text('{"phase": 2, "done": 300, "total": 1000}')
    pr = jobs.progress(proj)
    assert pr["mww-train"]["done"] == 500 + 139
    assert pr["oww-train"] == {"phase": 2, "done": 300, "total": 1000}


def test_eta_learns_this_machines_rate(proj, monkeypatch, tmp_path):
    from wakesmith import eta

    monkeypatch.setattr(eta, "_path", lambda: tmp_path / "timings.json")
    proj.microwakeword.steps = 1000
    assert eta.estimates(proj)["mww-train"] == 1000 * eta.DEFAULT_SECONDS_PER_UNIT["mww-train"]
    eta.record(proj, "mww-train", 100.0, {})  # 0.1 s/step measured
    assert eta.estimates(proj)["mww-train"] == 100.0
    eta.record(proj, "mww-train", 300.0, {})  # running average of the rates
    assert eta.estimates(proj)["mww-train"] == 200.0
    eta.record(proj, "generate", 5.0, {"skipped": "all clips present"})  # skipped: no rate
    assert "generate" not in (tmp_path / "timings.json").read_text()


def test_stage_cut_off_by_a_dead_job_shows_failed(proj):
    proj.path("state.json").write_text(
        json.dumps({"generate": {"status": "done"}, "mww-train": {"status": "running"}})
    )
    proj.path("job.json").write_text(
        json.dumps({"pid": 2**22 + 12345, "args": [], "title": "t", "started": 0})
    )
    st = jobs.stages(proj, ["generate", "mww-train", "test"])
    assert st["generate"]["status"] == "done" and st["test"] is None
    assert st["mww-train"]["status"] == "failed" and "memory" in st["mww-train"]["error"]


def _no_setsid(monkeypatch, systemd: str) -> None:
    """systemd "no-setsid": no user manager and no setsid command, as on macOS."""
    import shutil

    monkeypatch.setenv("WAKESMITH_JOB_SYSTEMD", "1" if systemd == "1" else "0")
    if systemd == "no-setsid":
        which = shutil.which
        monkeypatch.setattr(
            shutil, "which", lambda n, *a, **k: None if n == "setsid" else which(n, *a, **k)
        )


@pytest.mark.parametrize("systemd", ["0", "1", "no-setsid"])
def test_job_runs_detached_and_notes_its_pid(proj, monkeypatch, systemd):
    import time

    if systemd == "1" and not jobs._systemd_user():
        pytest.skip("no systemd user manager")
    _no_setsid(monkeypatch, systemd)
    job = jobs.start(proj, ["--help"], "help")
    assert job["pid"] and ("unit" in job) == (systemd == "1")
    for _ in range(100):
        if not jobs.status(proj)["running"]:
            break
        time.sleep(0.1)
    assert not jobs.status(proj)["running"]
    assert "Usage" in proj.path("logs", "job.log").read_text()


@pytest.mark.parametrize("systemd", ["0", "1"])
def test_queued_runs_start_one_after_another_without_the_server(proj, monkeypatch, systemd):
    import time

    if systemd == "1" and not jobs._systemd_user():
        pytest.skip("no systemd user manager")
    monkeypatch.setenv("WAKESMITH_JOB_SYSTEMD", systemd)
    other = project.create("ok husker", negatives=["husker"])

    def item(p, args):
        return {"id": p.name + args[0], "project": p.name, "label": p.display, "args": args,
                "title": " ".join(args), "queued_at": time.time()}  # fmt: skip

    with jobs._locked():  # queue two runs behind a running one: its launcher must start them
        jobs._start(proj, ["--help"], "help")
        jobs._write(
            jobs._file("queue.json"), [item(other, ["no-such-command"]), item(proj, ["--help"])]
        )
    for _ in range(300):
        if len(jobs.history()) == 3:
            break
        time.sleep(0.1)
    h = jobs.history()
    assert [(e["project"], e["result"]) for e in h] == [
        ("hey_jarvis", "ok"),
        ("ok_husker", "failed"),
        ("hey_jarvis", "ok"),
    ]
    assert h[1]["exit"] == 2 and "exit code 2" in h[1]["error"]
    assert (
        h[1]["started"] >= h[0]["finished"] and h[2]["started"] >= h[1]["finished"]
    )  # one at a time
    assert jobs.queue() == [] and not jobs.running_all()
    assert "No such command" in other.path("logs", "job.log").read_text()
    assert jobs.status(proj)["result"] == "ok" and not jobs.status(proj)["running"]


@pytest.mark.parametrize("systemd", ["0", "1", "no-setsid"])
def test_stopping_a_run_is_noted_and_the_queue_moves_on(proj, monkeypatch, systemd):
    import socket
    import time

    if systemd == "1" and not jobs._systemd_user():
        pytest.skip("no systemd user manager")
    _no_setsid(monkeypatch, systemd)
    other = project.create("ok husker", negatives=["husker"])
    with socket.socket() as s:  # a job that runs until stopped: the web UI on a free port
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    jobs.start(proj, ["serve", "--host", "127.0.0.1", "--port", str(port)], "serve")
    jobs.enqueue(other, ["--help"], "help")
    assert [i["project"] for i in jobs.queue()] == [other.name]  # waits behind the running job
    time.sleep(1)
    assert jobs.stop(proj)
    for _ in range(300):
        if len(jobs.history()) == 2:
            break
        time.sleep(0.1)
    assert [(e["project"], e["result"]) for e in jobs.history()] == [
        (proj.name, "stopped"),
        (other.name, "ok"),
    ]
    assert jobs.queue() == []


def test_memory_snapshot_is_none_without_proc(monkeypatch):
    import os
    from pathlib import Path

    from wakesmith import sysstat

    if Path("/proc/meminfo").exists():
        assert sysstat.snapshot({"running": False})["ram"]["total_gb"] > 0
    real = Path.exists
    monkeypatch.setattr(
        Path, "exists", lambda self: False if str(self) == "/proc/meminfo" else real(self)
    )
    assert (
        sysstat.snapshot({"running": True, "pid": os.getpid()}) is None
    )  # macOS: no bars, no crash


def test_macos_ram_and_worker_start(monkeypatch):
    import subprocess
    import sys

    from wakesmith import hardware

    vm_stat = (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free:                               100000.\n"
        "Pages active:                             900000.\n"
        "Pages inactive:                           300000.\n"
        "Pages speculative:                         50000.\n"
        "Pages purgeable:                           10000.\n"
        "Pages wired down:                         400000.\n"
    )
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=vm_stat)
    )
    monkeypatch.setattr(
        hardware.os, "sysconf", {"SC_PAGE_SIZE": 16384, "SC_PHYS_PAGES": 4194304}.__getitem__
    )
    assert hardware._mac_ram(False) == 64 * 2**30  # a 64 GB Mac
    assert hardware._mac_ram(True) == 450000 * 16384  # free + inactive + speculative
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("WAKESMITH_WORKER_START", raising=False)
    assert hardware.worker_start() == "spawn"
    monkeypatch.setattr(sys, "platform", "linux")
    assert hardware.worker_start() == "fork"
    monkeypatch.setenv("WAKESMITH_WORKER_START", "spawn")
    assert hardware.worker_start() == "spawn"


def test_more_open_files_falls_back_to_open_max(monkeypatch):
    import resource

    from wakesmith import mww

    tried = []

    def setrlimit(kind, limits):
        tried.append(limits[0])
        if limits[0] > 10240:
            # as macOS does with an unlimited hard limit
            raise ValueError("current limit exceeds maximum limit")

    monkeypatch.setattr(resource, "getrlimit", lambda kind: (256, resource.RLIM_INFINITY))
    monkeypatch.setattr(resource, "setrlimit", setrlimit)
    mww.more_open_files()
    assert tried == [65536, 10240]


def test_queue_api_add_reorder_remove(proj, monkeypatch):
    from fastapi.testclient import TestClient

    from wakesmith.web.server import create_app

    monkeypatch.setattr(jobs, "kick", list)  # nothing starts: only the queue file changes
    other = project.create("ok husker", negatives=["husker"])
    c = TestClient(create_app(), base_url="http://localhost")
    a = c.post(f"/api/projects/{proj.name}/train", json={"queue": True}).json()["queued"]
    b = c.post(f"/api/projects/{other.name}/train", json={"from": "real", "queue": True}).json()
    b = b["queued"]
    assert b["args"][-4:] == ["--from", "real", "--to", "export"] and b["label"] == other.display
    q = c.get("/api/queue").json()
    assert [i["id"] for i in q["queue"]] == [a["id"], b["id"]] and q["running"] == []
    assert q["max_jobs"] == 1
    r = c.post(f"/api/queue/{b['id']}/move", json={"to": 0})
    assert [i["id"] for i in r.json()["queue"]] == [b["id"], a["id"]]
    assert c.post(f"/api/queue/{b['id']}/move", json={}).status_code == 400
    assert c.delete(f"/api/queue/{a['id']}").status_code == 200
    assert c.delete(f"/api/queue/{a['id']}").status_code == 404
    assert c.delete(f"/api/projects/{other.name}").status_code == 200  # its queued run goes too
    assert c.get("/api/queue").json()["queue"] == []


def test_job_that_died_with_its_launcher_is_noted_once(proj, tmp_path):
    dead = 2**22 + 12345
    other = project.create("ok husker", negatives=["husker"])
    proj.path("job.json").write_text(
        json.dumps({"pid": dead, "title": "t", "started": 1, "project": proj.name, "launcher": 2})
    )
    other.path("job.json").write_text(  # from before the queue: never reported, never notified
        json.dumps({"pid": dead, "title": "old", "started": 1})
    )
    jobs.kick()
    jobs.kick()
    h = jobs.history()
    assert len(h) == 1 and h[0]["result"] == "failed" and "without reporting" in h[0]["error"]
    assert jobs.status(proj)["finished"] and not jobs.status(other).get("finished")
    proj.path("job.json").write_text(
        json.dumps({"pid": dead, "started": 1, "launcher": 2, "stop_requested": True})
    )
    jobs.kick()
    assert jobs.history()[-1]["result"] == "stopped" and jobs.history()[-1]["error"] == ""
    for _ in range(30):
        jobs._record({"result": "ok"})
    assert len(jobs.history()) == jobs.HISTORY_KEEP


def test_ram_warning_below_16_gb():
    from wakesmith import hardware

    assert "16 GB" in hardware.ram_warning(hardware.Hardware("cpu", [], 6, 9.6, 9.0))
    assert hardware.ram_warning(hardware.Hardware("cpu", [], 6, 15.6, 14.8)) == ""


def _wav(seconds: float, sr: int = 16000) -> bytes:
    import io

    a = (0.2 * np.sin(np.arange(int(seconds * sr)) * 2 * np.pi * 220 / sr)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, a, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_recorded_takes_feed_the_held_out_split(proj):
    from wakesmith import recordings

    a = recordings.add(proj, "pos", _wav(1.2), "desk, close")
    b = recordings.add(proj, "pos", _wav(1.0), "kitchen, 3 m", heldout=True)
    assert (a["file"], b["file"]) == ("take_001.wav", "take_002.wav")
    proj = project.load(proj.dir)  # add() switched the project to its recordings folder
    assert recordings.active(proj, "pos")
    r = RealData(proj)
    files, held, rows = r.listing("pos")
    assert files == ["take_001.wav", "take_002.wav"] and held == {"take_002.wav"}
    assert rows["take_001.wav"]["condition"] == "desk, close"
    recordings.set_heldout(proj, "pos", "take_001.wav", True)
    recordings.delete(proj, "pos", "take_002.wav")
    files, held, _ = RealData(proj).listing("pos")
    assert files == ["take_001.wav"] and held == {"take_001.wav"}
    with pytest.raises(ValueError, match="16000 Hz"):
        recordings.add(proj, "pos", _wav(1.0, 22050))
    with pytest.raises(ValueError):
        recordings.delete(proj, "pos", "../project.yaml")


def test_takes_recorded_here_add_to_a_chosen_folder(proj, tmp_path):
    from wakesmith import recordings

    (tmp_path / "mine").mkdir()
    (tmp_path / "mine" / "a.wav").write_bytes(_wav(1.0))
    proj.real.positives = str(tmp_path / "mine")
    proj.save()
    assert recordings.add(proj, "pos", _wav(1.0))["file"] == "take_001.wav"
    assert proj.real.positives == str(tmp_path / "mine")  # the folder stays
    real = RealData(proj)
    assert set(real.sources) == {"pos", "rpos"} and real.listing("rpos")[0] == ["take_001.wav"]
    recordings.use(proj, "pos")  # recordings folder as the chosen one: counted once, as "pos"
    assert set(RealData(proj).sources) == {"pos"}


def test_delete_project_keeps_outside_folders(proj, tmp_path):
    (tmp_path / "mine").mkdir()
    (tmp_path / "mine" / "a.wav").write_bytes(_wav(1.0))
    proj.real.positives = str(tmp_path / "mine")
    proj.save()
    (proj.dir / "clips").mkdir()
    (proj.dir / "clips" / "x.wav").write_bytes(b"0" * 1000)
    assert project.delete(proj) >= 1000
    assert not proj.dir.exists() and (tmp_path / "mine" / "a.wav").exists()
    with pytest.raises(ValueError):
        project.delete(proj)


def test_recorded_takes_keep_speaker_and_extra_columns(proj):
    from wakesmith import recordings

    recordings.add(proj, "pos", _wav(1.0), "close", speaker="voiceA")
    d = recordings.folder(proj, "pos")
    (d / "labels.csv").write_text(
        (d / "labels.csv").read_text().replace("recorded\n", "recorded,note\n", 1)
    )
    recordings.set_speaker(proj, "pos", "take_001.wav", "voiceB")
    recordings.add(proj, "pos", _wav(1.0), "far", speaker="voiceA")
    rows = {r["file"]: r for r in recordings.listing(proj, "pos")["takes"]}
    assert (
        rows["take_001.wav"]["speaker"] == "voiceB" and rows["take_002.wav"]["speaker"] == "voiceA"
    )
    assert "note" in (d / "labels.csv").read_text().splitlines()[0]


def test_runs_are_kept_labelled_and_listed(proj):
    from wakesmith import runs

    assert runs.snapshot(proj) is None  # nothing exported yet
    (proj.path("export", "microwakeword")).mkdir(parents=True)
    (proj.path("export", "microwakeword", "m.tflite")).write_bytes(b"x")
    proj.path("results").mkdir()
    table = {
        "table": [{"threshold": 0.5, "caught": 3, "false": 1}],
        "positives": 4,
        "negatives": 10,
    }
    proj.path("results", "mww.json").write_text(json.dumps(table))
    rid = runs.snapshot(proj)
    runs.set_label(proj, rid, "first try")
    (only,) = runs.listing(proj)
    assert only["label"] == "first try" and only["results"]["mww"]["positives"] == 4
    assert (
        only["exported"] == ["microwakeword"]
        and only["settings"]["microwakeword"]["frontend"] == "tflm"
    )
    with pytest.raises(ValueError):
        runs.path(proj, "../..")
    runs.delete(proj, rid)
    assert runs.listing(proj) == []


def test_results_break_down_by_test_set(proj, tmp_path):
    from fastapi.testclient import TestClient

    from wakesmith import evaluate
    from wakesmith.web.server import create_app

    pos, neg = tmp_path / "pos", tmp_path / "neg"
    for d, names in ((pos, ["house/a.wav", "house/b.wav", "takes/c.wav"]), (neg, ["x.wav"])):
        for n in names:
            (d / n).parent.mkdir(parents=True, exist_ok=True)
            sf.write(d / n, _tone(2.5, 1.0, 0.8), 16000)
    (neg / "alike" / "1").mkdir(parents=True)
    sf.write(neg / "alike" / "1" / "y__z.wav", _tone(2.5, 1.0, 0.8), 16000)
    (pos / "HELDOUT.txt").write_text("house/a.wav\nhouse/b.wav\ntakes/c.wav\n")
    (neg / "labels.csv").write_text("file,source\nx.wav,\nalike/1/y__z.wav,synthetic look-alike\n")
    (neg / "HELDOUT.txt").write_text("x.wav\nalike/1/y__z.wav\n")
    proj.real.positives, proj.real.negatives = str(pos), str(neg)
    proj.microwakeword.cutoff = 0.7
    proj.save()
    names = {"pos/house": ["house/a.wav", "house/b.wav"], "pos/takes": ["takes/c.wav"]}
    exact = evaluate._held_sets(proj)
    for sid, files in names.items():
        assert all(exact[real.flat_name("pos", f)] == sid for f in files)
    y = real.flat_name("neg", "alike/1/y__z.wav")
    assert exact[y] == evaluate.set_name(y) == "neg/alike" and exact["neg__x.wav"] == "neg"

    table = [{"threshold": t, "caught": 0, "false": 0} for t in (0.5, 0.9)]
    old = {  # written before "set_of": grouped from the clip names
        "table": table,
        "pos": {"pos__house__a.wav": 0.8, "pos__house__b.wav": 0.6, "pos__takes__c.wav": 0.95},
        "neg": {"neg__x.wav": 0.1, y: 0.75},
    }
    proj.path("results").mkdir()
    proj.path("results", "mww.json").write_text(json.dumps(old))
    proj.path("results", "mww-cv.json").write_text(json.dumps({"pos": {"t1.wav": 0.9}}))
    proj.path("results", "oww.json").write_text(json.dumps({"table": table}))  # no clips kept
    got = (
        TestClient(create_app(), base_url="http://localhost")
        .get(f"/api/projects/{proj.name}/sets")
        .json()
    )
    assert list(got) == ["mww"]
    m = got["mww"]
    i = m["cutoffs"].index(0.7)  # the chosen cutoff is always there
    assert m["cutoffs"] == [0.5, 0.7, 0.9]
    at = {s["id"]: (s["phrase"], s["hits"][i], s["clips"], s["note"]) for s in m["sets"]}
    assert at == {
        "pos/house": (True, 1, 2, ""),
        "pos/takes": (True, 1, 1, ""),
        "neg": (False, 0, 1, ""),
        "neg/alike": (False, 1, 1, "synthetic look-alike"),
        "cv": (True, 1, 1, ""),
    }


def _oww_model(proj, data: bytes) -> None:
    proj.microwakeword.enabled = False
    model = proj.path("oww", "model", "model.tflite")
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(data)


def test_export_records_model_history_once_per_file(proj):
    from wakesmith import models, runs

    _oww_model(proj, b"model one")
    proj.path("results").mkdir()
    table = [{"threshold": t, "caught": c, "false": f} for t, c, f in ((0.5, 9, 3), (0.7, 7, 1))]
    proj.path("results", "oww.json").write_text(
        json.dumps({"table": table, "positives": 10, "negatives": 50})
    )
    proj.openwakeword.threshold = 0.5
    export.export(proj)
    (e,) = models.load(proj)
    assert e["kind"] == "oww" and e["status"] == "exported" and e["sha"] == e["sha256"][:8]
    assert e["cutoff"] == 0.5 and e["heldout"] == {
        "caught": 9,
        "positives": 10,
        "false": 3,
        "negatives": 50,
    }
    rid = runs.snapshot(proj)
    assert rid
    models.attach_run(proj, rid)
    assert models.load(proj)[0]["run"] == rid
    proj.openwakeword.threshold = 0.7  # same file again at another cutoff: same entry
    export.export(proj)
    (e,) = models.load(proj)
    assert e["cutoff"] == 0.7 and e["heldout"]["false"] == 1 and "0.5" in e["history"][-1]["note"]
    _oww_model(proj, b"model two")  # trained after the results: no held-out numbers for it
    export.export(proj)
    new = models.load(proj)[-1]
    assert len(models.load(proj)) == 2 and new["heldout"] is None and new["run"] is None


def test_one_live_model_per_kind(proj):
    from wakesmith import models

    shas = []
    for data in (b"a", b"b"):
        _oww_model(proj, data)
        export.export(proj)
        shas.append(models.load(proj)[-1]["sha"])
    assert models.set_status(proj, shas[0], "live")["retired"] == []
    models.set_status(proj, shas[1], "offered")
    out = models.set_status(proj, shas[1], "live")
    assert out["retired"] == [shas[0]] and out["entry"]["status"] == "live"
    first, second = models.load(proj)
    assert first["status"] == "retired" and shas[1] in first["history"][-1]["note"]
    assert [h["status"] for h in second["history"]] == ["exported", "offered", "live"]
    assert models.live(proj) == {"oww": shas[1]}
    models.set_notes(proj, shas[1], "  on the kitchen device  ")
    assert models.load(proj)[1]["notes"] == "on the kitchen device"
    with pytest.raises(ValueError):
        models.set_status(proj, shas[1], "deployed")
    with pytest.raises(KeyError):
        models.set_status(proj, "0000", "live")  # shorter than the short sha


def test_backfill_finds_earlier_exports(proj):
    from wakesmith import models

    old = proj.path("runs", "20260101-120000")
    (old / "export" / "openwakeword").mkdir(parents=True)
    (old / "export" / "openwakeword" / "hey_jarvis.tflite").write_bytes(b"old model")
    (old / "export" / "openwakeword" / "hey_jarvis.json").write_text(json.dumps({"threshold": 0.7}))
    (old / "results").mkdir()
    table = [{"threshold": 0.7, "caught": 5, "false": 2}]
    (old / "results" / "oww.json").write_text(
        json.dumps({"table": table, "positives": 6, "negatives": 40})
    )
    (old / "run.json").write_text(json.dumps({"id": old.name, "time": 1767294000.0}))
    d = proj.path("export", "openwakeword")
    d.mkdir(parents=True)
    (d / "hey_jarvis.tflite").write_bytes(b"old model")  # the same model again: listed once
    (d / "hey_jarvis.json").write_text(json.dumps({"threshold": 0.7}))
    proj.path("export", "microwakeword").mkdir()
    proj.path("export", "microwakeword", "hey_jarvis.tflite").write_bytes(b"current")
    (added_old, added_cur) = models.backfill(proj)
    assert added_old["run"] == old.name and added_old["created"] == 1767294000.0
    assert added_old["cutoff"] == 0.7 and added_old["heldout"]["caught"] == 5
    assert added_cur["kind"] == "mww" and added_cur["run"] is None and added_cur["cutoff"] is None
    assert models.backfill(proj) == []


def _mms_cached() -> bool:
    from pathlib import Path as P

    return any((P.home() / ".cache/torch/hub/checkpoints").glob("model.pt"))


@pytest.mark.skipif(not _mms_cached(), reason="MMS aligner weights not downloaded")
def test_commonvoice_cuts_split_by_speaker(proj, tmp_path, monkeypatch):
    import io
    import tarfile

    from wakesmith import assets, commonvoice

    monkeypatch.setenv("WAKESMITH_HOME", str(tmp_path / "data"))
    ds = tmp_path / "cv"
    (ds / "transcript/en").mkdir(parents=True)
    (ds / "audio/en/train").mkdir(parents=True)
    head = "client_id\tpath\tsentence_id\tsentence\tsentence_domain\tup_votes\tdown_votes\tage\tgender\taccents\tvariant\tlocale\tsegment\n"
    rows = [
        ("spk1", "a.mp3", "Hey jarvis, are you there?"),
        ("spk2", "b.mp3", "Say hey jarvis now."),
        ("spk3", "c.mp3", "Nothing here."),
    ]
    (ds / "transcript/en/train.tsv").write_text(
        head + "".join(f"{c}\t{p}\tx\t{s}\t\t1\t0\t\t\t\t\ten\t\n" for c, p, s in rows)
    )
    for s in ("dev", "test"):
        (ds / "transcript/en" / f"{s}.tsv").write_text(head)
    with tarfile.open(ds / "audio/en/train/en_train_0.tar", "w") as tf:
        for _, name, _ in rows:
            data = _wav(2.0)
            ti = tarfile.TarInfo(f"en_train_0/{name}")
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    assert commonvoice.present(ds)
    assert commonvoice.count(ds, "hey jarvis") == {"takes": 2, "speakers": 2, "extracted": False}
    home = commonvoice.extract(ds, "hey jarvis", log=lambda m: None)
    assert sorted(f.name for f in (home / "cuts").glob("*.wav")) == ["a.wav", "b.wav"]
    assert commonvoice.count(ds, "hey jarvis")["extracted"]
    proj.commonvoice.dataset, proj.commonvoice.train, proj.commonvoice.test_share = (
        str(ds),
        True,
        0.5,
    )
    monkeypatch.setattr(commonvoice, "MIN_ALIGN", -1.0)  # a tone aligns badly; keep it for the test
    info = commonvoice.prepare(proj, log=lambda m: None)
    assert info["cv_test"] + info["cv_train"] == 2
    assert len(list(proj.path("real", "train").glob("cv__*.wav"))) == info["cv_train"]
    assert assets.home() == tmp_path / "data"


def test_license_note_lists_what_the_model_used(proj):
    proj.openwakeword.enabled = False
    note = export.license_note(proj)
    assert (
        "microWakeWord model" in note
        and "openWakeWord model" not in note
        and "Common Voice" not in note
    )
    proj.commonvoice.enabled, proj.commonvoice.train = True, True
    note = export.license_note(proj)
    assert (
        "Common Voice 17.0 English, training and testing: CC0" in note and "Non-commercial" in note
    )


def test_wsl_nvidia_found_through_nvidia_smi(monkeypatch):
    from wakesmith import setup_env as se

    monkeypatch.setattr(
        se, "pci_gpus", lambda: [("other", "0000:00:08.0")]
    )  # WSL2: no NVIDIA on PCI
    monkeypatch.setattr(se, "nvidia_smi_gpus", lambda: [("nvidia", "00000000:01:00.0")])
    assert se.choose_accel(se.find_gpus()) == "cuda"
    monkeypatch.setattr(se, "nvidia_smi_gpus", list)
    assert se.choose_accel(se.find_gpus()) == "cpu"


def test_new_project_without_sound_alikes_gets_suggestions(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from wakesmith.web.server import create_app

    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path / "projects"))
    c = TestClient(create_app(), base_url="http://localhost")
    r = c.post("/api/projects", json={"phrase": "ok husker", "negatives": [], "preset": "quick"})
    assert r.status_code == 200, r.text
    assert len(r.json()["negatives"]) >= 2 and "husker" in r.json()["negatives"]


def test_help_page_guide_is_the_running_versions_readme():
    from fastapi.testclient import TestClient

    from wakesmith.web.server import HERE, create_app

    r = TestClient(create_app(), base_url="http://localhost").get("/guide")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    assert r.text.startswith("# wakesmith") and "PolyForm Noncommercial" in r.text
    assert "github.com/HuskerMinion/wakesmith/issues" in (HERE / "index.html").read_text()  # public


def test_gpu_build_choice_prefers_nvidia_then_intel_then_amd():
    from wakesmith import setup_env as se

    assert se.choose_accel([("amd", "a")]) == "rocm"
    assert se.choose_accel([("intel", "i"), ("amd", "a")]) == "xpu"  # e.g. Arc + a Ryzen's iGPU
    assert se.choose_accel([("nvidia", "n"), ("amd", "a")]) == "cuda"
    assert se.torch_sees("cpu")


def test_torch_install_without_uv_passes_pip_only_pip_flags(monkeypatch):
    from wakesmith import setup_env as se

    calls = []
    monkeypatch.setattr(se.subprocess, "run", lambda cmd, check: calls.append(cmd))
    monkeypatch.setattr(se.shutil, "which", lambda name: None)
    se.install_torch("cuda")
    assert [c[3] for c in calls] == ["uninstall", "install"]
    assert not any("--reinstall-package" in c for c in calls)
    monkeypatch.setattr(se.shutil, "which", lambda name: "/usr/bin/uv")
    calls.clear()
    se.install_torch("cuda")
    assert calls[0][:3] == ["/usr/bin/uv", "pip", "install"] and "--reinstall-package" in calls[0]


def test_tensorflow_kept_off_the_gpu_after_generation(monkeypatch):
    import os

    from wakesmith import pipeline

    monkeypatch.delenv("WAKESMITH_TF_GPU", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    pipeline._tf_off_gpu()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    monkeypatch.setenv("WAKESMITH_TF_GPU", "0")
    pipeline._tf_off_gpu()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_count_clips_drops_a_cut_off_clip_and_what_follows(tmp_path):
    from wakesmith.tts.generate import count_clips, write_wav

    for i in range(5):
        write_wav(tmp_path / f"clip_{i:06d}.wav", np.zeros(1600, np.float32))
    (tmp_path / "clip_000002.wav").write_bytes(b"\0" * 3244)  # zero-filled by a reset
    (tmp_path / "clip_000005.wav.part").write_bytes(b"x")
    assert count_clips(tmp_path) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["clip_000000.wav", "clip_000001.wav"]


def test_mww_torch_sampler_cuts_like_microwakeword(tmp_path):
    from mmap_ninja.ragged import RaggedMmap

    from wakesmith.mww_torch import UINT16_SCALE, FeatureSet

    rng = np.random.default_rng(0)
    clips = [rng.integers(1, 600, (n, 40)).astype(np.uint16) for n in (50, 204, 300)]
    (tmp_path / "set" / "training").mkdir(parents=True)
    RaggedMmap.from_lists(str(tmp_path / "set" / "training" / "shard00_mmap"), clips)
    for trunc in ("truncate_start", "truncate_end"):
        fs = FeatureSet(
            {
                "features_dir": str(tmp_path / "set"),
                "truth": True,
                "sampling_weight": 1.0,
                "penalty_weight": 1.0,
                "truncation_strategy": trunc,
            }
        )
        got = fs.sample(64, 204, np.random.default_rng(1))
        for want in (
            clips
        ):  # every sample is one of the clips, cut or zero-padded in front like microWakeWord does
            n = len(want)
            ref = np.zeros((204, 40), np.float32)
            src = (want[-204:] if trunc == "truncate_start" else want[:204]).astype(
                np.float32
            ) * UINT16_SCALE
            ref[204 - min(n, 204) :] = src
            assert any(np.array_equal(g, ref) for g in got), (trunc, n)
    fs = FeatureSet(
        {
            "features_dir": str(tmp_path / "set"),
            "truth": False,
            "sampling_weight": 1.0,
            "penalty_weight": 1.0,
            "truncation_strategy": "random",
        }
    )
    scaled = [c.astype(np.float32) * UINT16_SCALE for c in clips]
    short = np.zeros((204, 40), np.float32)
    short[-50:] = scaled[0]
    # randint(0, n - L) for the long clip: its last possible start is excluded
    windows = {scaled[2][o : o + 204].tobytes() for o in range(300 - 204)}
    allowed = windows | {short.tobytes(), scaled[1].tobytes()}
    got = [g.tobytes() for g in fs.sample(200, 204, np.random.default_rng(2))]
    assert all(g in allowed for g in got)
    assert len({g for g in got if g in windows}) > 10  # random offsets, not one fixed cut


def test_mww_torch_metrics_average_viable_recall():
    from wakesmith.mww_torch import metrics

    y = np.array([1, 1, 1, 1, 0, 0], np.float32)
    p = np.array([0.95, 0.8, 0.6, 0.2, 0.1, 0.7], np.float32)
    m = metrics(p, y, np.array([0.05, 0.3, 0.55]), hours=1.0)
    assert m["recall"] == 0.75 and m["precision"] == 0.75
    assert m["ambient_false_positives"] == 1  # one ambient window above 0.5
    assert m["cutoff_for_no_faph"] == 0.55 and m["recall_at_no_faph"] == 0.75
    assert 0 < m["average_viable_recall"] <= 1


def test_mww_torch_weights_copy_into_keras_exactly():
    import subprocess
    import sys

    code = (
        "import numpy as np, torch\n"
        "from wakesmith import mww_torch as t\n"
        "import tensorflow as tf; from microwakeword import mixednet\n"
        "torch.manual_seed(0); m = t.MixedNet(40, 204)\n"
        "for b in m.blocks: b['bn'].running_mean.data.uniform_(-0.5, 0.5); b['bn'].running_var.data.uniform_(0.5, 2)\n"
        "m.eval(); k = mixednet.model(t.model_flags(), (204, 40), 8); t.copy_into_keras(m, k)\n"
        "x = (np.random.default_rng(0).random((8, 204, 40)) * 25).astype(np.float32)\n"
        "d = np.abs(torch.sigmoid(m(torch.from_numpy(x))).detach().numpy() - k.predict(x, verbose=0)).max()\n"
        "print('DIFF', d)\n"
    )
    env = {**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "-1", "TF_CPP_MIN_LOG_LEVEL": "3"}
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("DIFF")]
    assert line, out.stderr[-2000:]
    assert float(line[0].split()[1]) < 1e-5


@pytest.mark.skipif(
    not all(
        __import__("wakesmith.assets").assets.ASSETS[k].present()
        for k in ("oww-melspectrogram", "oww-embedding")
    ),
    reason="openWakeWord frontend models not downloaded",
)
def test_oww_torch_frontend_matches_onnx():
    """The PyTorch frontend (oww-features on a GPU) gives ONNX Runtime's features within float32
    rounding: a tone over noise, silence (the -80 dB floor) and a clip shorter than 16 embeddings."""
    from wakesmith import oww, oww_torch

    rng = np.random.default_rng(3)
    t = np.arange(32000) / 16000
    tone = 8000 * np.sin(2 * np.pi * 300 * t * (1 + t)) * (t > 1.2) + rng.normal(0, 300, len(t))
    clips = [oww.pcm_scale(np.clip(tone / 32767, -1, 1)), np.zeros(32000, np.float32)]
    clips += [oww.pcm_scale(rng.normal(0, 0.05, 32000)), clips[0][:20000]]
    ref = oww.Frontend().features(clips)
    got = oww_torch.Frontend("cpu", batch=2).features(clips)
    assert got.shape == ref.shape == (4, 16, 96)
    # 6 embeddings, zero-padded in front
    assert not ref[3, :10].any() and ref[3, 10:].any(axis=-1).all()
    assert np.abs(got - ref).max() < 1e-3 * max(1.0, np.abs(ref).max())


def test_oww_features_reused_until_an_input_changes(proj):
    import os

    from wakesmith import oww

    def wav(path, n=1600):
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, np.zeros(n, np.float32), 16000)

    for split in ("positive_train", "negative_train", "positive_test", "negative_test"):
        for i in range(2):
            wav(proj.path("clips", split, f"clip_{i:06d}.wav"))
    wav(proj.path("real", "train", "take1.wav"))
    d = proj.path("oww", "features")
    d.mkdir(parents=True)
    stamp = oww._stamp(proj, oww._feature_sets(proj))
    assert not oww.features_current(proj, stamp)  # nothing built yet
    for f in oww.FEATURE_FILES:
        (d / f).write_bytes(b"")
    (d / "stamp.json").write_text(json.dumps(stamp))
    assert oww.features_current(proj, oww._stamp(proj, oww._feature_sets(proj)))
    assert stamp["sets"]["positive_train"]["files"] == 2 + proj.real.copies

    def current() -> bool:
        return oww.features_current(proj, oww._stamp(proj, oww._feature_sets(proj)))

    proj.real.copies += 1  # a changed setting
    assert not current()
    proj.real.copies -= 1
    assert current()
    wav(proj.path("real", "train", "take2.wav"))  # a new recording
    assert not current()
    proj.path("real", "train", "take2.wav").unlink()
    assert current()
    clip = proj.path("clips", "negative_test", "clip_000001.wav")
    os.utime(clip, ns=(0, clip.stat().st_mtime_ns + 10**9))  # a regenerated clip
    assert not current()


def test_oww_torch_head_trains_like_keras():
    """oww_torch.HeadTrainer (oww-train on a GPU) takes the same steps as the Keras model with Keras'
    AdamW from the same weights, and its weights score identically once copied into Keras."""
    from wakesmith import oww, oww_torch

    tf = oww.hardware.tensorflow_on_cpu()
    tf.keras.utils.set_random_seed(0)
    model = oww.build_model(32, 1)
    w0 = model.get_weights()
    keras = oww._KerasHead(model, 1e-2)
    head = oww_torch.HeadTrainer(w0, "cpu", weight_decay=1e-2)
    rng = np.random.default_rng(0)
    x = rng.normal(0, 10, (64, 16, 96)).astype(np.float16)
    y = (rng.random(64) < 0.3).astype(np.float32)
    for s in range(5):
        lam, perm = np.float32(rng.beta(0.2, 0.2)), rng.permutation(64)
        args = (lam, perm, 0.05, 2.0, 1.0 + 10 * s, 1e-3)
        lk, lt = float(keras.step(x, y, *args)), float(head.step(x, y, *args))
        assert abs(lk - lt) < 1e-4 * max(1.0, abs(lk))
    moved = max(np.abs(a - b).max() for a, b in zip(keras.get_weights(), w0))
    diff = max(np.abs(a - b).max() for a, b in zip(keras.get_weights(), head.get_weights()))
    assert moved > 1e-3 and diff < 1e-5  # rounding only: ~1e-6 after 5 steps
    model.set_weights(head.get_weights())
    xs = x.astype(np.float32)
    assert np.abs(model.predict(xs, verbose=0)[:, 0] - head.predict(xs)).max() < 1e-6


def test_xpu_selftest_names_the_broken_runtime():
    from wakesmith import hardware

    ok = {"device": "xpu:0", "driver": "1.14.39758", "ok": True}
    bad = {"device": "xpu:1", "driver": "1.14.37020", "ok": False}
    assert hardware.xpu_selftest_message([ok]) == ""
    msg = hardware.xpu_selftest_message([ok, bad])
    assert "xpu:1" in msg and "xpu:0" not in msg and "1.14.37020" in msg and "26.35" in msg


def test_tts_memory_guard_shrinks_batches_to_fit_the_card(monkeypatch):
    import types

    import torch

    from wakesmith.tts import generate as G

    gb = 2**30
    fake = types.SimpleNamespace(
        get_device_properties=lambda d: types.SimpleNamespace(total_memory=32 * gb),
        max_memory_reserved=lambda d: fake.peak,
        reset_peak_memory_stats=lambda d: None,
        empty_cache=lambda: None,
        peak=0,
    )
    monkeypatch.setattr(torch, "xpu", fake, raising=False)
    g = G._MemoryGuard(torch.device("xpu", 0), 400, slowest=1.3)
    assert g.next_size(first=True) == 100 and g.next_size(first=False) == 400
    fake.peak = 13.3 * gb  # measured on an Arc Pro B70: 100 long clips at speed 1.3
    g.measured(100, 1.3)
    assert 140 <= g.limit <= 180  # 70% of 32 GB
    fake.peak = 4 * gb  # a fast batch can't raise it, and scales to the slowest speed
    g.measured(100, 0.75)
    assert g.limit <= 180
    assert G._MemoryGuard(torch.device("cpu"), 16, 1.3).next_size(first=True) == 16


def test_best_of_n_pick_puts_false_triggers_first():
    from wakesmith.candidates import choose

    c = lambda n, caught, false, val=0.9: {
        "n": n, "seed": n, "validation": val, "cutoff": 0.5,
        "positives": 50, "negatives": 400, "caught": caught, "false": false,
    }  # fmt: skip
    # one round's seeds: s1's fewest false triggers win over s2's extra recall
    best, why = choose([c(1, 45, 3), c(2, 48, 22), c(3, 47, 18)])
    assert best["n"] == 1 and "s1" in why
    # near-ties on false triggers compete on positives caught
    assert choose([c(1, 40, 8), c(2, 46, 9), c(3, 47, 30)])[0]["n"] == 2
    # a failed candidate is skipped; without held-out data the validation score decides
    v = [{"n": 1, "failed": "x"}, {"n": 2, "validation": 0.91}, {"n": 3, "validation": 0.94}]
    assert choose(v)[0]["n"] == 3


def test_cut_short_keeps_part_of_the_speech():
    import numpy as np

    from wakesmith import mww

    a = np.zeros(32000, np.float32)
    a[8000:24000] = 0.5  # 1 s of "speech" between silences
    for seed in range(20):
        b = mww.cut_short(a, np.random.default_rng(seed))
        kept = len(b) - 8000
        assert 0.25 * 16000 - 1 <= kept <= 0.6 * 16000 + 1
    assert len(mww.cut_short(np.zeros(100, np.float32))) == 100  # silence: unchanged


def test_device_clip_round_import_label_and_time_split(proj, tmp_path):
    from wakesmith import rounds
    from wakesmith.real import RealData

    src = tmp_path / "incoming"
    clips = [  # (device, name, at, transcript)
        ("kitchen", "a.wav", "2026-10-22T10:00:00Z", "turn it up"),
        ("kitchen", "b.wav", "2026-10-22T11:00:00Z", "hey jarvis"),
        ("desk", "c.wav", "2026-10-23T09:00:00Z", "hey there"),
        ("desk", "d.wav", "2026-10-24T09:00:00Z", "so anyway"),
        ("desk", "e.wav", "2026-10-24T10:00:00Z", "Hey, Jarvis!"),
    ]
    for dev, name, at, _ in clips:
        d = src / dev / "help"
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(_wav(1.0))
        (d / name).with_suffix(".json").write_text(
            json.dumps({"at": at, "model": "m", "score": 0.6})
        )
    r = rounds.import_folder(proj, src)
    assert (r["added"], r["known"], r["skipped"]) == (5, 0, 0)
    assert rounds.import_folder(proj, src)["added"] == 0  # already there
    said = {name: text for _, name, _, text in clips}  # what Whisper would have heard
    rounds._save(proj, _with_texts(proj, said))
    by = {c["id"]: c["suggestion"] for c in rounds.load(proj)}
    assert by["kitchen/b.wav"] == by["desk/e.wav"] == "wake"
    assert by["desk/c.wav"] == "unclear" and by["kitchen/a.wav"] == "not_wake"
    assert rounds.accept_suggestions(proj) == 2  # clear "not the phrase" only
    assert (
        rounds.to_check(rounds.load(proj)) == 3
    )  # heard the phrase (2) or unclear (1): a person listens
    assert rounds.accept_suggestions(proj, ("wake",)) == 2  # the person agreed with both
    rounds.label(proj, {"desk/c.wav": "not_wake"})
    stats = rounds.apply(proj, "2026-10-24T00:00")
    assert stats == {"pos": {"train": 1, "heldout": 1}, "neg": {"train": 2, "heldout": 1}}
    real = RealData(proj)
    assert set(real.sources) == {"dpos", "dneg"}
    assert real.heldout("dpos") == ["desk/e.wav"] and real.heldout("dneg") == ["desk/d.wav"]
    assert real.training_items("wake")[0] == [("dpos", "kitchen/b.wav")]


def _with_texts(proj, said):
    from wakesmith import rounds

    clips = rounds.load(proj)
    for c in clips:
        c["text"] = said[c["id"].split("/")[1]]
        c["suggestion"] = rounds.suggest(proj.phrase, c["text"])
    return clips


def test_round_suggestions_need_the_whole_phrase():
    from wakesmith.rounds import suggest

    assert suggest("hey jarvis", "Hey, Jarvis.") == "wake"
    assert suggest("hey jarvis", "It's not really too enjoyable. Hey. So...") == "unclear"
    assert suggest("hey jarvis", "Thank you.") == "not_wake"
    assert suggest("okay nabu", "The wake word, okay nabu.") == "wake"  # you still decide


# --- optional AI assistant ---------------------------------------------------------------------------


@pytest.fixture
def chat_stub():
    """A tiny OpenAI-compatible chat server: answers every POST with a fixed reply and keeps
    each request (path, headers, body) in .seen."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    seen: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(
                {"path": self.path, "auth": self.headers.get("Authorization"), "body": body}
            )
            out = json.dumps(
                {
                    "model": "stub-1",
                    "choices": [{"message": {"content": "<think>hmm</think>Cause: **stub**."}}],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    srv.seen = seen
    srv.url = f"http://127.0.0.1:{srv.server_address[1]}"
    yield srv
    srv.shutdown()


def _closed_port() -> int:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_assistant_settings_are_private_and_off_by_default(tmp_path, monkeypatch):
    import os

    from wakesmith import assistant

    monkeypatch.setenv("WAKESMITH_HOME", str(tmp_path / "data"))
    assert assistant.load_settings()["enabled"] is False
    s = assistant.save_settings({"base_url": "http://box:8080/", "api_key": "sk-secret-123456"})
    assert s["base_url"] == "http://box:8080" and s["api_key"] == "sk-secret-123456"
    assert os.stat(assistant.settings_path()).st_mode & 0o777 == 0o600
    pub = assistant.public(assistant.load_settings())
    assert "api_key" not in pub and pub["has_key"] and pub["key_hint"] == "…3456"
    assert "secret" not in json.dumps(pub)
    assistant.save_settings({"model": "qwen"})  # an absent key keeps the saved one
    assert assistant.load_settings()["api_key"] == "sk-secret-123456"
    assistant.save_settings({"api_key": ""})
    assert assistant.load_settings()["api_key"] == ""
    with pytest.raises(ValueError):
        assistant.save_settings({"base_url": "box:8080"})
    assert assistant.endpoint("http://h:11434/v1") == "http://h:11434/v1/chat/completions"
    assert assistant.endpoint("http://h:8080") == "http://h:8080/v1/chat/completions"


def test_quick_checks_spot_the_common_failures():
    from wakesmith.assistant import quick_causes

    def titles(text):
        return [q["title"] for q in quick_causes(text)]

    assert titles("torch.OutOfMemoryError: XPU out of memory. Tried to allocate 2 GiB") == [
        "The GPU ran out of memory"
    ]
    assert titles("OSError: [Errno 28] No space left on device") == ["The disk is full"]
    assert titles("RuntimeError: microWakeWord training failed (exit -9): see x.log") == [
        "The system ran out of memory and stopped the job"
    ]
    assert titles("Killed") == ["The system ran out of memory and stopped the job"]
    assert "A file or download is missing" in titles(
        "FileNotFoundError: [Errno 2] No such file or directory: 'mww/x.npy'"
    )
    assert titles("ValueError: shapes do not match") == []
    q = quick_causes("12:00 start\nOSError: [Errno 28] No space left on device\n")[0]
    assert q["evidence"] == "OSError: [Errno 28] No space left on device"


def test_failure_prompt_is_compact_and_scrubbed(tmp_path):
    from wakesmith import assistant

    logs = tmp_path / "logs"
    logs.mkdir()
    old = "".join(f"01:00:{i % 60:02d} old run line {i}\n" for i in range(50))
    new = "".join(f"02:00:{i % 60:02d} step {i}\r02:00:00 step {i} done\n" for i in range(300))
    (logs / "train.log").write_text(
        "01:00:00 hardware: cpu\n"
        + old
        + "02:00:00 hardware: xpu\n"
        + new
        + "02:10:00 generate failed: generation of negative clips failed: see "
        "logs/generate-negative.log\n"
    )
    (logs / "generate-negative.log").write_text("UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY\n")
    lines = assistant.failure_logs(logs, "see logs/generate-negative.log")
    assert len(lines) <= assistant.LOG_LINES
    assert lines[0].startswith("--- logs/generate-negative.log") and "OUT_OF_DEVICE" in lines[1]
    assert not any("old run" in ln for ln in lines)  # only the latest run
    assert not any("\r" in ln for ln in lines) and "step 299 done" in lines[-2]

    data = {
        "name": "x",
        "phrase": "hey jarvis",
        "negatives": ["a", "b"],
        "real": {"positives": "/home/me/rec", "negatives": "", "copies": 10},
        "samples": {"positive": 100},
        "microwakeword": {"enabled": True, "steps": 500, "cutoff": 0.9, "secret_stuff": 1},
    }
    summary = assistant.project_summary(data)
    assert summary["sound_alikes"] == 2 and summary["real"]["phrase_recordings"] is True
    assert "/home/me" not in json.dumps(summary) and "secret_stuff" not in summary["microwakeword"]
    hw = assistant.hardware_summary(
        {"accel": "xpu", "gpus": [{"name": "Arc B70", "memory_gb": 32}], "cpus": 8, "problem": ""}
    )
    assert hw == {"accel": "xpu", "cpus": 8, "gpus": ["Arc B70 (32 GB)"]}
    msgs = assistant.failure_messages(
        summary,
        "generate",
        "failed at /home/me/x token=abc123",
        ["/home/me/proj/logs hf_" + "a" * 30],
        hw,
        assistant.quick_causes("XPU out of memory"),
        home="/home/me",
    )
    assert msgs[0]["role"] == "system" and "mww-train" in msgs[0]["content"]
    user = msgs[1]["content"]
    assert "Failed stage: generate" in user and "The GPU ran out of memory" in user
    assert "/home/me" not in user and "abc123" not in user and "hf_aaaa" not in user
    assert "~/proj/logs" in user


def test_advice_numbers_carry_what_the_page_shows():
    from wakesmith import assistant

    table = [
        {"threshold": 0.5, "caught": 40, "false": 9},
        {"threshold": 0.7, "caught": 36, "false": 2},
    ]
    pick = {
        "picked": 2,
        "why": "fewest false",
        "candidates": [
            {"n": 1, "seed": 5, "validation": 0.9, "caught": 39, "positives": 45, "false": 12},
            {"n": 2, "seed": 6, "failed": "exit -9"},
        ],
    }
    runs = [
        {
            "id": "20261024-010000",
            "label": "",
            "results": {"mww": {"table": table, "positives": 45, "negatives": 3990}},
        },
        {
            "id": "20261023-010000",
            "label": "c2",
            "results": {"mww": {"table": table, "positives": 40, "negatives": 3000}},
        },
    ]
    n = assistant.advice_numbers(
        "hey jarvis",
        {"mww": {"table": table, "positives": 45, "negatives": 3990}},
        {"mww": 0.7, "oww": 0.5},
        pick,
        {
            "total": 12,
            "labels": {"wake": 3, "not_wake": 8, "skip": 1},
            "unlabelled": 0,
            "first": "x",
        },
        runs,
    )
    m = n["models"]["mww"]
    assert m["chosen_cutoff"] == 0.7 and m["held_out_phrase_clips"] == 45
    assert m["by_cutoff"][1] == {"cutoff": 0.7, "caught": 36, "false": 2}
    assert n["mww_candidates"]["kept"] == 2 and "seed" not in n["mww_candidates"]["runs"][0]
    assert n["mww_candidates"]["runs"][1]["failed"] == "exit -9"
    assert n["device_clips"]["labels"]["wake"] == 3 and "first" not in n["device_clips"]
    assert n["recent_runs_at_chosen_cutoffs"][1]["run"] == "c2"
    assert "36/40 caught, 2/3000 false at 0.7" in n["recent_runs_at_chosen_cutoffs"][1]["mww"]
    msgs = assistant.advice_messages(n)
    assert "quote the exact numbers" in msgs[0]["content"] and "too thin" in msgs[0]["content"]
    assert '"caught": 36' in msgs[1]["content"]
    assert assistant.advice_numbers("x", {}, {})["models"] == {}


def test_assistant_endpoints(proj, tmp_path, monkeypatch, chat_stub):
    from fastapi.testclient import TestClient

    from wakesmith import hardware, jobs
    from wakesmith.web.server import create_app

    monkeypatch.setenv("WAKESMITH_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(
        hardware, "detect_safe", lambda timeout=30.0: (hardware.Hardware("cpu", [], 4, 16, 8), "")
    )
    c = TestClient(create_app(), base_url="http://localhost")
    proj.set_state(
        **{"mww-train": {"status": "failed", "error": "exit -9: see logs/mww-train.log"}}
    )
    proj.path("logs").mkdir(exist_ok=True)
    proj.path("logs", "train.log").write_text(
        "10:00:00 hardware: cpu\n10:05:00 === mww-train\nKilled\n"
    )
    (tmp_path / "res").mkdir()
    proj.path("results").mkdir(exist_ok=True)
    proj.path("results", "mww.json").write_text(
        json.dumps(
            {
                "table": [{"threshold": 0.9, "caught": 3, "false": 1}],
                "positives": 4,
                "negatives": 50,
            }
        )
    )

    # off: nothing is sent, the quick checks still answer
    assert c.get("/api/assistant").json()["enabled"] is False
    f = c.get(f"/api/projects/{proj.name}/failure").json()
    assert f["stage"] == "mww-train" and f["quick"][0]["title"].startswith("The system ran out")
    # wakesmith's own "stopped before it finished" note lists every cause; on its own it proves nothing
    proj.set_state(**{"mww-train": {"status": "failed", "error": jobs.STOPPED}})
    proj.path("logs", "train.log").write_text(
        "10:00:00 hardware: cpu\n10:05:00 === mww-train\nstep 100\n"
    )
    assert c.get(f"/api/projects/{proj.name}/failure").json()["quick"] == []
    proj.path("job.json").write_text(json.dumps({"pid": 1, "finished": 1.0, "result": "stopped"}))
    assert (
        c.get(f"/api/projects/{proj.name}/failure").json()["quick"][0]["title"]
        == "You stopped this run"
    )
    proj.path("job.json").unlink()
    proj.set_state(
        **{"mww-train": {"status": "failed", "error": "exit -9: see logs/mww-train.log"}}
    )
    proj.path("logs", "train.log").write_text(
        "10:00:00 hardware: cpu\n10:05:00 === mww-train\nKilled\n"
    )
    for path in (
        "/api/assistant/test",
        f"/api/projects/{proj.name}/explain",
        f"/api/projects/{proj.name}/advice",
    ):
        assert c.post(path).status_code == 409
    assert chat_stub.seen == []

    # on, with a key: requests reach the stub, the key never comes back
    r = c.put(
        "/api/assistant",
        json={"enabled": True, "base_url": chat_stub.url, "api_key": "k-12345678", "model": "m"},
    )
    assert r.status_code == 200 and "api_key" not in r.json() and r.json()["key_hint"] == "…5678"
    t = c.post("/api/assistant/test").json()
    assert t["answer"] == "Cause: **stub**." and t["model"] == "stub-1"
    assert chat_stub.seen[-1]["path"] == "/v1/chat/completions"
    assert (
        chat_stub.seen[-1]["auth"] == "Bearer k-12345678"
        and chat_stub.seen[-1]["body"]["model"] == "m"
    )
    e = c.post(f"/api/projects/{proj.name}/explain").json()
    assert e["stage"] == "mww-train" and "Killed" in e["sent"]
    assert "Failed stage: mww-train" in chat_stub.seen[-1]["body"]["messages"][1]["content"]
    a = c.post(f"/api/projects/{proj.name}/advice", json={"cutoff": 0.9}).json()
    assert a["numbers"]["models"]["mww"]["by_cutoff"] == [{"cutoff": 0.9, "caught": 3, "false": 1}]
    assert '"held_out_phrase_clips": 4' in chat_stub.seen[-1]["body"]["messages"][1]["content"]

    # the server is down: a clear message, not a crash
    c.put("/api/assistant", json={"base_url": f"http://127.0.0.1:{_closed_port()}"})
    r = c.post("/api/assistant/test")
    assert r.status_code == 502 and "couldn't reach a model server" in r.json()["detail"]
    r = c.post(f"/api/projects/{proj.name}/explain")
    assert r.status_code == 502 and "Is it running" in r.json()["detail"]


# --- everyday sentences (conversational negatives) -------------------------------------------------


@pytest.fixture
def llm_stub():
    """A tiny OpenAI-compatible server: each request gets the next canned reply; requests are kept."""
    import http.server
    import threading

    seen, replies = [], []

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append({"path": self.path, "auth": self.headers.get("Authorization"), **body})
            text = replies.pop(0) if replies else ""
            out = json.dumps(
                {"choices": [{"message": {"role": "assistant", "content": text}}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", seen, replies
    srv.shutdown()


def test_everyday_sentences_drop_the_phrase_repeats_and_odd_lengths():
    from wakesmith import conversational as cv

    reply = """Here you go:
1. Can you hand Jarvis the dishes tonight?
2) "Heyday tips for keeping the kitchen clean every day."
- I said hey, jarvis when the shelf fell over
* can you HAND JARVIS the dishes tonight
Hey.
• She kept the kelp in a jar by the window sill"""
    got = cv.keep(cv.parse(reply), ["hey jarvis"], set())
    assert got == [
        "Can you hand Jarvis the dishes tonight?",
        "Heyday tips for keeping the kitchen clean every day.",
        "She kept the kelp in a jar by the window sill",
    ]  # the preamble and "Hey." are too short; "hey, jarvis" says the phrase; the repeat is dropped
    assert cv.says_phrase("Hey-jarvis me!", ["hey jarvis"]) and not cv.says_phrase(
        "heys jarvis", ["hey jarvis"]
    )


def test_everyday_sentences_asked_in_batches_and_cached(proj, llm_stub):
    from wakesmith import conversational as cv

    url, seen, replies = llm_stub
    proj.conversational.llm_url = url + "/v1"
    lines = [f"number {i} jar of pickles is on the top shelf" for i in range(130)]
    replies += ["\n".join(lines[:50]), "\n".join(lines[40:90] + ["hey jarvis is that you over there"]),
                "\n".join(lines[90:130])]  # fmt: skip
    got = cv.collect(proj, 120, log=lambda m: None)
    assert len(got) == 120 and len(set(got)) == 120 and not any("jarvis" in s for s in got)
    assert [s["path"] for s in seen] == ["/v1/chat/completions"] * 3
    asked = [int(s["messages"][1]["content"].split()[1]) for s in seen]
    assert asked == [50, 50, 30]  # batches of 50, then the 30 still missing (10 repeats came back)
    assert (
        "hey jarvis" in seen[0]["messages"][1]["content"]
        and "jarvis" in seen[0]["messages"][1]["content"]
    )
    assert cv.cache_path(proj).read_text().splitlines() == got
    assert cv.collect(proj, 100) == got[:100] and len(seen) == 3  # from the cache: nothing asked


def test_everyday_sentences_stage_fails_clearly_without_the_assistant(proj):
    from wakesmith import conversational as cv
    from wakesmith import pipeline

    assert not pipeline.enabled(proj, "conversational")
    proj.conversational.clips, proj.conversational.llm_url = 10, "http://127.0.0.1:9"
    assert pipeline.enabled(proj, "conversational")
    with pytest.raises(RuntimeError, match="can't reach the AI assistant at http://127.0.0.1:9"):
        cv.run(proj, "cpu", 4)
    proj.save()
    assert project.load("hey_jarvis").conversational.clips == 10


def test_end_at_pause_cuts_between_words():
    from wakesmith import mww

    a = np.zeros(48000, np.float32)
    for s, e in ((4000, 12000), (16000, 26000), (30000, 40000)):  # three "words"
        a[s:e] = 0.5
    ends = {len(mww.end_at_pause(a, np.random.default_rng(i))) for i in range(60)}
    assert len(a) in ends and len(ends) > 2
    assert all(n == len(a) or not a[n - 320 : n].any() for n in ends)  # never inside a word
    assert all(n == len(a) or n > 4000 + 8000 for n in ends)  # at least half a second of speech


def test_everyday_sentences_use_the_assistant_settings_when_it_is_on(proj, monkeypatch, tmp_path):
    from wakesmith import assistant, conversational

    monkeypatch.setenv("WAKESMITH_HOME", str(tmp_path / "data"))
    for k in ("WAKESMITH_LLM_URL", "WAKESMITH_LLM_MODEL", "WAKESMITH_LLM_KEY"):
        monkeypatch.delenv(k, raising=False)
    assistant.save_settings({"base_url": "http://127.0.0.1:9999", "model": "m1", "api_key": "k"})
    assert conversational.endpoint(proj)[0] == conversational.DEFAULT_URL  # off: not used
    assistant.save_settings({"enabled": True})
    assert conversational.endpoint(proj) == ("http://127.0.0.1:9999", "m1", "k")
    proj.conversational.llm_url = "http://127.0.0.1:7777"  # the project's own setting wins
    assert conversational.endpoint(proj)[0] == "http://127.0.0.1:7777"


def test_explain_only_asks_when_the_log_shows_a_cause():
    from wakesmith import jobs
    from wakesmith.assistant import shows_a_cause

    calm = [
        "10:05:00 === mww-train",
        "mww candidate 1 of 3 (seed 5)",
        "FutureWarning: torch.jit.script",
    ]
    assert not shows_a_cause(jobs.STOPPED, calm, [], "")
    assert shows_a_cause(jobs.STOPPED, [*calm, "RuntimeError: XPU out of memory"], [], "")
    assert shows_a_cause("", calm, [], "failed: killed: most likely the machine ran out of memory")
    assert shows_a_cause("", calm, [{"title": "t", "advice": "a", "evidence": ""}], "")


# --- archive / storage ---------------------------------------------------------------------------
def _put(p, rel, n=1000):
    f = p.dir / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"x" * n)


KEPT = [
    "project.yaml",
    "logs/train.log",
    "results/mww.json",
    "runs/20261001-000000/run.json",
    "runs/20261001-000000/export/microwakeword/m.tflite",
    "export/microwakeword/m.tflite",
    "devices/pos/a.wav",
    "conversational/sentences.txt",
    "review/decisions.json",
    "mww/trained/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite",
    "mww/trained/logs/events",
    "mww/training.yaml",
    "oww/model/model.tflite",
    "clips/positive_train/clip_000000.wav",
]


def _trained(p, tmp_path):
    """A project as a finished best-of-3 run leaves it, its source recordings outside."""
    for kind in ("pos", "neg"):
        src = tmp_path / f"src_{p.name}_{kind}"
        src.mkdir()
        sf.write(src / "a.wav", np.zeros(16000, np.float32), 16000)
    p.real.positives, p.real.negatives = (
        str(tmp_path / f"src_{p.name}_pos"),
        str(tmp_path / f"src_{p.name}_neg"),
    )
    p.save()
    for rel in KEPT[1:]:
        _put(p, rel)
    p.path("results", "mww.json").write_text(
        json.dumps({"table": [], "positives": 1, "negatives": 1})
    )
    _put(p, "clips/positive_train/clip_000001.wav", 50_000)
    _put(p, "mww/features/positive/training/shard00_mmap/data", 2_000_000)
    _put(p, "oww/features/positive_train.npy", 500_000)
    _put(p, "real/train/pos__a.wav", 10_000)
    _put(p, "real/test/pos__b.wav")
    _put(p, "real_neg/train/neg__a_w0.wav", 100_000)
    _put(p, "real_neg/test/neg__c.wav")
    _put(p, "mww/trained.previous/best_weights.weights.h5", 20_000)
    _put(p, "models.json.part", 10)
    for i in (1, 2, 3):
        for small in (
            "best_weights.weights.h5",
            "last_weights.weights.h5",
            "stream_state_internal/saved_model.pb",
            "tflite_stream_state_internal_quant/stream_state_internal_quant.tflite",
        ):
            _put(p, f"mww/candidates/s{i}/{small}", 300)
        _put(p, f"mww/candidates/s{i}/logs/events", 40_000)
        _put(p, f"mww/candidates/s{i}/restore/ckpt", 40_000)
    p.path("mww", "candidates", "pick.json").write_text(json.dumps({"picked": 2}))
    p.path("models.json").write_text("[]")
    p.set_state(
        **{
            s: {"status": "done", "time": 1.0}
            for s in (
                "generate",
                "real",
                "mww-features",
                "mww-train",
                "oww-features",
                "oww-train",
                "test",
                "export",
            )
        }
    )


def _age(p, days):
    import os
    import time

    t = time.time() - days * 86400
    for root, dirs, files in os.walk(p.dir):
        for n in dirs + files:
            os.utime(os.path.join(root, n), (t, t), follow_symlinks=False)


def test_archive_removes_only_what_training_rebuilds(proj, tmp_path):
    import os

    from wakesmith import archive

    _trained(proj, tmp_path)
    other = project.create("ok husker", negatives=["husker"])
    os.symlink(proj.path("clips"), other.path("clips"))  # shares proj's clips
    outside = tmp_path / "outside_features"
    outside.mkdir()
    (outside / "data").write_bytes(b"x" * 5000)
    other.path("mww").mkdir()
    os.symlink(outside, other.path("mww", "features"))  # a link: never followed
    pl = archive.plan(proj)
    got = {r["path"] for r in pl["remove"]}
    assert {"mww/features", "oww/features", "real", "real_neg", "mww/trained.previous"} <= got
    assert "mww/candidates/s1/logs" in got and "mww/candidates/s3/restore" in got
    assert not any(g.startswith("mww/candidates/s2") for g in got)  # the picked one stays whole
    assert "models.json.part" in got and "clips" not in got
    assert pl["stages"] == ["real", "mww-features", "oww-features"]
    assert pl["bytes"] == sum(r["bytes"] for r in pl["remove"]) > 2_600_000

    r = archive.archive(proj)
    assert r["freed_bytes"] == pl["bytes"]
    for rel in KEPT:
        assert proj.path(rel).exists(), rel
    for i in (1, 2, 3):
        left = sorted(x.name for x in proj.path("mww", "candidates", f"s{i}").iterdir())
        assert set(left) >= archive.KEEP_CANDIDATE
        assert ("logs" in left) == (i == 2)
    assert not proj.path("mww", "features").exists() and not proj.path("real_neg").exists()
    assert (tmp_path / f"src_{proj.name}_neg" / "a.wav").exists()  # sources untouched
    st = proj.state()
    assert st["mww-features"]["status"] == st["real"]["status"] == "archived"
    assert st["mww-train"]["status"] == "done" and st["generate"]["status"] == "done"
    a = archive.info(proj)
    assert a["archived"] and a["freed_bytes"] == pl["bytes"] and a["to_rebuild"] == pl["stages"]

    # the other project: its links to clips and features are left alone, and so are their targets
    assert archive.plan(other)["bytes"] == 0
    archive.archive(other, clips=True)
    assert other.path("clips").is_symlink() and other.path("mww", "features").is_symlink()
    assert (outside / "data").exists() and proj.path("clips", "positive_train").is_dir()
    assert archive.size(other.dir) < 2000  # links aren't counted


def test_archive_clips_only_when_no_other_project_links_to_them(proj, tmp_path):
    import os

    from wakesmith import archive

    _trained(proj, tmp_path)
    other = project.create("ok husker", negatives=["husker"])
    os.symlink(proj.path("clips"), other.path("clips"))
    pl = archive.plan(proj, clips=True)
    assert "clips" not in {r["path"] for r in pl["remove"]}
    assert pl["clips"]["shared_with"] == [other.name] and "used by" in pl["kept"][0]
    other.path("mww").mkdir()
    os.symlink(proj.path("real_neg"), other.path("real_neg"))  # shares prepared recordings
    os.symlink(proj.path("mww", "features"), other.path("mww", "features"))
    got = {r["path"] for r in archive.plan(proj)["remove"]}
    assert "real_neg" not in got and "mww/features" not in got and "real" in got
    other.path("real_neg").unlink()
    other.path("mww", "features").unlink()
    other.path("clips").unlink()
    pl = archive.plan(proj, clips=True)
    assert "clips" in {r["path"] for r in pl["remove"]} and pl["stages"][0] == "generate"
    archive.archive(proj, clips=True)
    assert not proj.path("clips").exists() and proj.state()["generate"]["status"] == "archived"


def test_features_kept_when_their_clips_or_recordings_are_gone(proj, tmp_path):
    import os

    from wakesmith import archive

    _trained(proj, tmp_path)
    shutil.rmtree(proj.path("clips"))
    os.symlink(tmp_path / "gone", proj.path("clips"))  # a link to a deleted project's clips
    shutil.rmtree(tmp_path / f"src_{proj.name}_neg")
    got = {r["path"] for r in archive.plan(proj)["remove"]}
    assert "mww/features" not in got and "real_neg" not in got and "real" in got


def test_archive_refused_while_a_job_runs_or_waits(proj, tmp_path, monkeypatch):
    import time

    from wakesmith import archive

    _trained(proj, tmp_path)
    proj.path("job.json").write_text(json.dumps({"pid": None, "args": [], "started": time.time()}))
    with pytest.raises(archive.Busy, match="running"):
        archive.archive(proj)
    proj.path("job.json").unlink()
    monkeypatch.setattr(jobs, "kick", list)
    jobs.enqueue(proj, ["train", str(proj.dir)], "t")
    with pytest.raises(archive.Busy, match="queue"):
        archive.archive(proj)
    assert proj.path("mww", "features").exists()


def test_training_after_archive_rebuilds_what_it_needs_first(proj, tmp_path, monkeypatch):
    from wakesmith import archive, hardware, pipeline

    monkeypatch.setenv("WAKESMITH_HOME", str(tmp_path / "data"))
    _trained(proj, tmp_path)
    archive.archive(proj)
    assert pipeline.rebuild_first(proj, ["mww-train"]) == ["real", "mww-features"]
    assert pipeline.rebuild_first(proj, ["oww-train"]) == ["real", "oww-features"]
    assert pipeline.rebuild_first(proj, ["test"]) == ["real"]
    assert pipeline.rebuild_first(proj, ["export"]) == []
    ran = []
    monkeypatch.setattr(pipeline, "_run_stage", lambda p, s, hw, say: ran.append(s))
    monkeypatch.setattr(hardware, "detect_safe", lambda: (hardware.detect(gpu=False), None))
    pipeline.run(proj, "mww-train", "mww-train", log=lambda m: None)
    assert ran == ["real", "mww-features", "mww-train"]
    assert proj.state()["mww-features"]["status"] == "done"
    assert proj.state()["oww-features"]["status"] == "archived"  # not needed yet
    assert "rebuilding what archiving removed" in proj.path("logs", "train.log").read_text()
    assert archive.info(proj)["archived"] and archive.info(proj)["to_rebuild"] == ["oww-features"]
    pipeline.run(proj, "oww-train", "oww-train", log=lambda m: None)
    assert not archive.info(proj)["archived"]  # everything it removed is back


def test_auto_archive_respects_idle_days_and_models_in_use(proj, tmp_path, monkeypatch):
    from wakesmith import archive

    monkeypatch.setenv("WAKESMITH_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(archive, "AUTO_MIN_BYTES", 1_000_000)  # the fake features are 2 MB
    a = proj
    b = project.create("ok husker", negatives=["husker"])
    c = project.create("hello there", negatives=["hello"])
    for p in (a, b, c):
        _trained(p, tmp_path)
    b.path("models.json").write_text(json.dumps([{"kind": "mww", "sha": "x", "status": "offered"}]))
    _age(a, 20)
    _age(b, 20)
    _age(c, 3)
    rows = {r["name"]: r for r in archive.storage()}
    assert rows[a.name]["idle_days"] >= 19 and rows[c.name]["idle_days"] < 4
    assert archive.eligible(rows[a.name], 14) == ""
    assert "offered or live" in archive.eligible(rows[b.name], 14)
    assert "days ago" in archive.eligible(rows[c.name], 14)
    assert archive.settings()["archive_after_days"] == archive.DEFAULT_DAYS
    archive.save_settings({"archive_after_days": 0})
    assert archive.archive_idle() == []
    archive.save_settings({"archive_after_days": 14})
    done = archive.archive_idle()
    assert [d["project"] for d in done] == [a.name] and done[0]["by"] == "auto"
    assert not done[0]["clips"] and a.path("clips", "positive_train").is_dir()
    assert b.path("mww", "features").exists() and c.path("mww", "features").exists()
    assert "archived " + a.name in (project.projects_home() / "archive.log").read_text()
    assert archive.archive_idle() == []  # nothing much left to free


def test_storage_and_archive_api(proj, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from wakesmith.web.server import create_app

    monkeypatch.setenv("WAKESMITH_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(jobs, "kick", list)
    _trained(proj, tmp_path)
    c = TestClient(create_app(), base_url="http://localhost")
    d = c.get("/api/storage").json()
    row = d["projects"][0]
    assert row["name"] == proj.name and row["freeable"] > 2_600_000 and not row["archived"]
    assert row["clips"] == {"own": True, "link_to": "", "shared_with": []}
    assert d["settings"]["archive_after_days"] == 14 and row["auto_skip"]
    assert c.put("/api/storage", json={"archive_after_days": "x"}).status_code == 400
    assert c.put("/api/storage", json={"archive_after_days": 30}).json()["archive_after_days"] == 30
    item = jobs.enqueue(proj, ["train", str(proj.dir)], "t")
    assert c.post(f"/api/projects/{proj.name}/archive", json={}).status_code == 409
    jobs.dequeue(item["id"])
    r = c.post(f"/api/projects/{proj.name}/archive", json={"clips": False}).json()
    assert r["freed_bytes"] == row["freeable"] and "mww/features" in r["removed"]
    assert c.get("/api/projects").json()[0]["archived"] is True
    s = c.get(f"/api/projects/{proj.name}").json()
    assert s["archive"]["archived"] and s["stages"]["mww-features"]["status"] == "archived"
    assert c.get("/api/storage").json()["projects"][0]["archived"]
    assert c.post("/api/storage/archive-idle").json() == {"archived": []}


# --- combined model (ensemble.py) ----------------------------------------------------------------


def _candidates(proj, failed=(2,), without_weights=(4,), n=4, pick=True):
    """Fake best-of-N output: s<N>/ with weights and a .tflite, failed ones listed in pick.json."""
    from wakesmith import ensemble

    root = proj.path("mww", "candidates")
    cands = []
    for i in range(1, n + 1):
        c = {"n": i, "seed": 100 + i}
        if i in failed:
            c["failed"] = "exit -9"
        else:
            d = root / f"s{i}"
            (d / ensemble.TFLITE).parent.mkdir(parents=True)
            (d / ensemble.TFLITE).write_bytes(b"cand")
            if i not in without_weights:
                (d / ensemble.WEIGHTS).write_bytes(b"w")
        cands.append(c)
    if pick:
        (root / "pick.json").write_text(json.dumps({"picked": 1, "why": "", "candidates": cands}))
    proj.path("mww", "training.yaml").write_text("train_dir: trained\n")


def _fake_build(monkeypatch, calls, data=b"combined", rc=0):
    """ensemble.combine's TensorFlow child replaced by one that writes `data` as the model."""
    import types
    from pathlib import Path

    from wakesmith import ensemble

    def run(cmd, cwd, stdout, stderr, env, check):
        calls.append({"cmd": cmd, "cwd": cwd, "env": env})
        if rc == 0:
            out = Path(cmd[3]) / ensemble.TFLITE
            out.parent.mkdir(parents=True)
            out.write_bytes(data)
        return types.SimpleNamespace(returncode=rc)

    monkeypatch.setattr(ensemble.subprocess, "run", run)


def test_combine_setting_round_trips_and_is_set_through_the_api(proj):
    from fastapi.testclient import TestClient

    from wakesmith.web.server import create_app

    assert proj.microwakeword.combine is False
    proj.microwakeword.combine = True
    proj.save()
    assert project.load("hey_jarvis").microwakeword.combine is True
    c = TestClient(create_app(), base_url="http://localhost")
    r = c.put("/api/projects/hey_jarvis", json={"microwakeword": {"combine": False}}).json()
    assert r["microwakeword"]["combine"] is False and r["combined"] is None
    assert project.load("hey_jarvis").microwakeword.combine is False


def test_combine_uses_finished_candidates_and_leaves_the_others_alone(proj, monkeypatch):
    from wakesmith import ensemble

    _candidates(proj)  # s2 failed, s4 lost its weights
    assert [(c["n"], c["seed"]) for c in ensemble.available(proj)] == [(1, 101), (3, 103)]
    trained = proj.path("mww", "trained", "keep.txt")
    trained.parent.mkdir(parents=True)
    trained.write_text("the kept candidate")
    before = sorted(str(f) for f in proj.path("mww", "candidates").rglob("*"))
    calls = []
    _fake_build(monkeypatch, calls)
    with pytest.raises(ValueError, match="s2"):
        ensemble.combine(proj, [1, 2])
    assert not calls
    out = ensemble.combine(proj, log=lambda m: None)
    assert out == ensemble.model_path(proj) and out.read_bytes() == b"combined"
    (call,) = calls
    assert call["cmd"][1:3] == ["-m", "wakesmith.ensemble_build"]
    assert call["cmd"][4:] == [
        str(proj.path("mww", "candidates", f"s{n}", ensemble.WEIGHTS)) for n in (1, 3)
    ]
    assert call["cwd"] == proj.path("mww") and call["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
    info = ensemble.info(proj)
    assert info["candidates"] == [1, 3] and info["seeds"] == [101, 103]
    assert info["tensor_arena_size"] == 2 * ensemble.ARENA_PER_MODEL and info["bytes"] == 8
    assert ensemble.note(info) == "combined from s1,s3"
    assert trained.read_text() == "the kept candidate"
    assert sorted(str(f) for f in proj.path("mww", "candidates").rglob("*")) == before
    assert not list(ensemble.folder(proj).glob(".build-*"))  # the temporary folder is gone
    # a failed build keeps the earlier model and leaves no temporary folder
    _fake_build(monkeypatch, calls, rc=1)
    with pytest.raises(RuntimeError, match="combine.log"):
        ensemble.combine(proj)
    assert out.read_bytes() == b"combined" and not list(ensemble.folder(proj).glob(".build-*"))
    # one candidate left is nothing to combine
    with pytest.raises(ValueError, match="at least 2"):
        ensemble.combine(proj, [3])


def test_candidates_found_without_pick_json(proj):
    """A run stopped before the pick: the s<N> folders with weights count."""
    from wakesmith import ensemble

    _candidates(proj, failed=(), without_weights=(2,), n=3, pick=False)
    assert [c["n"] for c in ensemble.available(proj)] == [1, 3]


def test_combined_results_are_their_own_kind(proj, monkeypatch):
    from fastapi.testclient import TestClient

    from wakesmith import ensemble, evaluate
    from wakesmith.web.server import create_app

    table = [{"threshold": t, "caught": 3, "false": f} for t, f in ((0.5, 4), (0.9, 1))]
    res = {
        "table": table,
        "positives": 4,
        "negatives": 9,
        "pos": {"a.wav": 0.95},
        "neg": {"b.wav": 0.6},
    }
    proj.path("results").mkdir()
    for kind in ("mww", "mww-combined"):
        proj.path("results", f"{kind}.json").write_text(json.dumps(res))
    _candidates(proj, failed=(), without_weights=(), n=3)
    _fake_build(monkeypatch, [])
    ensemble.combine(proj, log=lambda m: None)
    c = TestClient(create_app(), base_url="http://localhost")
    got = c.get("/api/projects/hey_jarvis").json()
    assert list(got["results"]) == ["mww", "mww-combined"]
    assert got["combined"]["note"] == "combined from s1,s2,s3"
    assert set(c.get("/api/projects/hey_jarvis/sets").json()) == {"mww", "mww-combined"}
    # new candidates replace the combined model; the test then drops its old results
    ensemble.remove(proj)
    assert evaluate.score_combined(proj, log=lambda m: None) == {}
    assert not proj.path("results", "mww-combined.json").exists()
    assert proj.path("results", "mww.json").exists()


def test_export_offers_the_combined_model(proj, monkeypatch):
    from fastapi.testclient import TestClient

    from wakesmith import ensemble, models, mww
    from wakesmith.web.server import create_app

    single = mww.model_path(proj)
    single.parent.mkdir(parents=True)
    single.write_bytes(_tiny_tflite())
    with pytest.raises(ValueError, match="wakesmith combine"):
        export.export(proj, combined=True)
    _candidates(proj, failed=(), without_weights=(), n=3)
    _fake_build(monkeypatch, [], data=_tiny_tflite())
    ensemble.combine(proj, log=lambda m: None)
    proj.path("results").mkdir()
    table = [{"threshold": 0.9, "caught": 7, "false": 1}]
    proj.path("results", "mww-combined.json").write_text(
        json.dumps({"table": table, "positives": 8, "negatives": 40})
    )
    proj.microwakeword.cutoff = 0.9
    export.export(proj, combined=True)
    d = proj.path("export", "microwakeword")
    manifest = json.loads((d / "hey_jarvis.json").read_text())
    assert manifest["micro"]["tensor_arena_size"] == 3 * export.TENSOR_ARENA
    assert manifest["wake_word"] == "Hey Jarvis" and manifest["model"] == "hey_jarvis.tflite"
    assert export.model_frontend(d / "hey_jarvis.tflite") == "tflm"
    (e,) = models.load(proj)
    assert (
        e["kind"] == "mww" and e["notes"] == "combined from s1,s2,s3" and e["combined"] == [1, 2, 3]
    )
    assert e["history"][0]["note"] == "combined from s1,s2,s3"
    assert e["heldout"] == {"caught": 7, "positives": 8, "false": 1, "negatives": 40}
    # the kept candidate: the usual arena, no note; the setting picks when nothing is asked
    export.export(proj, combined=False)
    manifest = json.loads((d / "hey_jarvis.json").read_text())
    assert manifest["micro"]["tensor_arena_size"] == export.TENSOR_ARENA
    assert export.use_combined(proj) is None
    proj.microwakeword.combine = True
    assert export.use_combined(proj)["candidates"] == [1, 2, 3]
    proj.save()
    c = TestClient(create_app(), base_url="http://localhost")
    assert c.post("/api/projects/hey_jarvis/export", json={"combined": False}).status_code == 200
    assert json.loads((d / "hey_jarvis.json").read_text())["micro"]["tensor_arena_size"] == 30000
    assert c.post("/api/projects/hey_jarvis/export", json={}).status_code == 200  # the setting
    assert json.loads((d / "hey_jarvis.json").read_text())["micro"]["tensor_arena_size"] == 90000
    ensemble.remove(proj)
    r = c.post("/api/projects/hey_jarvis/export", json={"combined": True})
    assert r.status_code == 409 and "combine" in r.json()["detail"]


def test_combine_cli_takes_candidate_numbers(proj, monkeypatch):
    from typer.testing import CliRunner

    from wakesmith import ensemble
    from wakesmith.cli import app

    got = {}
    monkeypatch.setattr(
        ensemble, "combine", lambda p, c, log, previous=False: got.setdefault("c", c) or "m"
    )
    r = CliRunner().invoke(app, ["combine", str(proj.dir), "--candidates", "s1,3", "--no-test"])
    assert r.exit_code == 0 and got["c"] == [1, 3]


def test_pipeline_combines_after_the_pick_and_carries_on_if_it_fails(proj, monkeypatch):
    import types

    from wakesmith import candidates, ensemble, pipeline

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("WAKESMITH_CUDA_DEVICES", "all")
    monkeypatch.setattr(candidates, "train", lambda p, log_file, device, log: {"picked": 2})
    built = []
    monkeypatch.setattr(ensemble, "combine", lambda p, log: built.append(1))
    monkeypatch.setattr(ensemble, "info", lambda p: {"candidates": [1, 2, 3]})
    hw = types.SimpleNamespace(gpus=[])
    assert pipeline._run_stage(proj, "mww-train", hw, print) == {"picked": 2} and not built
    proj.microwakeword.combine = True
    info = pipeline._run_stage(proj, "mww-train", hw, print)
    assert built and info == {"picked": 2, "combined": "combined from s1,s2,s3"}

    def fail(p, log):
        raise RuntimeError("combining the candidates failed (exit 1)")

    monkeypatch.setattr(ensemble, "combine", fail)
    info = pipeline._run_stage(proj, "mww-train", hw, print)
    assert info["picked"] == 2 and info["combined"].startswith("failed:")


# --- best of N on two halves: pick on one, report on the other -----------------------------------


def _names(prefix, n, half=None):
    from wakesmith.evaluate import half_of

    out, i = [], 0
    while len(out) < n:
        name = f"{prefix}__c{i}.wav"
        if half is None or half_of(name) == half:
            out.append(name)
        i += 1
    return out


def test_held_out_halves_are_stable_and_disjoint():
    from wakesmith import evaluate

    names = [f"pos__house__take{i}.wav" for i in range(400)]
    first = {n: evaluate.half_of(n) for n in names}
    assert first == {n: evaluate.half_of(n) for n in reversed(names)}  # order and run don't matter
    pick = {n for n, h in first.items() if h == "pick"}
    report = {n for n, h in first.items() if h == "report"}
    assert pick and report and not pick & report and pick | report == set(names)
    assert 150 < len(pick) < 250  # about half
    r = {"pos": dict.fromkeys(names[:40], 0.9), "neg": dict.fromkeys(names[40:], 0.1)}
    a, b = evaluate.half(r, "pick"), evaluate.half(r, "report")
    assert a["positives"] + b["positives"] == 40 and not a["pos"].keys() & b["pos"].keys()
    sizes, note = evaluate.split(r["pos"], r["neg"])
    assert sizes["pick"]["positives"] == a["positives"] and "no part in choosing" in note


def _two_candidates(proj, monkeypatch, pos, neg, scores):
    """Fake best of 2: each mww.train writes a model whose bytes name it (c1, c2); its peak score per
    held-out clip comes from scores[model][clip] (0 when missing). Held-out clips are empty files."""
    from wakesmith import candidates, evaluate, mww

    src = proj.dir.parent / "negsrc"
    src.mkdir()
    proj.real.negatives = str(src)
    proj.microwakeword.candidates = 2
    for d, names in (("real", pos), ("real_neg", neg)):
        proj.path(d, "test").mkdir(parents=True)
        for n in names:
            proj.path(d, "test", n).write_bytes(b"")
    made = []

    def train(p, log_file, device, seed=None):
        made.append(seed)
        m = mww.model_path(p)
        shutil.rmtree(p.path("mww", "trained"), ignore_errors=True)
        m.parent.mkdir(parents=True)
        m.write_bytes(f"c{len(made)}".encode())

    monkeypatch.setattr(mww, "train", train)

    def scorer(tflite, window=5, frontend="tflm"):
        table = scores[tflite.read_text()]
        return lambda f: table.get(f.name, 0.0)

    monkeypatch.setattr(evaluate, "mww_scorer", scorer)
    return candidates.train(proj, proj.path("logs", "mww-train.log"), "cpu", log=lambda m: None)


def test_best_of_n_picks_on_one_half_and_the_test_reports_the_other(proj, monkeypatch):
    from wakesmith import evaluate, mww

    pos = _names("pos", 10, "pick") + _names("dpos", 10, "report")
    pk, rp = _names("neg", 20, "pick"), _names("dneg", 20, "report")
    # c1: clean on the pick half, 8 false triggers on the report half (9 in all)
    # c2: 3 false on the pick half, clean on the report half (3 in all: the whole-set winner)
    scores = {
        "c1": {**dict.fromkeys(pos, 0.9), **dict.fromkeys(rp[:8], 0.9)},
        "c2": {**dict.fromkeys(pos, 0.9), **dict.fromkeys(pk[1:4], 0.9)},
    }
    info = _two_candidates(proj, monkeypatch, pos, pk + rp, scores)
    assert info["picked"] == 1  # the whole set would have kept c2
    pick = json.loads(proj.path("mww", "candidates", "pick.json").read_text())
    assert pick["split"] == {
        "pick": {"positives": 10, "negatives": 20},
        "report": {"positives": 10, "negatives": 20},
    }
    c1, c2 = pick["candidates"]
    assert (c1["false"], c1["negatives"], c1["report"]["false"], c1["full"]["false"]) == (
        0,
        20,
        8,
        8,
    )
    assert (c2["false"], c2["report"]["false"], c2["full"]["false"]) == (3, 0, 3)
    assert "used for choosing" in pick["why"] and "no part in choosing" in pick["split_note"]
    # the test stage reports the kept model (c1) on the report half only, the whole set under "full"
    r = evaluate.score(proj, "mww", report_half=True)
    assert r["half"] == "report" and set(r["pos"]) == set(pos[10:]) and set(r["neg"]) == set(rp)
    row = next(x for x in r["table"] if x["threshold"] == 0.5)
    assert (r["positives"], r["negatives"], row["caught"], row["false"]) == (10, 20, 10, 8)
    full = next(x for x in r["full"]["table"] if x["threshold"] == 0.5)
    assert (r["full"]["positives"], r["full"]["negatives"], full["false"]) == (20, 40, 8)
    assert evaluate.brief(r)["full"]["positives"] == 20 and "pos" not in evaluate.brief(r)["full"]
    assert "no part in picking" in evaluate.format_table(r)
    saved = json.loads(proj.path("results", "mww.json").read_text())
    assert saved["half"] == "report" and saved["negatives"] == 20
    # by test set: only the report half's clips; the app says which half it shows
    assert sum(s["clips"] for s in evaluate.by_set(r, [0.5])) == 30
    from fastapi.testclient import TestClient

    from wakesmith.web.server import create_app

    c = TestClient(create_app(), base_url="http://localhost")
    got = c.get("/api/projects/hey_jarvis").json()["results"]["mww"]
    assert got["half"] == "report" and got["full"]["negatives"] == 40
    assert c.get("/api/projects/hey_jarvis/sets").json()["mww"]["half"] == "report"
    assert c.get("/api/projects/hey_jarvis/candidates").json()["split"]["report"]["positives"] == 10
    # a model not picked here (no report_half), or a project without halves: the whole set
    assert "half" not in evaluate.score(proj, "mww")
    model = mww.model_path(proj)
    kept = model.read_bytes()
    model.write_bytes(b"retrained with one candidate since")
    assert evaluate.pick_split(proj) is None
    model.write_bytes(kept)
    assert evaluate.pick_split(proj)
    proj.path("mww", "candidates", "pick.json").write_text(json.dumps({"picked": 1}))
    assert evaluate.score(proj, "mww", report_half=True)["positives"] == 20


def test_best_of_n_tiny_held_out_set_keeps_one_set(proj, monkeypatch):
    from wakesmith import evaluate

    pos = _names("pos", 3, "pick") + _names("pos", 3, "report")  # a half would have 3 < 5
    neg = _names("neg", 40)
    scores = {"c1": dict.fromkeys(neg[:5], 0.9), "c2": dict.fromkeys(pos + neg[:1], 0.9)}
    info = _two_candidates(proj, monkeypatch, pos, neg, scores)
    pick = json.loads(proj.path("mww", "candidates", "pick.json").read_text())
    assert info["picked"] == 2 and pick["split"] is None and "too few" in pick["split_note"]
    assert pick["candidates"][1]["positives"] == 6 and "report" not in pick["candidates"][1]
    r = evaluate.score(proj, "mww", report_half=True)
    assert "half" not in r and r["positives"] == 6 and r["negatives"] == 40


# --- the previous round's candidates are kept -----------------------------------------------------


def _round(proj, n=3, extras=True, combined=True):
    """A finished best-of-N round: s<N>/ with the small kept files (+ logs and a checkpoint), pick.json,
    a combined model."""
    from wakesmith import ensemble

    for i in range(1, n + 1):
        for small in (
            "best_weights.weights.h5",
            "last_weights.weights.h5",
            "stream_state_internal/saved_model.pb",
            "tflite_stream_state_internal_quant/stream_state_internal_quant.tflite",
        ):
            _put(proj, f"mww/candidates/s{i}/{small}", 300)
        if extras:
            _put(proj, f"mww/candidates/s{i}/logs/events", 40_000)
            _put(proj, f"mww/candidates/s{i}/restore/ckpt", 40_000)
            _put(proj, f"mww/candidates/s{i}/train.log", 4_000)
    cands = [{"n": i, "seed": 100 + i} for i in range(1, n + 1)]
    proj.path("mww", "candidates", "pick.json").write_text(
        json.dumps({"picked": 2, "why": "", "candidates": cands})
    )
    if combined:
        _put(proj, f"mww/combined/{ensemble.TFLITE}", 900)
        proj.path("mww", "combined", "combined.json").write_text(json.dumps({"candidates": [1, 2]}))
    proj.path("mww", "training.yaml").write_text("train_dir: trained\n")


def test_a_new_round_moves_the_candidates_to_previous_and_slims_them(proj):
    from wakesmith import candidates, ensemble

    _put(proj, "mww/candidates.previous/s9/best_weights.weights.h5", 10)  # two rounds ago
    _put(proj, "mww/combined.previous/old.json", 10)
    _round(proj)
    said = []
    candidates.keep_previous(proj, log=said.append)
    assert not proj.path("mww", "candidates").exists() and not proj.path("mww", "combined").exists()
    prev = proj.path("mww", "candidates.previous")
    assert sorted(d.name for d in prev.iterdir()) == [
        "pick.json",
        "s1",
        "s2",
        "s3",
    ]  # older replaced
    for i in (1, 2, 3):
        assert {x.name for x in (prev / f"s{i}").iterdir()} == candidates.KEEP_CANDIDATE
    assert (prev / "s1" / "stream_state_internal" / "saved_model.pb").exists()
    cprev = proj.path("mww", "combined.previous")
    assert (cprev / ensemble.TFLITE).exists() and not (cprev / "old.json").exists()
    assert "candidates and combined model" in said[0]
    assert ensemble.previous(proj) == {
        "candidates": 3,
        "numbers": [1, 2, 3],
        "picked": 2,
        "combined": True,
    }
    # the next round without a combined model: the older combined.previous goes with its candidates
    _round(proj, n=2, combined=False)
    candidates.keep_previous(proj, log=said.append)
    assert ensemble.previous(proj)["candidates"] == 2
    assert not proj.path("mww", "combined.previous").exists()
    # nothing to keep: nothing changes
    candidates.keep_previous(proj, log=said.append)
    assert ensemble.previous(proj)["candidates"] == 2 and len(said) == 2


def test_best_of_n_training_keeps_the_last_round(proj, monkeypatch):
    from wakesmith import candidates, mww

    _round(proj)
    proj.microwakeword.candidates = 2

    def train(p, log_file, device, seed=None):
        m = mww.model_path(p)
        shutil.rmtree(p.path("mww", "trained"), ignore_errors=True)
        m.parent.mkdir(parents=True)
        m.write_bytes(b"new")

    monkeypatch.setattr(mww, "train", train)
    candidates.train(proj, proj.path("logs", "mww-train.log"), "cpu", log=lambda m: None)
    now = sorted(d.name for d in proj.path("mww", "candidates").iterdir())
    assert now == ["pick.json", "s1", "s2"]
    prev = proj.path("mww", "candidates.previous")
    assert sorted(d.name for d in prev.iterdir()) == ["pick.json", "s1", "s2", "s3"]
    assert (prev / "s3" / "best_weights.weights.h5").stat().st_size == 300
    assert proj.path("mww", "combined.previous").is_dir()


def test_archive_keeps_the_previous_rounds_weights(proj):
    from wakesmith import archive, candidates, ensemble

    _round(proj)
    candidates.keep_previous(proj, log=lambda m: None)
    for i in (1, 2):  # leftovers in the previous round (e.g. moved by an older wakesmith)
        _put(proj, f"mww/candidates.previous/s{i}/logs/events", 40_000)
    got = {r["path"] for r in archive.plan(proj)["remove"]}
    assert got >= {"mww/candidates.previous/s1/logs", "mww/candidates.previous/s2/logs"}
    assert not any("combined.previous" in g or g.endswith(".h5") for g in got)
    archive.archive(proj)
    for i in (1, 2, 3):
        left = {x.name for x in proj.path("mww", "candidates.previous", f"s{i}").iterdir()}
        assert left == candidates.KEEP_CANDIDATE
    assert proj.path("mww", "candidates.previous", "pick.json").exists()
    assert (proj.path("mww", "combined.previous") / ensemble.TFLITE).exists()


def test_combine_the_previous_round(proj, monkeypatch):
    from fastapi.testclient import TestClient
    from typer.testing import CliRunner

    from wakesmith import candidates, ensemble
    from wakesmith.cli import app
    from wakesmith.web.server import create_app

    _round(proj, combined=False)
    candidates.keep_previous(proj, log=lambda m: None)
    _candidates(proj, failed=(), without_weights=(), n=2)  # this round's s1, s2
    calls = []
    _fake_build(monkeypatch, calls)
    with pytest.raises(ValueError, match="previous round's candidate"):
        ensemble.combine(proj, [4], log=lambda m: None, previous=True)
    out = ensemble.combine(proj, log=lambda m: None, previous=True)
    assert out == ensemble.model_path(proj)
    assert calls[0]["cmd"][4:] == [
        str(proj.path("mww", "candidates.previous", f"s{n}", ensemble.WEIGHTS)) for n in (1, 2, 3)
    ]
    info = ensemble.info(proj)
    assert info["previous"] and info["seeds"] == [101, 102, 103]
    assert ensemble.note(info) == "combined from the previous round's s1,s2,s3"
    assert proj.path("mww", "candidates", "s1", ensemble.WEIGHTS).exists()  # this round untouched
    # the CLI flag and the app
    got = {}
    monkeypatch.setattr(
        ensemble, "combine", lambda p, c, log, previous: got.update(c=c, prev=previous) or "m"
    )
    r = CliRunner().invoke(app, ["combine", str(proj.dir), "--previous", "--no-test"])
    assert r.exit_code == 0 and got == {"c": None, "prev": True}
    started = []
    monkeypatch.setattr(
        "wakesmith.jobs.start", lambda p, args, title: started.append(args) or {"ok": 1}
    )
    c = TestClient(create_app(), base_url="http://localhost")
    assert c.get("/api/projects/hey_jarvis/candidates").json()["previous"]["candidates"] == 3
    c.post("/api/projects/hey_jarvis/combine", json={"previous": True})
    c.post("/api/projects/hey_jarvis/combine")
    assert started == [["combine", str(proj.dir), "--previous"], ["combine", str(proj.dir)]]


# --- routine device-clip sync (clipsync.py) ------------------------------------------------------


def _puller(root, fires):
    """A puller folder: <device>/help/<name>.wav + .json, fires = [(device, name, at, model)]."""
    for dev, name, at, model in fires:
        d = root / dev / "help"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.wav").write_bytes(_wav(0.5))
        side = {"at": at, "device": dev, "model": model, "score": 0.6, "slot": 107}
        (d / f"{name}.json").write_text(json.dumps(side))


class _FakeWhisper:
    def __init__(self, said):
        self.said, self.heard = said, []

    def __call__(self, paths):
        self.heard += [f.stem for f in paths]
        return [self.said[f.stem] for f in paths]


SAID = {"a": "hey jarvis", "b": "thank you", "c": "hey there", "d": "Hey, Jarvis!", "e": "x"}


def test_clip_sync_takes_only_the_words_model_and_transcribes_only_new_clips(proj, tmp_path):
    from wakesmith import clipsync, rounds

    src = tmp_path / "incoming"
    _puller(
        src,
        [
            ("kitchen", "a", "2026-10-24T18:10:00Z", "hey_jarvis_c3_s1"),
            ("kitchen", "b", "2026-10-24T18:20:00Z", "hey_jarvis"),
            ("desk", "c", "2026-10-24T18:30:00Z", "hey_jarvis_c3_s1"),
            ("desk", "x", "2026-10-24T18:40:00Z", "okay_nabu_c3_s1"),  # the other word
        ],
    )
    (src / "desk" / "help" / "nosidecar.wav").write_bytes(_wav(0.5))  # no model: not ours
    w = _FakeWhisper(SAID)
    r = clipsync.sync(proj, str(src), "hey_jarvis", "cpu", whisper=w)
    assert {c["id"] for c in rounds.load(proj)} == {"kitchen/a.wav", "kitchen/b.wav", "desk/c.wav"}
    assert sorted(w.heard) == ["a", "b", "c"]
    assert (r["new"], r["wake"], r["not_wake"], r["unclear"], r["to_check"]) == (3, 1, 1, 1, 2)
    assert r["auto_labelled"] == 1  # the clear "not the phrase" clip
    assert r["device"] == "cpu" and clipsync.last_result(proj)["new"] == 3
    # the next run: one new fire; the old ones are neither copied nor transcribed again
    _puller(src, [("kitchen", "d", "2026-10-24T19:00:00Z", "hey_jarvis_c3_s1")])
    w.heard.clear()
    r = clipsync.sync(proj, str(src), "hey_jarvis", "cpu", whisper=w)
    assert w.heard == ["d"] and (r["new"], r["wake"], r["to_check"], r["total"]) == (1, 1, 3, 4)
    w.heard.clear()
    r = clipsync.sync(proj, str(src), "hey_jarvis", "cpu", whisper=w)
    assert w.heard == [] and r["new"] == 0 and r["transcribed"] == 0
    assert "0 new clips; 3 waiting for a person" in clipsync.describe(r)
    # nothing in the source folder was moved or removed
    assert len(list(src.rglob("*.wav"))) == 6 and len(list(src.rglob("*.json"))) == 5
    log = clipsync.log_file().read_text()
    assert log.count(f"{proj.name}: ") == 3 and "1 unclear" in log


def test_clip_sync_since_limit_and_project_settings(proj, tmp_path):
    from wakesmith import clipsync, rounds

    src = tmp_path / "incoming"
    _puller(
        src,
        [
            ("desk", "a", "2026-10-23T10:00:00Z", "hey_jarvis"),  # before the model went live
            ("desk", "b", "2026-10-24T18:20:00Z", "hey_jarvis"),
            ("desk", "c", "2026-10-24T18:30:00Z", "hey_jarvis"),
            ("desk", "d", "2026-10-24T18:40:00Z", "hey_jarvis"),
        ],
    )
    with pytest.raises(ValueError, match="source"):
        clipsync.sync(proj, whisper=_FakeWhisper(SAID))
    proj.devices.source = str(src)
    proj.devices.model_prefix = "hey_jarvis"
    proj.devices.since = "2026-10-24T18:00"
    proj.save()
    p = project.load(proj.name)
    assert vars(p.devices) == {
        "source": str(src),
        "model_prefix": "hey_jarvis",
        "since": "2026-10-24T18:00",
    }
    r = clipsync.sync(p, device="cpu", limit=2, whisper=_FakeWhisper(SAID))
    assert (r["new"], r["left"]) == (2, 1) and "1 more next time" in clipsync.describe(r)
    assert [c["id"] for c in rounds.load(p)] == ["desk/b.wav", "desk/c.wav"]  # oldest first
    r = clipsync.sync(p, device="cpu", whisper=_FakeWhisper(SAID))
    assert (r["new"], r["left"], r["total"]) == (1, 0, 3)
    # an old project.yaml without a devices section still loads
    y = proj.path("project.yaml")
    data = project.yaml.safe_load(y.read_text())
    data.pop("devices", None)
    y.write_text(project.yaml.safe_dump(data))
    assert project.load(proj.name).devices.source == ""


def test_clip_sync_auto_device_and_running_jobs(proj, tmp_path, monkeypatch):
    import os

    from wakesmith import clipsync, hardware

    other = project.create("computer")
    gpu = type("G", (), {"device": "xpu:0", "memory_gb": 32})()
    hw = type("H", (), {"gpus": [gpu]})()
    monkeypatch.setattr(hardware, "detect_safe", lambda: (hw, None))
    monkeypatch.setattr(jobs, "running_all", list)
    assert clipsync.pick_device(proj, "auto") == "xpu:0"
    assert clipsync.pick_device(proj, "cuda:1") == "cuda:1"
    monkeypatch.setattr(jobs, "running_all", lambda: [other])  # training holds the GPU
    assert clipsync.pick_device(proj, "auto") == "cpu"
    # the same project's job is running: no sync (unless that job is this sync, from the UI)
    src = tmp_path / "incoming"
    _puller(src, [("desk", "a", "2026-10-24T18:10:00Z", "hey_jarvis")])
    monkeypatch.setattr(jobs, "status", lambda p: {"running": True, "pid": 999999})
    with pytest.raises(clipsync.Busy, match="running"):
        clipsync.sync(proj, str(src), "", "cpu", whisper=_FakeWhisper(SAID))
    monkeypatch.setattr(jobs, "status", lambda p: {"running": True, "pid": os.getppid()})
    assert clipsync.sync(proj, str(src), "", "auto", whisper=_FakeWhisper(SAID))["new"] == 1
    # sync-all: every project with a source; a busy one is skipped, the rest carry on
    proj.devices.source = str(src)
    proj.save()
    monkeypatch.setattr(jobs, "status", lambda p: {"running": True, "pid": 999999})
    (r,) = clipsync.sync_all("cpu")
    assert r["result"] == "skipped" and r["project"] == proj.name


def test_clip_sync_timer_units(tmp_path):
    from wakesmith import clipsync

    files = clipsync.unit_files(
        "3:05", ["/v/bin/python", "-m", "wakesmith", "clips", "sync-all"], {}, tmp_path / "l.log"
    )
    svc, tmr = files["wakesmith-clips-sync.service"], files["wakesmith-clips-sync.timer"]
    assert "Type=oneshot" in svc and "ExecStart=/v/bin/python -m wakesmith clips sync-all" in svc
    assert f"StandardOutput=append:{tmp_path / 'l.log'}" in svc
    assert "OnCalendar=*-*-* 03:05:00" in tmr and "Persistent=true" in tmr
    assert "Unit=wakesmith-clips-sync.service" in tmr and "WantedBy=timers.target" in tmr
    svc = clipsync.unit_files("23:59", ["/a b/python"], {"WAKESMITH_PROJECTS": "/p"}, tmp_path)[
        "wakesmith-clips-sync.service"
    ]
    assert 'Environment="WAKESMITH_PROJECTS=/p"' in svc and 'ExecStart="/a b/python"' in svc
    for bad in ("24:00", "3", "noon", "12:60"):
        with pytest.raises(ValueError):
            clipsync.unit_files(bad, ["x"], {}, tmp_path)
    assert clipsync.command()[1:] == ["-m", "wakesmith", "clips", "sync-all"]


class _FakeSystemctl:
    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **kw):
        import subprocess

        self.calls.append(cmd)
        out = ""
        if "show" in cmd and cmd[3].endswith(".timer"):
            out = "UnitFileState=enabled\nNextElapseUSecRealtime=Sun 2026-10-25 03:30:00 MDT\n"
        elif "show" in cmd:
            out = "Result=success\nActiveState=inactive\n"
        return subprocess.CompletedProcess(cmd, 0, out, "")


def test_clip_sync_schedule_installer(proj, tmp_path, monkeypatch):
    from wakesmith import clipsync

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    fake = _FakeSystemctl()
    s = clipsync.install("03:30", run=fake)
    d = tmp_path / "config" / "systemd" / "user"
    assert (d / "wakesmith-clips-sync.timer").exists() and (
        d / "wakesmith-clips-sync.service"
    ).exists()
    assert (
        f"WAKESMITH_PROJECTS={project.projects_home()}"
        in (d / "wakesmith-clips-sync.service").read_text()
    )
    assert fake.calls[:2] == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", "--now", "wakesmith-clips-sync.timer"],
    ]
    assert s["installed"] and s["enabled"] and s["at"] == "03:30" and "03:30" in s["next"]
    assert s["last_result"] == "success"
    s = clipsync.remove(run=fake)
    assert not s["installed"] and not list(d.iterdir())
    assert ["systemctl", "--user", "disable", "--now", "wakesmith-clips-sync.timer"] in fake.calls

    def failing(cmd, **kw):
        import subprocess

        return subprocess.CompletedProcess(cmd, 1, "", "Failed to connect to bus")

    with pytest.raises(RuntimeError, match="bus"):
        clipsync.install("04:00", run=failing)


def test_clip_sync_cli_and_api(proj, tmp_path, monkeypatch):
    import subprocess

    from fastapi.testclient import TestClient
    from typer.testing import CliRunner

    from wakesmith import clipsync, rounds
    from wakesmith.cli import app
    from wakesmith.web.server import create_app

    monkeypatch.setenv("WAKESMITH_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(jobs, "kick", list)
    fake = _FakeSystemctl()
    monkeypatch.setattr(subprocess, "run", fake)  # no real systemctl anywhere below
    monkeypatch.setattr(rounds, "_Whisper", lambda device, **kw: _FakeWhisper(SAID))
    src = tmp_path / "incoming"
    _puller(
        src,
        [
            ("desk", "a", "2026-10-24T18:10:00Z", "hey_jarvis_c3"),
            ("desk", "c", "2026-10-24T18:30:00Z", "hey_jarvis_c3"),
            ("desk", "e", "2026-10-24T18:40:00Z", "okay_nabu"),
        ],
    )
    r = CliRunner().invoke(
        app,
        ["clips", "sync", proj.name, "--from", str(src), "--model-prefix", "hey_jarvis"]
        + ["--transcribe-device", "cpu", "--save"],
    )
    assert r.exit_code == 0, r.output
    assert "2 new clips; suggested: 1 the phrase, 0 not the phrase, 1 unclear" in r.output
    assert project.load(proj.name).devices.model_prefix == "hey_jarvis"
    r = CliRunner().invoke(app, ["clips", "sync-all", "--transcribe-device", "cpu"])
    assert r.exit_code == 0 and f"{proj.name}: 0 new clips; 2 waiting for a person" in r.output
    r = CliRunner().invoke(app, ["clips", "schedule", "--at", "04:15"])
    assert r.exit_code == 0 and "on, at 04:15" in r.output
    r = CliRunner().invoke(app, ["clips", "schedule", "--status"])
    assert "next run: Sun 2026-10-25" in r.output
    c = TestClient(create_app(), base_url="http://localhost")
    assert c.get("/api/projects").json()[0]["to_check"] == 2
    d = c.get(f"/api/projects/{proj.name}/devices").json()
    assert d["sync"]["source"] == str(src) and d["sync"]["last"]["to_check"] == 2
    assert "waiting for a person" in d["sync"]["last"]["text"] and d["summary"]["to_check"] == 2
    base = f"/api/projects/{proj.name}/devices/sync"
    assert c.put(base, json={"source": str(tmp_path / "nope")}).status_code == 400
    assert (
        c.put(base, json={"source": str(src), "model_prefix": "hey_jarvis_c3"}).json()[
            "model_prefix"
        ]
        == "hey_jarvis_c3"
    )
    started = []
    monkeypatch.setattr(jobs, "start", lambda p, args, title: started.append(args) or {"ok": 1})
    assert c.post(base).status_code == 200 and started == [["clips", "sync", str(proj.dir)]]
    s = c.get("/api/clips-schedule").json()
    assert s["installed"] and s["at"] == "04:15"
    assert c.put("/api/clips-schedule", json={"on": True, "at": "25:00"}).status_code == 400
    assert c.put("/api/clips-schedule", json={"on": True, "at": "05:00"}).json()["at"] == "05:00"
    assert c.put("/api/clips-schedule", json={"on": False}).json()["installed"] is False
    c.put(base, json={"source": ""})
    assert c.post(base).status_code == 400  # no folder yet
    assert not (tmp_path / "data" / "settings.json").exists()
    assert all(call[:2] == ["systemctl", "--user"] for call in fake.calls)
    assert clipsync.unit_dir() == tmp_path / "config" / "systemd" / "user"


# --- record a room (sessions.py) -----------------------------------------------------------------


def _room_audio(seed=0):
    """Synthetic room recording: 5 s of quiet room, 3 claps 2 s apart, then 'speech' bursts with
    pauses; returns (audio, clap onsets, speech (start, end) list)."""
    rng = np.random.default_rng(seed)
    sr = 16000
    parts, t = [], 0.0
    claps, speech = [], []

    def quiet(s):
        nonlocal t
        parts.append(rng.normal(0, 0.001, int(s * sr)))
        t += s

    def clap():
        nonlocal t
        n = int(0.4 * sr)
        burst = rng.normal(0, 1, n) * 0.5 * np.exp(-np.arange(n) / (0.03 * sr))
        parts.append(burst + rng.normal(0, 0.001, n))
        claps.append(t)
        t += 0.4

    def say(s):
        nonlocal t
        n = int(s * sr)
        x = np.arange(n) / sr
        env = np.minimum(1, x / 0.08) * np.minimum(1, (s - x) / 0.08)
        voice = sum(np.sin(2 * np.pi * f * x) / k for k, f in enumerate((140, 280, 420, 700), 1))
        parts.append(
            0.08 * env * voice * (1 + 0.3 * np.sin(2 * np.pi * 4 * x)) + rng.normal(0, 0.001, n)
        )
        speech.append((t, t + s))
        t += s

    quiet(5.0)
    for _ in range(3):
        clap()
        quiet(1.6)
    for s in (0.6, 0.7, 0.9):  # spot marker, phrase, phrase
        say(s)
        quiet(1.2)
    quiet(3.4)  # other speech, away from the takes
    say(0.6)
    quiet(4.6)
    say(0.5)  # three quick phrases in one breath: pauses of 0.3 s (under the merge gap)
    quiet(0.3)
    say(0.5)
    quiet(0.3)
    say(0.5)
    quiet(1.0)
    return np.concatenate(parts).astype(np.float32), claps, speech


@pytest.fixture
def room(tmp_path, monkeypatch):
    monkeypatch.setenv("WAKESMITH_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path / "data" / "projects"))
    a = project.create("hey jarvis", name="hh")
    b = project.create("okay computer", name="ha")
    return a, b


def _room_session(room, monkeypatch, texts):
    from wakesmith import sessions

    s = sessions.create("Study", ["Sam"], ["hh", "ha"])
    a, claps, speech = _room_audio()
    pcm = (np.clip(a, -1, 1) * 32767).astype("<i2")
    step = 16000 * 10
    for i in range(0, len(pcm), step):
        sessions.add_chunk(s["id"], i // step, pcm[i : i + step].tobytes())
    sessions.finish(s["id"])
    it = iter(texts)
    monkeypatch.setattr(
        sessions,
        "_transcriber",
        lambda device, model="turbo", language="en": lambda paths: [next(it) for _ in paths],
    )
    return s["id"], claps, speech


def test_also_written_as_finds_a_phonetic_phrase_in_transcripts(room, monkeypatch):
    from fastapi.testclient import TestClient

    from wakesmith import rounds, sessions, versions
    from wakesmith.web.server import create_app

    texts = [
        "Across the room.",
        "Hey, Jarvus!",
        "Okay, computer.",
        "What's for dinner?",
        "Hey jarvis.",
    ]
    sid, _, _ = _room_session(room, monkeypatch, texts)
    sessions.split(sid, "cpu", log=lambda m: None)
    sp = [x for x in sessions.segments(sid) if x["kind"] == "speech"]
    assert sp[1]["label"] == "unclear" and sp[1]["note"] == "part of a wake word"
    v2 = versions.create(project.load("hh"))
    c = TestClient(create_app(), base_url="http://localhost")
    r = c.put(
        "/api/projects/hh", json={"written_as": " Hey  Jarvus, hey jarvis,, hey jarvus"}
    ).json()
    assert r["written_as"] == ["Hey Jarvus"]  # tidied; the phrase itself and repeats are dropped
    assert r["folder"] == str(project.load("hh").dir)
    assert project.load(v2.name).written_as == ["Hey Jarvus"]  # every version of the wake word
    assert project.load("hh").heard_forms == ["hey jarvis", "Hey Jarvus"]
    it = iter(texts)  # Split again sees the spelling added after the recording was made
    monkeypatch.setattr(
        sessions,
        "_transcriber",
        lambda device, model="turbo", language="en": lambda paths: [next(it) for _ in paths],
    )
    sessions.split(sid, "cpu", log=lambda m: None)
    assert [x for x in sessions.segments(sid) if x["kind"] == "speech"][1]["label"] == "hh"
    assert project.load("hh").written_as and "Jarvus" not in str(project.load("hh").phrase)
    # device clips: any form of the phrase is the wake word
    forms = project.load("hh").heard_forms
    assert rounds.suggest(forms, "Hey Jarvus.") == "wake"
    assert rounds.suggest(forms, "Hey there.") == "unclear"
    assert rounds.suggest(forms, "Thank you.") == "not_wake"
    assert c.put("/api/projects/hh", json={"written_as": 5}).status_code == 400
    # one word of a spelling alone is part of a wake word (listen and pick), not other speech
    s = sessions.load(sid)
    sessions._fresh_spellings(s)  # as split does
    assert sessions.label_text("jarvus", s)[:1] == ("unclear",)
    # set together with other settings, it still reaches every version
    c.put(
        "/api/projects/hh",
        json={"written_as": "hey jervis", "note": "x", "negatives": ["hey travis"]},
    )
    assert project.load(v2.name).written_as == ["hey jervis"]
    # the list the menu is drawn from carries them (the prompt starts from what's saved)
    row = next(x for x in c.get("/api/projects").json() if x["name"] == "hh")
    assert row["written_as"] == ["hey jervis"]
    # another wake word's phrase is not a spelling of this one
    r = c.put("/api/projects/hh", json={"written_as": "Okay  Computer!"})
    assert r.status_code == 400 and project.load("hh").written_as == ["hey jervis"]
    # a mixed save refused for another setting changes no version
    r = c.put(
        "/api/projects/hh", json={"written_as": "hey jarvas", "conversational": {"clips": "lots"}}
    )
    assert r.status_code == 400
    assert project.load("hh").written_as == project.load(v2.name).written_as == ["hey jervis"]


def test_room_segments_claps_speech_and_room_sound():
    from wakesmith import sessions

    a, claps, speech = _room_audio()
    found = sessions.find(a)
    kinds = [x["kind"] for x in found]
    assert kinds.count("clap") == 3
    assert kinds[0] == "room" and found[0]["end"] - found[0]["start"] >= 4
    got_claps = [x["onset"] for x in found if x["kind"] == "clap"]
    assert all(abs(g - c) < 0.02 for g, c in zip(got_claps, claps, strict=True))
    sp = [x for x in found if x["kind"] == "speech"]
    assert len(sp) == 5  # the three quick ones are one stretch
    assert all(
        x["start"] < s and x["end"] > e for x, (s, e) in zip(sp[:4], speech[:4], strict=True)
    )
    assert sessions.room_pieces(0, 12) == [(0, 4), (4, 8), (8, 12)]
    assert all(3 <= e - s <= 5 for s, e in sessions.room_pieces(1.0, 15.3))
    assert sessions.room_pieces(0, 2.0) == []


def test_room_labels_by_phrase_and_spot_markers_carry_on(room, monkeypatch):
    from wakesmith import sessions

    texts = [
        "Across the room.",
        "Hey, Jarvis!",
        "Okay, computer.",
        "What's for dinner?",
        "Hey jarvis. Hey jarvis. Hey jarvis.",
    ]
    sid, _, speech = _room_session(room, monkeypatch, texts)
    r = sessions.split(sid, "cpu", log=lambda m: None)
    segs = sessions.segments(sid)
    labels = [x["label"] for x in segs]
    assert labels.count("clap") == 3 and labels[0] == "room"
    sp = [x for x in segs if x["kind"] == "speech"]
    assert [x["label"] for x in sp] == ["spot", "hh", "ha", "other", "hh", "hh", "hh"]
    assert all(x["spot"] == "across the room" for x in sp)
    parts = sp[4:]
    assert parts[0]["start"] < speech[4][0] and speech[4][1] <= parts[1]["start"] + 0.2
    assert all("of 3" in x["note"] for x in parts) and r["labels"]["hh"] == 4
    assert all(x["speaker"] == "Sam" for x in segs)
    # one "hey" of "hey jarvis" is unclear; two wake words in one stretch too
    s = sessions.load(sid)
    assert sessions.label_text("jarvis", s)[0] == "unclear"
    assert sessions.label_text("hey jarvis okay computer", s)[0] == "unclear"
    assert sessions.label_text("with the T.V. on", s)[:3] == ("spot", 0, "TV or music on")
    # a take Whisper misheard, next to other takes, is unclear (never sent as not-the-phrase)
    texts[2] = "Well, they work inside the food place."
    it = iter(texts)
    monkeypatch.setattr(
        sessions,
        "_transcriber",
        lambda device, model="turbo", language="en": lambda paths: [next(it) for _ in paths],
    )
    sessions.split(sid, "cpu", log=lambda m: None)
    sp = [x for x in sessions.segments(sid) if x["kind"] == "speech"]
    assert [x["label"] for x in sp][:4] == ["spot", "hh", "unclear", "other"] and "misheard" in sp[
        2
    ]["note"]


def test_room_send_goes_to_the_right_projects_once(room, monkeypatch):
    from wakesmith import recordings, sessions

    hh, ha = room
    texts = [
        "Across the room.",
        "Hey, Jarvis!",
        "Okay, computer.",
        "What's for dinner?",
        "Hey jarvis. Hey jarvis. Hey jarvis.",
    ]
    sid, _, _ = _room_session(room, monkeypatch, texts)
    sessions.split(sid, "cpu", log=lambda m: None)
    r = sessions.send(sid, busy=lambda p: False)
    assert r["errors"] == {} and r["claps"] == 3
    assert r["added"]["hh"]["pos"] == 4 and r["added"]["ha"]["pos"] == 1
    rooms = [x for x in sessions.segments(sid) if x["kind"] == "room"]
    n_room = sum(len(sessions.room_pieces(x["start"], x["end"])) for x in rooms)
    assert len(rooms) == 3 and n_room == 3
    # room sound and other speech to both; the other wake word never as not-the-phrase
    assert r["added"]["hh"]["neg"] == r["added"]["ha"]["neg"] == n_room + 1
    pos = recordings.listing(hh, "pos")["takes"]
    assert pos[0]["condition"] == "Study, across the room" and pos[0]["speaker"] == "Sam"
    assert [t["held"] for t in pos] == [False, False, False, True]  # every fourth kept aside
    neg = recordings.listing(ha, "neg")["takes"]
    assert {t["condition"] for t in neg} == {
        "Study, room sound",
        "Study, across the room, other speech",
    }
    rirs = sessions.house_rirs()
    assert len(rirs) == 3 and all(p.parent.parent.name == "study" for p in rirs)
    x, sr = sf.read(rirs[0])
    assert sr == 16000 and len(x) == int(1.02 * 16000) and abs(np.abs(x).max() - 0.9) < 1e-3
    assert np.argmax(np.abs(x)) < 0.04 * sr  # the clap starts right at the beginning
    # sending again, and after splitting again, adds nothing
    again = sessions.send(sid, busy=lambda p: False)
    assert again["claps"] == 0 and all(v == {"pos": 0, "neg": 0} for v in again["added"].values())
    monkeypatch.setattr(
        sessions,
        "_transcriber",
        lambda device, model="turbo", language="en": (
            lambda paths: [texts[i] for i in range(len(paths))]
        ),
    )
    sessions.split(sid, "cpu", log=lambda m: None)
    third = sessions.send(sid, busy=lambda p: False)
    assert all(v == {"pos": 0, "neg": 0} for v in third["added"].values())
    assert len(recordings.listing(hh, "pos")["takes"]) == 4
    marks = sessions.sent_marks(sid)
    assert any(t == ["hh"] for t in marks.values()) and any("claps" in t for t in marks.values())
    # a busy project is skipped and gets its takes next time
    sid2, _, _ = _room_session(room, monkeypatch, texts)
    sessions.split(sid2, "cpu", log=lambda m: None)
    r = sessions.send(sid2, busy=lambda p: p.name == "ha")
    assert (
        "job" in r["errors"]["ha"] and r["added"]["ha"]["pos"] == 0 and r["added"]["hh"]["pos"] == 4
    )
    assert sessions.send(sid2, busy=lambda p: False)["added"]["ha"]["pos"] == 1


def test_room_label_changes_and_choices(room, monkeypatch):
    from wakesmith import sessions

    texts = [
        "Across the room.",
        "Hey, Jarvis!",
        "Okay, computer.",
        "What's for dinner?",
        "Hey jarvis. Hey jarvis. Hey jarvis.",
    ]
    sid, _, _ = _room_session(room, monkeypatch, texts)
    sessions.split(sid, "cpu", log=lambda m: None)
    i = next(x["i"] for x in sessions.segments(sid) if x["label"] == "other")
    segs = sessions.set_labels(sid, {str(i): {"label": "skip", "spot": "doorway"}})
    assert (
        segs[i]["label"] == "skip" and segs[i]["spot"] == "doorway" and segs[i]["auto"] == "other"
    )
    with pytest.raises(ValueError):
        sessions.set_labels(sid, {str(i): {"label": "nope"}})
    project.create("hey jarvis", name="hh2")  # same phrase, newer: becomes the default
    ch = {c["name"]: c["default"] for c in sessions.choices()}
    assert ch == {"ha": True, "hh": False, "hh2": True}
    assert sessions.rooms() == ["Study"]


def test_house_rirs_setting_and_impulse_paths(room, tmp_path):
    from wakesmith import mww, sessions

    hh, _ = room
    assert hh.microwakeword.house_rirs is False
    mit = tmp_path / "mit"
    (mit / "sub").mkdir(parents=True)
    for i in range(6):
        sf.write(mit / "sub" / f"r{i}.wav", np.zeros(100), 16000)
    assert mww.impulse_paths(hh, str(mit)) == [str(mit)]
    hh.microwakeword.house_rirs = True
    hh.save()
    assert project.load("hh").microwakeword.house_rirs is True
    assert mww.impulse_paths(hh, str(mit)) == [str(mit)]  # no claps yet
    d = sessions.rooms_home() / "kitchen" / "claps"
    d.mkdir(parents=True)
    for i in range(2):
        sf.write(d / f"c{i}.wav", np.zeros(100), 16000)
    got = mww.impulse_paths(hh, str(mit))
    assert (
        got[0] == str(mit)
        and len(got) == 1 + 6
        and set(got[1:]) == {str(d / "c0.wav"), str(d / "c1.wav")}
    )
    from audiomentations.core.utils import find_audio_files_in_paths

    assert len(find_audio_files_in_paths(got)) == 12  # a list of folders and files works


def test_room_api_chunks_in_order(room, monkeypatch):
    from fastapi.testclient import TestClient

    from wakesmith import sessions
    from wakesmith.web.server import create_app

    started = []
    monkeypatch.setattr(sessions, "start_split", lambda sid, device="auto": started.append(sid))
    c = TestClient(create_app(), base_url="http://localhost")
    info = c.get("/api/sessions").json()
    assert {p["name"] for p in info["projects"] if p["default"]} == {"hh", "ha"}
    assert c.post("/api/sessions", json={"room": "", "projects": ["hh"]}).status_code == 400
    s = c.post(
        "/api/sessions", json={"room": "Den", "speakers": ["Sam", " "], "projects": ["hh", "ha"]}
    ).json()
    assert s["speakers"] == ["Sam"] and [x["project"] for x in s["phrases"]] == ["hh", "ha"]
    base = f"/api/sessions/{s['id']}"
    pieces = [np.full(1600, v, "<i2") for v in (100, 200, 300)]
    # out of order, one as WAV, one retried
    import io as _io

    w = _io.BytesIO()
    sf.write(w, pieces[1], 16000, format="WAV", subtype="PCM_16")
    assert c.post(f"{base}/chunk?seq=2", content=pieces[2].tobytes()).status_code == 200
    assert c.post(f"{base}/chunk?seq=1", content=w.getvalue()).status_code == 200
    assert c.post(f"{base}/chunk?seq=0", content=np.zeros(1600, "<i2").tobytes()).status_code == 200
    assert c.post(f"{base}/chunk?seq=0", content=pieces[0].tobytes()).status_code == 200
    assert c.post(f"{base}/chunk?seq=0", content=b"abc").status_code == 400
    r = c.post(f"{base}/stop").json()
    assert r["session"]["seconds"] == 0.3 and r["session"]["missing"] == [] and started == [s["id"]]
    a, _ = sf.read(sessions.folder(s["id"]) / "audio.wav", dtype="int16")
    assert list(a[::1600]) == [100, 200, 300]
    assert c.post(f"{base}/chunk?seq=3", content=pieces[0].tobytes()).status_code == 409
    seg = c.get(f"{base}/audio?start=0.1&end=0.2")
    assert sf.read(_io.BytesIO(seg.content), dtype="int16")[0].tolist() == [200] * 1600
    assert c.get("/api/sessions/nope").status_code == 400
    assert c.get("/api/sessions/20260101-000000-x").status_code == 404
    assert c.post(f"{base}/send").status_code == 409  # nothing split yet
    assert c.put("/api/projects/hh", json={"microwakeword": {"house_rirs": True}}).json()[
        "microwakeword"
    ]["house_rirs"]
    assert c.delete(base).status_code == 200 and c.get("/api/sessions").json()["sessions"] == []


def test_room_send_keeps_takes_for_a_project_that_trains_on_another_folder(
    room, monkeypatch, tmp_path
):
    from wakesmith import recordings, sessions

    hh, _ = room
    outside = tmp_path / "round_pos"
    outside.mkdir()
    hh.real.positives = str(outside)
    hh.save()
    texts = [
        "Across the room.",
        "Hey, Jarvis!",
        "Okay, computer.",
        "What's for dinner?",
        "Hey jarvis.",
    ]
    sid, _, _ = _room_session(room, monkeypatch, texts)
    sessions.split(sid, "cpu", log=lambda m: None)
    r = sessions.send(sid, busy=lambda p: False)
    assert "hh" not in r["errors"] and r["added"]["hh"]["pos"] >= 2
    assert r["notes"] == {}  # nothing to explain: the takes are used next to that folder
    hh2 = project.load("hh")
    assert hh2.real.positives == str(outside)  # never switched behind the user's back
    assert len(recordings.listing(hh2, "pos")["takes"]) == r["added"]["hh"]["pos"]
    # the sent takes are trained on next to that folder, and so is a take recorded on the page
    wav = sessions._wav(np.full(16000, 0.1, np.float32))
    recordings.add(hh2, "pos", wav)
    assert "rpos" in RealData(hh2).sources


def test_versions_of_one_wake_word(tmp_path, monkeypatch):
    import os
    import time

    from wakesmith import project as projects

    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path / "projects"))
    old = projects.create("hey jarvis", "hey_jarvis", label="Hey Jarvis")
    c3 = projects.create("hey jarvis", "hey_jarvis_c3", label="Hey Jarvis (c3)")
    e = projects.create(
        "hey jarvis", "hey_jarvis_c3e", label="Hey Jarvis (c3 + everyday sentences)"
    )
    other = projects.create("computer", "computer", label="Computer")
    for i, p in enumerate((old, c3, e, other)):  # created in this order
        (p.dir / "logs").mkdir()
        os.utime(p.dir / "logs", (time.time() - 100 + i, time.time() - 100 + i))
    projects.assign_versions(projects.all_projects())
    got = {p.name: p for p in projects.all_projects()}
    assert [got[n].version for n in ("hey_jarvis", "hey_jarvis_c3", "hey_jarvis_c3e")] == [1, 2, 3]
    assert (
        got["hey_jarvis_c3"].label == "Hey Jarvis" and got["hey_jarvis_c3"].note == ""
    )  # a bare code says nothing
    assert got["hey_jarvis_c3e"].note == "c3 + everyday sentences"
    assert (
        got["hey_jarvis_c3e"].title == "Hey Jarvis v3"
        and got["hey_jarvis_c3e"].display == "Hey Jarvis"
    )
    assert got["computer"].version == 1
    new = projects.create("hey jarvis", "hey_jarvis_c4")  # the next version, with the word's name
    assert new.version == 4 and new.title == "Hey Jarvis v4"
    projects.assign_versions(projects.all_projects())  # numbers never move once given
    assert projects.load("hey_jarvis_c3e").version == 3


def test_new_version_carries_recipe_clips_and_device_labels(tmp_path, monkeypatch):
    import json

    from wakesmith import project as projects
    from wakesmith import rounds, versions

    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path / "projects"))
    p = projects.create("hey jarvis", "hey_jarvis", label="Hey Jarvis")
    p.microwakeword.candidates = 6
    p.save()
    (p.dir / "clips" / "positive").mkdir(parents=True)
    p.set_state(generate={"status": "done"}, real={"status": "done"})
    wav = p.dir / "devices" / "inbox" / "kitchen" / "a.wav"
    wav.parent.mkdir(parents=True)
    wav.write_bytes(b"RIFF")
    clips = [
        {
            "id": "kitchen/a.wav",
            "device": "kitchen",
            "at": f"2026-10-2{d}T10:00:00Z",
            "model": "hey_jarvis",
            "score": 0.6,
            "text": "no",
            "suggestion": "not_wake",
            "label": "not_wake",
        }
        for d in (3, 4, 5, 6)
    ]
    (p.dir / "devices" / "clips.json").write_text(json.dumps(clips))

    q = versions.create(projects.load("hey_jarvis"))
    assert q.name == "hey_jarvis_v2" and q.version == 2 and q.title == "Hey Jarvis v2"
    assert q.microwakeword.candidates == 6 and q.note == "v1 + newer device clips"
    assert (q.dir / "clips").resolve() == (p.dir / "clips").resolve()  # shared, not made again
    assert q.state() == {"generate": {"status": "done"}}  # generate stays done; real runs again
    assert [c["label"] for c in rounds.load(q)] == ["not_wake"] * 4
    assert (q.dir / "devices" / "inbox" / "kitchen" / "a.wav").stat().st_ino == wav.stat().st_ino
    assert versions.hold_from(clips) == "2026-10-25T10:00"  # the newest day
    assert versions.hold_from(clips[:1]) == "2026-10-22T10:00"  # one clip: it is held out


def test_server_refuses_cross_site_writes_and_foreign_hosts(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from wakesmith.web.server import create_app

    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path / "projects"))
    c = TestClient(create_app(), base_url="http://localhost")
    assert c.get("/api/projects").status_code == 200
    evil = c.post(
        "/api/projects", json={"phrase": "hey jarvis"}, headers={"Origin": "https://evil.example"}
    )
    assert evil.status_code == 403  # a web page elsewhere can't make or change wake words
    assert (
        c.get("/api/projects", headers={"Host": "rebound.example"}).status_code == 403
    )  # DNS rebinding
    same = c.post(
        "/api/projects", json={"phrase": "hey jarvis"}, headers={"Origin": "http://localhost"}
    )
    assert same.status_code != 403


def test_verdict_drops_when_there_is_nothing_to_compare(tmp_path, monkeypatch):
    import json

    from wakesmith import project as projects
    from wakesmith import verdict

    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path / "projects"))
    p = projects.create("hey jarvis")
    f = p.path("results", "verdict-mww.json")
    f.parent.mkdir()
    f.write_text(json.dumps({"better": True, "sentence": "old"}))
    assert (
        verdict.compare(p, "mww") is None and not f.exists()
    )  # no live version: no stale "better"


def test_new_clips_since_a_version_count_auto_labeled_ones():
    from wakesmith import rounds

    clips = [
        {"at": "2026-10-25T10:00:00Z", "label": "not_wake"},  # labeled by itself on sync: still new
        {"at": "2026-10-25T11:00:00Z", "label": None},
        {"at": "2026-10-25T12:00:00Z", "label": "skip"},  # left out: not counted
        {"at": "2026-10-20T10:00:00Z", "label": "not_wake"},  # older than the version
    ]
    made = 1792540800  # 2026-10-21 00:00 UTC
    assert rounds.newer_than(clips, made) == 2


def test_record_a_room_takes_one_version_per_word(tmp_path, monkeypatch):
    from wakesmith import project as projects
    from wakesmith import sessions

    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path / "projects"))
    monkeypatch.setenv("WAKESMITH_HOME", str(tmp_path / "home"))
    projects.create("hey jarvis", "hj1")
    projects.create("hey jarvis", "hj2")
    with pytest.raises(ValueError, match="one version"):
        sessions.create("Kitchen", ["Sam"], ["hj1", "hj2"])


def test_verdict_never_calls_fewer_catches_better():
    from wakesmith.verdict import is_better

    assert not is_better({"caught": 18, "false": 2}, {"caught": 21, "false": 9})  # a trade-off
    assert is_better({"caught": 21, "false": 2}, {"caught": 21, "false": 9})
    assert is_better({"caught": 22, "false": 9}, {"caught": 21, "false": 9})
    assert not is_better({"caught": 21, "false": 9}, {"caught": 21, "false": 9}) and not is_better(
        None, {}
    )


def test_versions_survive_deleting_the_one_they_came_from(tmp_path, monkeypatch):
    from wakesmith import project as projects
    from wakesmith import recordings, versions
    from wakesmith.real import RealData

    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path / "projects"))
    (tmp_path / "mine").mkdir()
    (tmp_path / "mine" / "a.wav").write_bytes(_wav(1.0))
    p = projects.create("hey jarvis", "hey_jarvis", label="Hey Jarvis")
    p.real.positives = str(tmp_path / "mine")
    p.save()
    (p.dir / "clips" / "positive_train").mkdir(parents=True)
    recordings.add(p, "pos", _wav(1.0))  # a take recorded on the page, next to the folder
    (p.dir / "devices").mkdir()
    (p.dir / "devices" / "clips.json").write_text("[]")  # a sync that found nothing: no inbox
    q = versions.create(projects.load("hey_jarvis"))
    assert set(RealData(q).sources) == set(RealData(p).sources) == {"pos", "rpos"}
    assert not list((tmp_path / "projects").glob(".*"))  # no half-made folder left
    with pytest.raises(ValueError, match="use"):  # its synthetic voices are q's too
        projects.delete(projects.load("hey_jarvis"))
    assert (q.dir / "recordings" / "pos" / "take_001.wav").exists()  # q's own copy of the take
    assert not list(q.dir.glob("*.tmp"))  # project.yaml is replaced in one step


def test_runs_started_outside_the_page_count_as_busy(proj):
    import os

    from wakesmith import jobs

    proj.set_state(**{"mww-train": {"status": "running", "started": 1.0, "pid": os.getpid()}})
    assert proj.name in [p.name for p in jobs.running_all()]
    proj.set_state(**{"mww-train": {"status": "running", "started": 1.0, "pid": 999999999}})
    assert proj.name not in [p.name for p in jobs.running_all()]  # its process is gone


def test_a_new_version_owns_its_label_files_and_its_creation_time(tmp_path, monkeypatch):
    import os
    import time

    from wakesmith import project as projects
    from wakesmith import recordings, rounds, versions

    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path / "projects"))
    p = projects.create("hey jarvis", "hey_jarvis", label="Hey Jarvis")
    recordings.add(p, "pos", _wav(1.0))
    old = time.time() - 30 * 86400
    os.utime(p.dir / "recordings", (old, old))  # copied folders keep their dates
    q = versions.create(projects.load("hey_jarvis"))
    recordings.set_heldout(q, "pos", "take_001.wav", True)
    assert "take_001.wav" not in (p.dir / "recordings" / "pos" / "HELDOUT.txt").read_text()
    assert abs(projects.created(q) - time.time()) < 60
    clips = [{"at": "2026-01-01T00:00:00Z", "label": "not_wake"}]  # carried over, not new
    assert rounds.newer_than(clips, projects.created(q)) == 0


def test_half_made_versions_are_never_listed(tmp_path, monkeypatch):
    from wakesmith import archive
    from wakesmith import project as projects

    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path / "projects"))
    projects.create("hey jarvis")
    half = tmp_path / "projects" / ".hey_jarvis_v2.making"
    half.mkdir()
    (half / "project.yaml").write_text("name: hey_jarvis_v2\nphrase: hey jarvis\n")
    assert [p.name for p in projects.all_projects()] == ["hey_jarvis"]
    assert all(".making" not in str(f) for f in projects.project_files())
    assert all(".making" not in k for k in archive.links())


def test_take_uploads_are_capped_without_trusting_content_length(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from wakesmith.web.server import TAKE_MAX_BYTES, create_app

    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path / "projects"))
    c = TestClient(create_app(), base_url="http://localhost")
    name = c.post("/api/projects", json={"phrase": "hey jarvis"}).json()["name"]
    chunks = iter([b"x" * 1_000_000] * (TAKE_MAX_BYTES // 1_000_000 + 2))  # sent chunked, no length
    r = c.post(f"/api/projects/{name}/recordings/pos", content=chunks)
    assert r.status_code == 413


def test_a_new_take_never_writes_into_an_existing_file(proj):
    from wakesmith import recordings

    d = recordings.folder(proj, "pos")
    d.mkdir(parents=True)
    (d / "take_001.wav").write_bytes(b"another version's audio")  # on disk, no row in labels.csv
    r = recordings.add(proj, "pos", _wav(1.0))
    assert (
        r["file"] == "take_002.wav"
        and (d / "take_001.wav").read_bytes() == b"another version's audio"
    )


def test_wake_words_table_combines_common_voice_takes_with_own_false_trigger_set(
    tmp_path, monkeypatch
):
    import json

    from fastapi.testclient import TestClient

    from wakesmith import project as projects
    from wakesmith.web.server import create_app

    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path / "projects"))
    p = projects.create("computer")
    p.openwakeword.enabled, p.microwakeword.enabled = True, False
    p.save()
    (p.dir / "results").mkdir()
    t = lambda c, f: [{"threshold": 0.5, "caught": c, "false": f}]
    own = {"positives": 0, "negatives": 300, "table": t(0, 2)}  # no takes of the phrase of its own
    cv = {"positives": 510, "negatives": 0, "table": t(390, 0)}
    (p.dir / "results" / "oww.json").write_text(json.dumps(own))
    (p.dir / "results" / "oww-cv.json").write_text(json.dumps(cv))
    c = TestClient(create_app(), base_url="http://localhost")
    b = c.get("/api/projects").json()[0]["brief"]["oww"]
    assert (b["caught"], b["positives"], b["false"], b["negatives"]) == (390, 510, 2, 300)


def test_everyday_terms_use_common_voice_takes_when_there_are_none_of_its_own(
    tmp_path, monkeypatch
):
    import json

    from wakesmith import project as projects
    from wakesmith import verdict

    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path / "projects"))
    p = projects.create("computer")
    (p.dir / "results").mkdir()
    (p.dir / "results" / "oww.json").write_text(
        json.dumps({"pos": {}, "neg": {"n1": 0.9, "n2": 0.1}})
    )
    (p.dir / "results" / "oww-cv.json").write_text(
        json.dumps({"pos": {"c1": 0.9, "c2": 0.2}, "neg": {}})
    )
    row = next(r for r in verdict.everyday(p, "oww")["rows"] if r["cutoff"] == 0.5)
    assert (row["caught"], row["false"]) == (1, 1)


# --- languages (German is experimental, off by default) ---------------------------------------------


def test_german_is_off_until_turned_on(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from wakesmith import languages
    from wakesmith.cli import app

    monkeypatch.setenv("WAKESMITH_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path / "projects"))
    monkeypatch.delenv("WAKESMITH_EXPERIMENTAL_LANGUAGES", raising=False)
    assert [x.code for x in languages.available()] == ["en"]
    r = CliRunner().invoke(app, ["new", "hallo computer", "--language", "de"])
    assert r.exit_code != 0 and "experimental" in r.output
    languages.set_experimental(True)
    assert [x.code for x in languages.available()] == ["en", "de"]
    assert json.loads((tmp_path / "data" / "settings.json").read_text())["languages"] == {
        "experimental": True
    }
    languages.set_experimental(False)
    monkeypatch.setenv("WAKESMITH_EXPERIMENTAL_LANGUAGES", "1")
    assert "de" in [x.code for x in languages.available()]


def test_setup_leaves_the_german_voices_for_later():
    from wakesmith import assets

    german = [a for a in assets.ASSETS.values() if a.name.startswith("tts-de-")]
    assert german and all(a.optional for a in german)
    assert not set(assets.required()) & set(german)
    assert all(a.sha256 for a in german)  # pinned like every single-file download


def test_a_german_word_uses_german_voices_and_names(tmp_path, monkeypatch):
    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path / "projects"))
    p = project.create("Grüß Gott", language="de")
    assert p.name == "gruess_gott__de" and p.language == "de"
    assert p.tts_models and all(m.startswith("de-") for m in p.tts_models)
    assert project.load(p.name).language == "de"
    note = export.license_note(p)
    assert "de_DE-thorsten-high: CC0" in note and "CMU" not in note
    assert project.create("hey jarvis").language == "en"  # English stays the default
    # the same word in two languages: two wake words, not two versions of one
    en, de = project.create("computer"), project.create("computer", language="de")
    assert (en.name, en.version, de.name, de.version) == ("computer", 0, "computer__de", 0)
    assert de.family != en.family
    assert project.create("hey de").family != project.create("hey", language="de").family


@needs_espeak
def test_german_sound_alikes_keep_umlauts_and_never_say_the_phrase():
    from wakesmith import phrases

    out = phrases.suggest("hallo küche", 40, language="de")
    assert out and "hallo küche" not in out
    assert "küche" in out  # the last word alone
    target = phrases._ipa_key("hallo küche", "de")
    assert all(phrases._ipa_key(s, "de") != target for s in out)
    assert any(c in "".join(out) for c in "äöü")  # German vowels in the variants


def test_a_german_word_gets_german_everyday_sentences_and_no_english_common_voice(
    tmp_path, monkeypatch
):
    from wakesmith import commonvoice, conversational

    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path / "projects"))
    assert conversational.says_phrase("Grüß Gott, wie geht's?", ["grüß gott"])
    assert not conversational.says_phrase("Gruß an Gott", ["grüß gott"])
    ask = conversational.messages("hallo computer", 5, [], [], "de")[-1]["content"]
    assert "in German" in ask
    assert "German" not in conversational.messages("hey jarvis", 5, [], [])[-1]["content"]
    p = project.create("hallo computer", language="de")
    with pytest.raises(RuntimeError, match="English"):
        commonvoice.prepare(p)


def test_german_endpoints_follow_the_switch(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from wakesmith.web.server import create_app

    monkeypatch.setenv("WAKESMITH_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path / "projects"))
    monkeypatch.delenv("WAKESMITH_EXPERIMENTAL_LANGUAGES", raising=False)
    c = TestClient(create_app(), base_url="http://localhost")
    # off: no German sound-alikes, and no German voices downloaded for a preview
    assert c.get("/api/suggest", params={"phrase": "hallo", "language": "de"}).status_code == 400
    assert c.get("/api/preview", params={"text": "hallo", "language": "de"}).status_code == 400
    assert c.post("/api/languages", json=[True]).status_code == 400
    assert c.post("/api/languages", content=b"not json").status_code == 400
    assert c.post("/api/languages", json={"experimental": "yes"}).status_code == 400
    assert c.post("/api/languages", json={"experimental": True}).json()["experimental"] is True
    # Common Voice can't be switched on for a German word (the set here is English)
    p = project.create("hallo computer", language="de", negatives=["hallo"])
    p.commonvoice.enabled = True  # as if project.yaml had been edited by hand
    assert not p.uses_commonvoice
    r = c.put(f"/api/projects/{p.name}", json={"commonvoice": {"enabled": True}})
    assert r.status_code == 400


def test_switch_values_that_mean_off(monkeypatch, tmp_path):
    from wakesmith import languages

    monkeypatch.setenv("WAKESMITH_HOME", str(tmp_path))
    for v in ("0", "false", "No", "off", ""):
        monkeypatch.setenv("WAKESMITH_EXPERIMENTAL_LANGUAGES", v)
        assert not languages.experimental_on(), v
    monkeypatch.setenv("WAKESMITH_EXPERIMENTAL_LANGUAGES", "1")
    assert languages.experimental_on()


def test_overlapping_ipv4_downloads_put_the_real_lookup_back():
    import socket

    from wakesmith import assets

    real = socket.getaddrinfo
    with assets._ipv4_only():
        with assets._ipv4_only():
            assert socket.getaddrinfo is not real
        assert socket.getaddrinfo is not real  # the other download still runs
    assert socket.getaddrinfo is real


def test_a_download_larger_than_announced_stops_and_leaves_nothing(tmp_path, monkeypatch):
    import io
    import urllib.request

    from wakesmith import assets

    class Reply(io.BytesIO):
        status, headers = 200, {"Content-Length": "0"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: Reply(b"x" * (3 << 20)))
    out = tmp_path / "voice.onnx.part"
    with pytest.raises(RuntimeError, match="larger than expected"):
        assets._fetch(object(), out, 0, 1000, False, "voice")
    assert not out.exists()


def test_german_text_details(tmp_path, monkeypatch):
    import unicodedata

    from wakesmith import conversational, eta, sessions

    nfd = unicodedata.normalize("NFD", "Ich putze die Küche")
    assert conversational.says_phrase(nfd, ["küche"])
    spots = [n for n, _ in sessions.SPOTS]
    assert sessions.spot_of(sessions._words("jetzt an der Tür"), spots, "de") == "doorway"
    assert sessions.spot_of(sessions._words("across the room"), spots, "de") == "across the room"
    assert (
        sessions.spot_of(sessions._words("nah, again"), spots) is None
    )  # German words: German only
    monkeypatch.setenv("WAKESMITH_HOME", str(tmp_path))
    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path / "projects"))
    en = project.create("hey jarvis", negatives=["jarvis"])
    de = project.create("hallo computer", language="de", negatives=["hallo"])
    eta.record(de, "generate", 10.0, {})
    t = eta._load()
    assert (
        "generate:de" in t and "generate" not in t
    )  # German speech doesn't change English's estimate
    assert eta.estimates(en)["generate"] != eta.estimates(de)["generate"]


def test_a_preview_does_not_wait_for_a_download_in_another_process(tmp_path, monkeypatch):
    import fcntl

    from wakesmith import assets

    monkeypatch.setenv("WAKESMITH_HOME", str(tmp_path))
    a = assets.Asset("voice", "tts/voice.onnx", url="https://example.invalid/voice.onnx", size=10)
    a.local.parent.mkdir(parents=True)
    with open(a.local.with_name(".voice.onnx.lock"), "w") as held:  # as another process would
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(RuntimeError, match="being downloaded"):
            assets.fetch(a, progress=False, wait=False)


def test_a_hand_edited_language_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path / "projects"))
    p = project.create("hallo computer", language="de", negatives=["hallo"])
    f = p.path("project.yaml")
    f.write_text(f.read_text().replace("language: de", "language: ../../x"))
    with pytest.raises(ValueError, match="unknown language"):
        project.load(p.name)
    assert project.all_projects() == []  # left out of every listing instead of naming folders
    from fastapi.testclient import TestClient

    from wakesmith.web.server import create_app

    r = TestClient(create_app(), base_url="http://localhost").get(f"/api/projects/{p.name}")
    assert r.status_code == 409 and "unknown language" in r.json()["detail"]


def test_a_folder_of_another_word_does_not_block_a_new_one(tmp_path, monkeypatch):
    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path / "projects"))
    old = project.create(
        "hallo computer", name="hallo_computer", language="de"
    )  # an early German one
    new = project.create("hallo computer")
    assert old.family != new.family and new.name == "hallo_computer_2" and new.version == 0


def test_setup_stops_early_on_a_python_it_cannot_use(monkeypatch):
    import sys

    from wakesmith import setup_env as se

    monkeypatch.setattr(sys, "version_info", (3, 14, 0))
    assert "--python 3.12" in se.python_problem()
    monkeypatch.setattr(sys, "version_info", (3, 12, 7))
    assert se.python_problem() is None
    monkeypatch.setattr(sys, "platform", "darwin")
    assert se.install_hint(["espeak-ng"]) == "brew install espeak-ng"
    monkeypatch.setattr(sys, "platform", "linux")
    assert se.install_hint(["espeak-ng", "git"]) == "sudo apt install espeak-ng git"
