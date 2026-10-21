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
    with pytest.raises(FileExistsError):
        project.create("hey jarvis")


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


def test_recorder_wont_mix_into_a_chosen_folder(proj, tmp_path):
    from wakesmith import recordings

    (tmp_path / "mine").mkdir()
    proj.real.positives = str(tmp_path / "mine")
    proj.save()
    with pytest.raises(PermissionError):
        recordings.add(proj, "pos", _wav(1.0))
    recordings.use(proj, "pos")
    assert recordings.add(proj, "pos", _wav(1.0))["file"] == "take_001.wav"


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
