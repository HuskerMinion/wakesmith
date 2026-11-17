"""The microWakeWord training recipe: masking, augmentation, real-clip trimming and weights, the
FLEURS background download, and the note that device-clip tests undercount misses."""

import io
import json
import tarfile

import numpy as np
import pytest
import soundfile as sf

from wakesmith import assets, export, mww, project, verdict


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.setenv("WAKESMITH_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path / "projects"))
    return project.create("hey jarvis", negatives=["jarvis"])


def test_new_projects_get_the_recipe_and_old_settings_load_unchanged(proj):
    m, r = proj.microwakeword, proj.real
    assert (
        m.spec_augment
        and m.speech_background
        and (m.min_trailing_s, m.max_trailing_s) == (0.10, 0.21)
    )
    assert (r.copies, r.negative_copies, r.sampling_weight) == (3, 3, 2.0)
    assert (r.negative_weight, r.negative_penalty, r.trim_silence) == (1.5, 3.0, True)
    proj.save()
    y = (proj.dir / "project.yaml").read_text()
    # a project saved by an older wakesmith: its own values stay, the new fields take the defaults
    y = y.replace("copies: 3\n", "copies: 10\n", 1).replace(
        "sampling_weight: 2.0", "sampling_weight: 0.0"
    )
    y = "\n".join(line for line in y.splitlines() if "spec_augment" not in line)
    (proj.dir / "project.yaml").write_text(y)
    q = project.load(proj.dir)
    assert q.real.copies == 10 and q.real.sampling_weight == 0.0 and q.microwakeword.spec_augment


def test_config_masks_and_real_negative_weights(proj, monkeypatch):
    monkeypatch.setattr(assets, "fetch", lambda a, **k: a.local)
    proj.path("mww", "features", "real_negative").mkdir(parents=True)
    proj.real.negative_weight, proj.real.negative_penalty = 1.25, 4.0
    proj.microwakeword.finetune_steps = 5000
    cfg = __import__("yaml").safe_load(mww.write_config(proj).read_text())
    for k, v in (
        ("time_mask_max_size", 5),
        ("time_mask_count", 2),
        ("freq_mask_max_size", 5),
        ("freq_mask_count", 2),
    ):
        assert cfg[k] == [v, v]  # both phases
    rn = next(f for f in cfg["features"] if f["features_dir"] == "features/real_negative")
    assert (rn["sampling_weight"], rn["penalty_weight"]) == (1.25, 4.0)
    proj.microwakeword.spec_augment = False
    cfg = __import__("yaml").safe_load(mww.write_config(proj).read_text())
    assert cfg["time_mask_count"] == [0, 0] and cfg["freq_mask_max_size"] == [0, 0]


def test_sampler_masks_blank_stretches_and_bands():
    from wakesmith.mww_torch import Sampler

    s = Sampler.__new__(Sampler)
    s.length, s.rng = 49, np.random.default_rng(0)
    s.masks = [5, 2, 5, 2]
    x = np.ones((200, 49, 40), np.float32)
    s._mask(x)
    zero_frames = (x == 0).all(axis=2).sum(axis=1)
    zero_bins = (x == 0).all(axis=1).sum(axis=1)
    assert zero_frames.max() <= 8 and zero_bins.max() <= 8  # 2 x at most 4 (uniform(0, 5) floored)
    assert zero_frames.mean() > 1 and zero_bins.mean() > 1
    s.masks = [0, 0, 0, 0]
    y = np.ones((4, 49, 40), np.float32)
    s._mask(y)
    assert (y == 1).all()


def test_sampler_reads_masks_from_the_config():
    from wakesmith.mww_torch import Sampler

    cfg = {"features": [], "spectrogram_length": 49, "time_mask_max_size": [5, 5],
           "time_mask_count": [2, 2], "freq_mask_max_size": [3, 3], "freq_mask_count": [1, 1]}  # fmt: skip
    assert Sampler(cfg, 1).masks == [5, 2, 3, 1]
    assert Sampler({"features": [], "spectrogram_length": 49}, 1).masks == [0, 0, 0, 0]


def test_augmentation_settings_and_trailing_quiet():
    a = mww.augmentation_settings(["/n", "/s"], ["/r"], (0.1, 0.21))
    assert a["augmentation_probabilities"]["GainTransition"] == 0.2
    assert a["background_paths"] == ["/n", "/s"] and a["background_max_snr_db"] == 15
    assert (a["min_jitter_s"], a["max_jitter_s"]) == (0.1, 0.21)
    pytest.importorskip("microwakeword.audio.augmentation")
    from microwakeword.audio.augmentation import Augmentation

    Augmentation(**{**a, "background_paths": [], "impulse_paths": []})  # microWakeWord takes them


def test_build_features_refuses_bad_trailing_and_adds_speech(proj, monkeypatch):
    got = []
    monkeypatch.setattr(assets, "fetch", lambda a, **k: got.append(a.name) or a.local)
    proj.microwakeword.min_trailing_s, proj.microwakeword.max_trailing_s = 0.3, 0.2
    with pytest.raises(ValueError, match="min_trailing_s"):
        mww.build_features(proj, 1, log=lambda m: None)
    assert got == []  # refused before any download

    class Stop(Exception):
        pass

    def stop(*a, **k):
        raise Stop

    proj.microwakeword.min_trailing_s = 0.1
    monkeypatch.setattr(mww, "feature_jobs", stop)
    with pytest.raises(Stop):
        mww.build_features(proj, 1, log=lambda m: None)
    assert "fleurs-en-speech" in got and "musan-noise" in got
    got.clear()
    proj.microwakeword.speech_background = False
    with pytest.raises(Stop):
        mww.build_features(proj, 1, log=lambda m: None)
    assert "fleurs-en-speech" not in got


def test_trim_silence_cuts_quiet_and_keeps_mostly_quiet_clips():
    pytest.importorskip("webrtcvad")
    rng = np.random.default_rng(0)
    t = np.arange(16000) / 16000
    voice = (0.3 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(
        np.float32
    )
    voice += 0.05 * rng.standard_normal(16000).astype(np.float32)
    a = np.concatenate([np.zeros(16000, np.float32), voice, np.zeros(16000, np.float32)])
    out = mww.trim_silence(a)
    assert len(out) < len(a)
    quiet = np.zeros(32000, np.float32)
    assert mww.trim_silence(quiet) is quiet  # under 0.2 s of voice left: unchanged


def test_shard_trims_the_clips_it_is_given_in_training_only(tmp_path, monkeypatch):
    pytest.importorskip("mmap_ninja")
    pytest.importorskip("microwakeword.audio.audio_utils")
    real, synth = tmp_path / "r.wav", tmp_path / "s.wav"
    sf.write(real, np.zeros(44100, np.float32), 44100)  # a real clip: any rate
    sf.write(synth, np.zeros(16000, np.float32), 16000)
    seen = []
    monkeypatch.setattr(mww, "trim_silence", lambda a: seen.append(len(a)) or a)
    aug = type("A", (), {"augment_clip": staticmethod(lambda a: np.zeros(48000, np.float32))})()
    monkeypatch.setattr(mww, "_AUG", aug)
    trim = frozenset({str(real)})
    for name, split in (("positive", "training"), ("real_positive", "validation")):
        mww._shard((name, split, [str(real), str(synth)], 1, 1, str(tmp_path / name / split), trim))
    assert seen == [16000]  # the real clip, converted, in training; not the synthetic one


def test_build_features_trims_real_positives_mixed_into_the_synthetic_set(proj, monkeypatch):
    monkeypatch.setattr(assets, "fetch", lambda a, **k: a.local)
    proj.real.sampling_weight = 0.0
    proj.path("real", "train").mkdir(parents=True)
    r = proj.path("real", "train", "pos__a.wav")
    sf.write(r, np.zeros(16000, np.float32), 16000)
    tasks = []

    class Pool:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def imap_unordered(self, fn, ts):
            tasks.extend(ts)
            return iter(())

    monkeypatch.setattr(mww, "get_context", lambda *a: type("C", (), {"Pool": Pool})())
    mww.build_features(proj, 1, log=lambda m: None)
    pos = [t for t in tasks if t[0] == "positive" and t[1] == "training"]
    assert pos and all(str(r) in t[6] for t in pos)
    assert json.loads(proj.path("mww", "recipe.json").read_text()) == {"speech_background": True}


def test_fleurs_asset_unpacks_a_tar_and_refuses_escapes(tmp_path, monkeypatch):
    monkeypatch.setenv("WAKESMITH_HOME", str(tmp_path / "data"))

    def tar_bytes(members):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as t:
            for name, data in members:
                i = tarfile.TarInfo(name)
                i.size = len(data)
                t.addfile(i, io.BytesIO(data))
        return buf.getvalue()

    good = tar_bytes([("train/a.wav", b"RIFF")])
    a = assets.Asset("t", "noise/t", url="http://x/t.tar.gz", untar=True)
    monkeypatch.setattr(
        assets, "_download", lambda url, out, size, progress, label: out.write_bytes(good)
    )
    assert assets.fetch(a, progress=False) == a.local
    assert (a.local / "train" / "a.wav").read_bytes() == b"RIFF" and a.present()
    assert not list(a.local.parent.glob("*.tar.gz*")) and not list(
        a.local.parent.glob("*.unpacking")
    )

    bad = tar_bytes([("../evil.wav", b"x")])
    b = assets.Asset("u", "noise/u", url="http://x/u.tar.gz", untar=True)
    monkeypatch.setattr(
        assets, "_download", lambda url, out, size, progress, label: out.write_bytes(bad)
    )
    with pytest.raises((tarfile.TarError, RuntimeError)):  # Python's filter, or ours without one
        assets.fetch(b, progress=False)
    assert not (tmp_path / "data" / "evil.wav").exists() and not b.present()
    with tarfile.open(fileobj=io.BytesIO(bad), mode="r:gz") as t:
        monkeypatch.setattr(t, "extractall", _no_filter(t.extractall))
        with pytest.raises(RuntimeError, match="outside"):
            assets._safe_extract(t, tmp_path / "x")


def _no_filter(real):
    """extractall as on a Python without extraction filters."""

    def f(path, members=None, **kw):
        if "filter" in kw:
            raise TypeError("unexpected keyword argument 'filter'")
        import warnings

        with warnings.catch_warnings():  # Python 3.12+ warns when no filter is given
            warnings.simplefilter("ignore", DeprecationWarning)
            return real(path, members=members)

    return f


def test_fleurs_is_a_known_asset_and_credited(proj):
    a = assets.ASSETS["fleurs-en-speech"]
    assert a.untar and a.license == "CC BY 4.0" and len(a.sha256) == 64 and not a.optional
    assert "FLEURS" in export.license_note(proj)  # nothing built yet: the setting
    proj.path("mww", "features").mkdir(parents=True)
    assert "FLEURS" not in export.license_note(proj)  # built by an older wakesmith: not used
    proj.path("mww", "recipe.json").write_text('{"speech_background": true}')
    __import__("shutil").rmtree(
        proj.path("mww", "features")
    )  # archived: features gone, record kept
    assert "FLEURS" in export.license_note(proj)
    proj.microwakeword.speech_background = False  # changed since: the model still used it
    assert "FLEURS" in export.license_note(proj)


def test_everyday_counts_device_takes(proj):
    proj.path("results").mkdir(parents=True, exist_ok=True)
    pos = {"dpos__desk__a.wav": 0.99, "dpos__desk__b.wav": 0.2, "rpos__take_001.wav": 0.95}
    proj.path("results", "mww.json").write_text(json.dumps({"pos": pos, "neg": {"x.wav": 0.1}}))
    d = verdict.everyday(proj, "mww")
    assert (d["positives"], d["device_takes"]) == (3, 2)


def test_tar_fallback_normalizes_permissions(tmp_path, monkeypatch):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        i = tarfile.TarInfo("train/a.wav")
        i.size, i.mode = 4, 0o4777
        t.addfile(i, io.BytesIO(b"RIFF"))
    buf.seek(0)
    with tarfile.open(fileobj=buf, mode="r:gz") as t:
        monkeypatch.setattr(t, "extractall", _no_filter(t.extractall))
        assets._safe_extract(t, tmp_path / "x")
    assert (tmp_path / "x" / "train" / "a.wav").stat().st_mode & 0o7777 == 0o644


def test_old_projects_keep_their_openwakeword_copies(proj):
    proj.save()
    y = (proj.dir / "project.yaml").read_text()
    y = "\n".join(
        line
        for line in y.splitlines()
        if "real_copies" not in line and "real_negative_copies" not in line
    )
    y = y.replace("  copies: 3\n", "  copies: 20\n", 1).replace(
        "  negative_copies: 3\n", "  negative_copies: 7\n", 1
    )
    (proj.dir / "project.yaml").write_text(y + "\n")
    q = project.load(proj.dir)
    assert (q.openwakeword.real_copies, q.openwakeword.real_negative_copies) == (20, 7)
    assert (q.real.copies, q.real.negative_copies) == (20, 7)
    q.openwakeword.real_copies = 4
    q.save()
    assert project.load(proj.dir).openwakeword.real_copies == 4  # saved now: its own


def test_the_model_keeps_its_recipe_record(proj):
    proj.path("mww", "trained").mkdir(parents=True)
    proj.path("mww", "trained", "recipe.json").write_text('{"speech_background": true}')
    proj.microwakeword.speech_background = False
    assert export.speech_background_used(proj)  # features archived, setting changed: the model's
    proj.path("mww", "trained", "recipe.json").unlink()
    assert not export.speech_background_used(proj)  # a model from an older wakesmith


def test_import_limit_counts_clips_that_came_in(proj, tmp_path):
    from wakesmith import rounds

    src = tmp_path / "in" / "desk"
    src.mkdir(parents=True)
    for i in range(3):
        (src / f"a{i}.wav").write_bytes(b"RIFFjunk")  # oldest: broken
        __import__("os").utime(src / f"a{i}.wav", (1000 + i, 1000 + i))
    for i in range(3):
        sf.write(src / f"b{i}.wav", np.zeros(1600, np.float32), 16000, subtype="PCM_16")
        __import__("os").utime(src / f"b{i}.wav", (2000 + i, 2000 + i))
    r = rounds.import_folder(proj, tmp_path / "in", limit=2)
    assert (r["added"], r["unreadable"], r["left"]) == (2, 3, 1)
    assert not list(proj.dir.rglob("a*.wav"))  # nothing half copied left behind


def test_sync_reports_unreadable_clips():
    from wakesmith import clipsync

    r = {"result": "ok", "new": 2, "unreadable": 1, "transcribed": 0, "left": 0, "to_check": 0}
    assert "1 unreadable" in clipsync.describe(r)


def test_combined_model_license_note_follows_its_candidates(proj):
    assert "FLEURS" in export.license_note(proj, {"speech_background": True})
    proj.path("mww", "trained").mkdir(parents=True)
    proj.path("mww", "trained", "recipe.json").write_text('{"speech_background": true}')
    assert "FLEURS" not in export.license_note(proj, {"speech_background": False})
    assert "FLEURS" in export.license_note(proj, {"tensor_arena_size": 1})  # older combined info


def test_a_model_without_a_record_never_takes_the_newer_features_record(proj):
    from wakesmith import mww as m

    m.model_path(proj).parent.mkdir(parents=True)
    m.model_path(proj).write_bytes(b"tflite")  # a model from an older wakesmith
    proj.path("mww", "recipe.json").write_text('{"speech_background": true}')  # newer features
    assert not export.speech_background_used(proj)
