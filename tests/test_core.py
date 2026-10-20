"""Fast tests: no training, no downloads beyond the small lexicon (skipped when espeak-ng is missing)."""

from __future__ import annotations

import json
import pickle
import shutil

import numpy as np
import pytest
import soundfile as sf

from wakesmith import export, project, real
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


def test_export_manifest(proj, monkeypatch):
    from wakesmith import mww

    model = mww.model_path(proj)
    model.parent.mkdir(parents=True)
    model.write_bytes(b"tflite")
    proj.microwakeword.cutoff = 0.93
    files = export.export(proj)
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
