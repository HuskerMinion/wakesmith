import numpy as np
import soundfile as sf

from wakesmith.real import SR, copy_as_16k_mono, read_16k_mono


def _tone(sr, seconds=1.0, channels=1):
    t = np.arange(int(sr * seconds)) / sr
    a = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return np.stack([a] * channels, axis=1) if channels > 1 else a


def test_44k_stereo_reads_as_16k_mono(tmp_path):
    f = tmp_path / "phone.wav"
    sf.write(f, _tone(44100, channels=2), 44100)
    a = read_16k_mono(f)
    assert a.ndim == 1 and a.dtype == np.float32
    assert abs(len(a) - SR) <= 1
    peak = np.argmax(np.abs(np.fft.rfft(a)))  # 1 Hz bins over 1 s
    assert abs(peak - 440) <= 2


def test_copy_converts_only_what_needs_it(tmp_path):
    ok, odd = tmp_path / "ok.wav", tmp_path / "odd.wav"
    sf.write(ok, _tone(SR), SR, subtype="PCM_16")
    sf.write(odd, _tone(48000, channels=2), 48000, subtype="PCM_24")
    copy_as_16k_mono(ok, tmp_path / "ok2.wav")
    copy_as_16k_mono(odd, tmp_path / "odd2.wav")
    assert (tmp_path / "ok2.wav").read_bytes() == ok.read_bytes()
    i = sf.info(str(tmp_path / "odd2.wav"))
    assert (i.samplerate, i.channels, i.subtype) == (SR, 1, "PCM_16")
    assert abs(i.frames - SR) <= 1


def test_odd_rates_and_long_clips_are_refused(tmp_path):
    import pytest

    low = tmp_path / "low.wav"
    sf.write(low, np.zeros(100, np.float32), 1000)
    with pytest.raises(ValueError, match="4-768 kHz"):
        read_16k_mono(low)
    from wakesmith import real

    long = tmp_path / "long.wav"
    sf.write(long, np.zeros(8000 * 61, np.int16), 8000, subtype="PCM_16")
    assert len(read_16k_mono(long)) == 16000 * 61
    real_max = real.MAX_SECONDS
    try:
        real.MAX_SECONDS = 60.0
        with pytest.raises(ValueError, match="too long to convert"):
            read_16k_mono(long)
    finally:
        real.MAX_SECONDS = real_max


def test_overshoot_is_clipped(tmp_path):
    f = tmp_path / "hot.wav"
    sf.write(f, np.full(4410, 1.5, np.float32), 44100, subtype="FLOAT")
    a = read_16k_mono(f)
    assert a.max() <= 1.0 and a.min() >= -1.0


def test_failed_conversion_leaves_no_file(tmp_path, monkeypatch):
    import pytest

    src = tmp_path / "odd.wav"
    sf.write(src, _tone(48000), 48000)
    real_write = sf.write

    def broken(path, *a, **k):
        real_write(path, *a, **k)
        raise OSError("disk full")

    monkeypatch.setattr(sf, "write", broken)
    with pytest.raises(OSError):
        copy_as_16k_mono(src, tmp_path / "out.wav")
    assert not (tmp_path / "out.wav").exists() and not list(tmp_path.glob(".*.tmp"))


def test_device_import_skips_unreadable_clips(tmp_path, monkeypatch):
    from wakesmith import project, rounds

    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path / "projects"))
    p = project.create("hey jarvis")
    d = tmp_path / "in" / "desk"
    d.mkdir(parents=True)
    sf.write(d / "good.wav", _tone(44100, channels=2), 44100)
    (d / "bad.wav").write_bytes(b"RIFF0000WAVEjunk")
    r = rounds.import_folder(p, tmp_path / "in")
    assert (r["added"], r["unreadable"], r["ids"]) == (1, 1, ["desk/good.wav"])
    i = sf.info(str(rounds.file(p, r["ids"][0])))
    assert (i.samplerate, i.channels) == (16000, 1)


def test_44k_stereo_not_the_phrase_clips_reach_tests_and_features_converted(tmp_path, monkeypatch):
    from wakesmith import mww, project
    from wakesmith.real import RealData

    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path / "projects"))
    neg = tmp_path / "neg"
    neg.mkdir()
    for i in range(8):
        sf.write(neg / f"n{i}.wav", _tone(44100, seconds=2.0, channels=2), 44100)
    (neg / "HELDOUT.txt").write_text("n0.wav\nn1.wav\n")
    p = project.create("hey jarvis")
    p.real.negatives = str(neg)
    p.save()
    r = RealData(p)
    r.prepare()
    for f in list(p.path("real_neg", "test").glob("*.wav")) + list(
        p.path("real_neg", "train").glob("*.wav")
    ):
        i = sf.info(str(f))
        assert (i.samplerate, i.channels) == (SR, 1), f
    assert list(p.path("real_neg", "test").glob("*.wav"))  # some held out
    # the feature job for real negatives reads the source clips: they come in converted too
    job = next(j for j in mww.feature_jobs(p) if j[0] == "real_negative")
    a = read_16k_mono(job[2][0])
    assert abs(len(a) - 2 * SR) <= 1


def test_16k_clips_of_any_length_read_as_before(tmp_path):
    f = tmp_path / "long16k.wav"
    sf.write(f, np.zeros(SR * 601, np.int16), SR, subtype="PCM_16")
    assert len(read_16k_mono(f)) == SR * 601


def test_converted_copy_keeps_the_clip_permissions(tmp_path):
    import os

    src = tmp_path / "odd.wav"
    sf.write(src, _tone(48000), 48000)
    os.chmod(src, 0o640)
    copy_as_16k_mono(src, tmp_path / "out.wav")
    assert (tmp_path / "out.wav").stat().st_mode & 0o777 == 0o640


def test_a_read_only_clip_still_converts(tmp_path):
    import os

    src = tmp_path / "ro.wav"
    sf.write(src, _tone(8000, channels=2), 8000)
    os.chmod(src, 0o2444)
    copy_as_16k_mono(src, tmp_path / "out.wav")
    assert (tmp_path / "out.wav").stat().st_mode & 0o7777 == 0o444
