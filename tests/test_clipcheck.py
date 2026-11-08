"""The listening check (clipcheck.py): which sound-alike clips are left out, and that the features skip them.
Whisper itself is replaced by a stand-in: these tests download nothing and need no GPU."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import soundfile as sf

from wakesmith import clipcheck, eta, mww, oww, project, sysstat

HEARD = {  # what the stand-in "hears" in each clip
    "clip_000000.wav": "Hey, Jarvis.",  # the phrase: left out
    "clip_000001.wav": "Hey Jarvis is",  # contains the phrase: a real sound-alike, kept
    "clip_000002.wav": "hey harvest",
    "clip_000003.wav": "",  # heard nothing: kept
}


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path / "projects"))
    monkeypatch.setenv("WAKESMITH_HOME", str(tmp_path / "data"))
    p = project.create("hey jarvis", negatives=["jarvis", "hey harvest"])
    for split in ("positive_train", "positive_test", "negative_train", "negative_test"):
        d = p.path("clips", split)
        d.mkdir(parents=True)
        for name in HEARD:
            sf.write(d / name, np.zeros(1600, np.float32), 16000)
    return p


def listen(paths):
    return [HEARD[f.name] for f in paths]


def names(files):
    return sorted(os.path.basename(f) for f in files)


def test_only_clips_heard_as_exactly_the_phrase_are_left_out(proj):
    assert len(clipcheck.todo(proj)) == 8  # 4 sound-alike clips in each of the two splits
    assert clipcheck.run_share(proj, "cpu", 0, 1, listen=listen) == 8
    assert clipcheck.todo(proj) == []
    assert clipcheck.phrase_clips(proj, "negative_train") == {"clip_000000.wav"}
    files = [str(f) for f in sorted(proj.path("clips", "negative_train").glob("*.wav"))]
    assert names(clipcheck.keep(proj, "negative_train", files)) == [
        "clip_000001.wav",
        "clip_000002.wav",
        "clip_000003.wav",
    ]
    # the phrase's own clips are never touched
    pos = [str(f) for f in proj.path("clips", "positive_train").glob("*.wav")]
    assert clipcheck.keep(proj, "positive_train", pos) == pos
    s = clipcheck.summary(proj)
    assert (s["checked"], s["heard_as_phrase"], s["heard"]) == (8, 2, ["hey jarvis"])
    # another way of writing the phrase counts too
    proj.written_as = ["hey harvest"]
    assert clipcheck.phrase_clips(proj, "negative_test") == {"clip_000000.wav", "clip_000002.wav"}


def test_a_clip_made_again_is_heard_again(proj):
    clipcheck.run_share(proj, "cpu", 0, 1, listen=listen)
    f = proj.path("clips", "negative_train", "clip_000000.wav")
    os.utime(f, ns=(f.stat().st_atime_ns, f.stat().st_mtime_ns + 10**9))
    assert clipcheck.todo(proj) == [("negative_train", "clip_000000.wav")]
    assert "clip_000000.wav" not in clipcheck.phrase_clips(proj, "negative_train")


def test_shares_split_one_list_between_cards(proj):
    items = clipcheck.todo(proj)
    for k in range(3):
        clipcheck.run_share(proj, "cpu", k, 3, items, listen=listen)
    lines = [
        json.loads(x)
        for f in proj.path("clips", "whisper").glob("negatives-*.jsonl")
        for x in f.read_text().splitlines()
    ]
    assert sorted((r["split"], r["file"]) for r in lines) == sorted(items)  # each clip once
    total = sum(
        json.loads(proj.path("logs", f"check-clips-{k}.json").read_text())["total"]
        for k in range(3)
    )
    assert total == len(items)


def test_a_cut_off_line_is_ignored(proj):
    clipcheck.run_share(proj, "cpu", 0, 1, listen=listen)
    with open(proj.path("clips", "whisper", "negatives-0.jsonl"), "a") as fh:
        fh.write('{"split": "negative_train", "fi')
    assert clipcheck.phrase_clips(proj, "negative_train") == {"clip_000000.wav"}


def test_skipped_without_a_gpu_or_when_turned_off(proj):
    assert clipcheck.run(proj, ["cpu"], log=lambda m: None)["skipped"] == "needs a GPU"
    clipcheck.run_share(proj, "cpu", 0, 1, listen=listen)
    r = clipcheck.run(proj, ["cuda:0"], log=lambda m: None)
    assert r["skipped"].startswith("every sound-alike clip") and r["heard_as_phrase"] == 2
    proj.samples.check_negatives = False
    files = [str(f) for f in proj.path("clips", "negative_train").glob("*.wav")]
    assert clipcheck.keep(proj, "negative_train", files) == files


def test_features_leave_out_what_was_heard_as_the_phrase(proj):
    clipcheck.run_share(proj, "cpu", 0, 1, listen=listen)
    jobs = {(s, split): files for s, split, files, _, _ in mww.feature_jobs(proj)}
    adv = jobs[("adversarial", "training")]
    assert "clip_000000.wav" not in names(adv) and len(adv) == 3
    assert len(jobs[("positive", "training")]) == 4
    sets = oww._feature_sets(proj)
    assert names(sets["negative_train"][0]) == [
        "clip_000001.wav",
        "clip_000002.wav",
        "clip_000003.wav",
    ]
    assert len(sets["negative_test"][0]) == 3 and len(sets["positive_train"][0]) == 4


def test_time_estimate_counts_clips_still_to_hear(proj, monkeypatch):
    monkeypatch.setattr(eta, "_can_listen", lambda: True)  # GitHub's runners have no GPU
    proj.samples.negative, proj.samples.negative_test = 4, 4
    assert eta.units(proj, "check-clips") == 8
    clipcheck.run_share(proj, "cpu", 0, 1, listen=listen)
    assert eta.units(proj, "check-clips") == 0
    eta._path().parent.mkdir(parents=True, exist_ok=True)
    eta.record(proj, "check-clips", 4.0, {"listened": 8, "checked": 8})
    assert eta._load()["check-clips"] == pytest.approx(0.5)


def test_a_run_started_outside_the_page_shows_as_running(proj):
    import subprocess
    import sys

    from wakesmith import jobs

    assert jobs.shown(proj) == {"running": False}
    stand_in = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    started = 5.0  # however the clock has moved since
    try:
        proj.set_state(
            **{
                "mww-features": {
                    "status": "running",
                    "pid": stand_in.pid,
                    "started": started,
                    "ticks": sysstat.start_ticks(stand_in.pid),
                }
            }
        )
        job = jobs.shown(proj)
        assert job["running"] and job["outside"] and job["started"] == started
        assert job["pid"] == os.getsid(stand_in.pid)
        assert job["run_pid"] == stand_in.pid and jobs.busy(proj)
        # another process start than the stage noted: it only got the pid of a run that's gone
        gone = {"status": "running", "pid": stand_in.pid, "started": 5.0, "ticks": 1}
        proj.set_state(**{"mww-features": gone})
        assert jobs.shown(proj)["running"] is False and not jobs.busy(proj)
        assert jobs.stages(proj, ["mww-features"])["mww-features"]["status"] == "failed"
        # noted by an older version, without ticks: from before this boot, it can't be this process
        old = {"status": "running", "pid": stand_in.pid, "started": 5.0}
        proj.set_state(**{"mww-features": old})
        assert not jobs.busy(proj)
        proj.set_state(**{"mww-features": {**old, "started": os.stat("/proc/self").st_mtime}})
        assert jobs.busy(proj)
    finally:
        stand_in.kill()
        stand_in.wait()
    assert not jobs.shown(proj)["running"]


def _clip(p, split, name):
    d = p.path("clips", split)
    d.mkdir(parents=True, exist_ok=True)
    p.path("logs").mkdir(exist_ok=True)
    f = d / name
    f.write_bytes(b"x")
    return f


def test_a_newer_record_wins_whichever_cards_file_holds_it(tmp_path, monkeypatch):
    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path))
    p = project.create("hey jarvis")
    f = _clip(p, "negative_train", "clip_000000.wav")
    now = f.stat().st_mtime_ns
    d = clipcheck._dir(p)
    d.mkdir(parents=True)
    rec = {"split": "negative_train", "file": f.name}
    (d / "negatives-0.jsonl").write_text(
        json.dumps({**rec, "mtime": now, "text": "hey jarvis"}) + "\n"
    )
    (d / "negatives-1.jsonl").write_text(
        json.dumps({**rec, "mtime": now - 5, "text": "hey harvest"}) + "\n"
    )
    assert clipcheck.todo(p) == []
    assert clipcheck.phrase_clips(p, "negative_train") == {f.name}


def test_whisper_that_cant_be_loaded_skips_the_check_not_the_training(tmp_path, monkeypatch):
    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path))
    p = project.create("hey jarvis")
    _clip(p, "negative_train", "clip_000000.wav")

    def no_model(*a):
        raise OSError("couldn't connect to huggingface.co")

    monkeypatch.setattr(clipcheck, "_Listener", no_model)
    monkeypatch.setattr(clipcheck, "available", lambda: "")
    monkeypatch.setattr(clipcheck, "_free_gpu", lambda: None)
    logged = []
    out = clipcheck.run(p, ["cuda:0"], log=logged.append)
    assert out["skipped"] == "Whisper couldn't be loaded" and "huggingface" in out["error"]
    assert out["unchecked"] == 1 and out["checked"] == 0
    assert any("huggingface" in m for m in logged)
    assert clipcheck.todo(p)  # tried again on the next run


def test_cards_that_cant_load_whisper_skip_too(tmp_path, monkeypatch):
    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path))
    p = project.create("hey jarvis")
    _clip(p, "negative_train", "clip_000000.wav")
    monkeypatch.setattr(clipcheck, "available", lambda: "")

    class Proc:
        def __init__(self, *a, **k):
            pass

        def wait(self):
            return clipcheck.CANT_LOAD_EXIT

    monkeypatch.setattr(clipcheck.subprocess, "Popen", Proc)
    assert clipcheck.run(p, ["cuda:0", "cuda:1"], log=lambda m: None)["skipped"]


def test_another_failure_on_a_card_still_fails_the_stage(tmp_path, monkeypatch):
    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path))
    p = project.create("hey jarvis")
    _clip(p, "negative_train", "clip_000000.wav")
    monkeypatch.setattr(clipcheck, "available", lambda: "")

    class Proc:
        def __init__(self, *a, **k):
            pass

        def wait(self):
            return 1

    monkeypatch.setattr(clipcheck.subprocess, "Popen", Proc)
    with pytest.raises(RuntimeError, match="failed on"):
        clipcheck.run(p, ["cuda:0", "cuda:1"], log=lambda m: None)


def test_the_estimate_leaves_out_a_check_that_would_be_skipped(tmp_path, monkeypatch):
    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path))
    p = project.create("hey jarvis")
    monkeypatch.setattr(eta, "_can_listen", lambda: False)
    assert eta.units(p, "check-clips") == 0
    monkeypatch.setattr(eta, "_can_listen", lambda: True)
    assert eta.units(p, "check-clips") > 0


def test_the_record_for_the_file_as_it_is_wins_even_if_its_older(tmp_path, monkeypatch):
    """A copy restored with its old time isn't heard again and again."""
    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path))
    p = project.create("hey jarvis")
    f = _clip(p, "negative_train", "clip_000000.wav")
    now = f.stat().st_mtime_ns
    d = clipcheck._dir(p)
    d.mkdir(parents=True)
    rec = {"split": "negative_train", "file": f.name}
    lines = [
        {**rec, "mtime": now, "text": "hey jarvis"},
        {**rec, "mtime": now + 5, "text": "hey harvest"},
    ]
    (d / "negatives-0.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    assert clipcheck.todo(p) == []
    assert clipcheck.phrase_clips(p, "negative_train") == {f.name}


def test_a_skip_keeps_saying_what_earlier_checks_left_out(tmp_path, monkeypatch):
    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path))
    p = project.create("hey jarvis")
    f = _clip(p, "negative_train", "clip_000000.wav")
    _clip(p, "negative_train", "clip_000001.wav")
    d = clipcheck._dir(p)
    d.mkdir(parents=True)
    rec = {
        "split": "negative_train",
        "file": f.name,
        "mtime": f.stat().st_mtime_ns,
        "text": "hey jarvis",
    }
    (d / "negatives-0.jsonl").write_text(json.dumps(rec) + "\n")
    out = clipcheck.run(p, ["cpu"], log=lambda m: None)
    assert out["skipped"] == "needs a GPU" and out["unchecked"] == 1
    assert out["checked"] == 1 and out["heard_as_phrase"] == 1


def test_partly_loaded_cards_count_only_what_was_heard(tmp_path, monkeypatch):
    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path))
    p = project.create("hey jarvis")
    _clip(p, "negative_train", "clip_000000.wav")
    _clip(p, "negative_train", "clip_000001.wav")
    monkeypatch.setattr(clipcheck, "available", lambda: "")
    codes = iter([0, clipcheck.CANT_LOAD_EXIT])

    class Proc:
        def __init__(self, cmd, **k):
            self.k = int(cmd[5])

        def wait(self):
            code = next(codes)
            if code == 0:
                p.path("logs", f"check-clips-{self.k}.json").write_text(json.dumps({"done": 1}))
            return code

    monkeypatch.setattr(clipcheck.subprocess, "Popen", Proc)
    out = clipcheck.run(p, ["cuda:0", "cuda:1"], log=lambda m: None)
    assert out["listened"] == 1
