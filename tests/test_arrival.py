"""0.4.12 device clips: sync on arrival (arrival.py), deleting a clip (rounds.delete), the device filter."""

import io
import json
import threading
import time
import wave
from pathlib import Path

import pytest

from wakesmith import arrival, clipsync, inbox, rounds
from wakesmith import project as projects

UI_TOKEN = "ui-token-for-the-page-0123"


def _wav(seconds=1.0) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * int(16000 * seconds))
    return buf.getvalue()


def _meta(name="20261117-035233-971-s1-p1.00-c0.85", model="hey_jarvis_v2", device="kitchen"):
    return {"name": name, "device": device, "kind": "wake", "at": "2026-11-17T03:52:33Z",
            "slot": 1, "model": model, "score": 1, "cutoff": 0.85}  # fmt: skip


@pytest.fixture(autouse=True)
def _fresh_gpu_reading(monkeypatch):
    """No GPU reading cached by one test (a fake nvidia-smi) reaches another, here or in other files."""
    from wakesmith import mww

    monkeypatch.setattr(mww, "_free_cache", (0.0, {}))


def _until(cond, seconds=10.0):
    end = time.monotonic() + seconds
    while not cond() and time.monotonic() < end:
        time.sleep(0.02)
    return cond()


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("WAKESMITH_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path / "data" / "projects"))
    return tmp_path


def _word(phrase, name, source="", prefix="", on=True):
    p = projects.create(phrase, name)
    p.devices.source, p.devices.model_prefix, p.devices.on_arrival = str(source), prefix, on
    p.save()
    return p


def _store(meta) -> Path:
    r = inbox.store(_wav(), json.dumps(meta).encode())
    return inbox.folder() / r["stored"]


def test_two_words_on_one_folder_each_take_only_their_model(home):
    src = inbox.folder()
    jarvis = _word("hey jarvis", "hey_jarvis", src, "hey_jarvis")
    alexa = _word("alexa", "alexa", src, "alexa")
    _word("computer", "computer", src, "computer", on=False)  # off: never synced on arrival
    _word("hey mycroft", "hey_mycroft", home / "elsewhere", "")  # another folder
    j = _store(_meta(model="hey_jarvis_v2"))
    a = _store(_meta(name="20261117-035426-890-s1-p0.99-c0.85", model="alexa_v2"))
    assert arrival.takers(j, "hey_jarvis_v2") == [str(jarvis.dir)]
    assert arrival.takers(a, "alexa_v2") == [str(alexa.dir)]
    every = _word("anything", "anything", src, "")  # empty filter: every clip in its folder
    assert str(every.dir) in arrival.takers(a, "alexa_v2")
    # a source above the inbox (the device folders' parent) takes it too
    up = _word("up", "up", src.parent, "alexa")
    assert str(up.dir) in arrival.takers(a, "alexa_v2")


def test_a_burst_is_one_sync_per_word_after_the_last_clip(home):
    src = inbox.folder()
    jarvis = _word("hey jarvis", "hey_jarvis", src, "hey_jarvis")
    alexa = _word("alexa", "alexa", src, "alexa")
    ran, done = [], threading.Event()

    def run(w):
        ran.append((w, time.monotonic()))
        if len(ran) == 2:
            done.set()

    a = arrival.Arrivals(run=run, delay=0.3)
    t0 = time.monotonic()
    for i in range(4):
        a.note(_store(_meta(name=f"20261117-0352{i:02d}-971-s1-p1.00-c0.85")), "hey_jarvis_x")
        last = time.monotonic()
        time.sleep(0.1)
    a.note(_store(_meta(name="20261117-035426-890-s1-p0.99-c0.85", model="alexa_x")), "alexa_x")
    assert done.wait(5)
    time.sleep(0.4)
    assert sorted(w for w, _ in ran) == sorted([str(jarvis.dir), str(alexa.dir)])  # once each
    j_at = next(t for w, t in ran if w == str(jarvis.dir))
    assert j_at - last >= 0.25 and j_at - t0 >= 0.55  # waited out the burst
    assert a.pending() == []


def test_nothing_runs_while_wakesmith_is_updating(home):
    _word("hey jarvis", "hey_jarvis", inbox.folder(), "hey_jarvis")
    frozen, ran = [True], []
    a = arrival.Arrivals(run=ran.append, delay=0.05, frozen=lambda: frozen[0])
    a.note(_store(_meta()), "hey_jarvis_v2")
    time.sleep(0.3)
    assert ran == [] and len(a.pending()) == 1
    frozen[0] = False
    a.note(_store(_meta(name="20261117-035254-929-s1-p1.00-c0.85")), "hey_jarvis_v2")
    assert _until(lambda: len(ran) == 1)


def test_a_failing_sync_doesnt_stop_the_next(home):
    _word("hey jarvis", "hey_jarvis", inbox.folder(), "hey_jarvis")
    ran = []

    def run(w):
        ran.append(w)
        if len(ran) == 1:
            raise OSError("disk full")

    a = arrival.Arrivals(run=run, delay=0.05)
    a.note(_store(_meta()), "hey_jarvis_v2")
    assert _until(lambda: len(ran) == 1)
    a.note(_store(_meta(name="20261117-035254-929-s1-p1.00-c0.85")), "hey_jarvis_v2")
    assert _until(lambda: len(ran) == 2)
    assert "sync on arrival failed: OSError: disk full" in clipsync.log_file().read_text()


def test_run_sync_starts_clips_sync_for_the_word(home, monkeypatch):
    seen = {}
    monkeypatch.setattr(arrival.subprocess, "run", lambda cmd, **kw: seen.update(cmd=cmd, kw=kw))
    arrival._run_sync("/x/hey_jarvis")
    i = seen["cmd"].index("wakesmith")
    assert seen["cmd"][i : i + 4] == ["wakesmith", "clips", "sync", "/x/hey_jarvis"]
    assert seen["kw"]["timeout"] == arrival.SYNC_TIMEOUT_S


@pytest.fixture
def client(home):
    from fastapi.testclient import TestClient

    from wakesmith.web.server import create_app

    app = create_app(UI_TOKEN, {"host": "0.0.0.0", "port": 8765, "https": True})
    return TestClient(app, base_url="https://192.0.2.20:8765")


def _send(client, token, meta):
    files = {
        "audio": ("c.wav", _wav(), "audio/wav"),
        "meta": (None, json.dumps(meta), "application/json"),
    }
    return client.post("/api/clips", files=files, headers={"Authorization": f"Bearer {token}"})


def test_an_upload_notes_the_words_that_take_it_but_not_a_duplicate(client, monkeypatch):
    noted = []
    monkeypatch.setattr(
        client.app.state.arrivals, "note", lambda stored, model: noted.append((stored, model))
    )
    tok = client.post(f"/api/device-upload/token?token={UI_TOKEN}").json()["token"]
    assert _send(client, tok, _meta()).status_code == 200
    assert _send(client, tok, _meta()).json()["duplicate"] is True
    assert len(noted) == 1
    stored, model = noted[0]
    assert model == "hey_jarvis_v2" and stored.exists() and stored.suffix == ".wav"


def test_an_arrival_failure_still_stores_the_clip(client, monkeypatch):
    def boom(stored, model):
        raise RuntimeError("no")

    monkeypatch.setattr(client.app.state.arrivals, "note", boom)
    tok = client.post(f"/api/device-upload/token?token={UI_TOKEN}").json()["token"]
    r = _send(client, tok, _meta())
    assert r.status_code == 200 and r.json()["ok"] is True


def test_the_setting_saves_and_old_projects_have_it_on(client, home):
    p = _word("hey jarvis", "hey_jarvis")
    y = p.path("project.yaml")
    y.write_text(y.read_text().replace("on_arrival: true", ""))
    assert projects.load(p.dir).devices.on_arrival is True
    r = client.put(
        f"/api/projects/hey_jarvis/devices/sync?token={UI_TOKEN}",
        json={"source": "", "on_arrival": False},
    )
    assert r.status_code == 200 and r.json()["on_arrival"] is False
    assert projects.load(p.dir).devices.on_arrival is False


# -- delete ---------------------------------------------------------------------------------------
def _clips(home, n=3):
    src = home / "device-clips"
    for i in range(n):
        d = src / "kitchen" / "wake"
        d.mkdir(parents=True, exist_ok=True)
        name = f"20261117-0352{i:02d}-971-s1-p1.00-c0.85"
        (d / f"{name}.wav").write_bytes(_wav(0.5))
        (d / f"{name}.json").write_text(json.dumps(_meta(name=name)))
    return src


def test_a_deleted_clip_is_gone_and_never_comes_back(home):
    p = _word("hey jarvis", "hey_jarvis")
    src = _clips(home)
    assert rounds.import_folder(p, src)["added"] == 3
    cid = rounds.load(p)[0]["id"]
    f = rounds.file(p, cid)
    assert rounds.delete(p, [cid, "kitchen/not-there.wav"]) == 1
    assert not f.exists() and cid not in {c["id"] for c in rounds.load(p)}
    r = rounds.import_folder(p, src)
    assert r["added"] == 0 and len(rounds.load(p)) == 2
    assert (src / "kitchen" / "wake" / Path(cid).name).exists()  # the source is only read
    assert rounds.delete(p, [rounds.load(p)[0]["id"]]) == 1
    assert len(rounds.deleted(p)) == 2  # remembered together


def test_delete_refuses_a_path_outside_the_inbox(home):
    p = _word("hey jarvis", "hey_jarvis")
    rounds.import_folder(p, _clips(home))
    clips = rounds.load(p)
    clips[0]["id"] = "../../project.yaml"
    rounds._save(p, clips)
    with pytest.raises(ValueError):
        rounds.delete(p, ["../../project.yaml"])
    assert p.path("project.yaml").exists()


def test_delete_endpoint(client, home, monkeypatch):
    p = _word("hey jarvis", "hey_jarvis")
    rounds.import_folder(p, _clips(home))
    cid = rounds.load(p)[0]["id"]
    url = f"/api/projects/hey_jarvis/devices/delete?token={UI_TOKEN}"
    assert client.post(url, json={"ids": "x"}).status_code == 400
    with clipsync._locked(p):  # a sync is running
        assert client.post(url, json={"ids": [cid]}).status_code == 409
    monkeypatch.setattr("wakesmith.jobs.busy", lambda p: True)
    assert client.post(url, json={"ids": [cid]}).status_code == 409
    monkeypatch.setattr("wakesmith.jobs.busy", lambda p: False)
    r = client.post(url, json={"ids": [cid]})
    assert r.status_code == 200 and r.json() == {"deleted": 1}


def test_page_has_the_filter_delete_and_arrival_setting():
    page = (Path(arrival.__file__).parent / "web" / "index.html").read_text()
    for s in ('id="devfrom"', "data-ddel", "/devices/delete", 'id="syncarr"', "on_arrival:"):
        assert s in page


# -- trim -----------------------------------------------------------------------------------------
def _tone_clips(home, seconds=2.0):
    import numpy as np
    import soundfile as sf

    src = home / "device-clips" / "kitchen" / "wake"
    src.mkdir(parents=True)
    for i, label in enumerate(("wake", "not_wake")):
        name = f"20261117-0352{i:02d}-971-s1-p1.00-c0.85"
        a = (np.arange(int(16000 * seconds)) % 100).astype("int16")  # sample k holds k % 100
        sf.write(src / f"{name}.wav", a, 16000, subtype="PCM_16")
        (src / f"{name}.json").write_text(json.dumps(_meta(name=name)))
    return home / "device-clips"


def test_a_trim_keeps_the_file_whole_and_trains_on_the_part(home):
    import numpy as np
    import soundfile as sf

    p = _word("hey jarvis", "hey_jarvis")
    rounds.import_folder(p, _tone_clips(home))
    a, b = (c["id"] for c in rounds.load(p))
    rounds.label(p, {a: "wake", b: "not_wake"})
    c = rounds.set_trim(p, a, 0.5, 1.25)
    assert c["trim"] == [0.5, 1.25]
    rounds.set_trim(p, b, 1.0)  # no end: to the end of the clip
    assert next(x for x in rounds.load(p) if x["id"] == b)["trim"] == [1.0, 2.0]
    rounds.apply(p, "2099-01-01T00:00")
    pos = p.path("devices", "pos", a)
    assert not pos.is_symlink()
    got, sr = sf.read(pos, dtype="int16")
    assert sr == 16000 and len(got) == 12000 and got[0] == 8000 % 100  # 0.5 s .. 1.25 s exactly
    assert len(sf.read(rounds.file(p, a), dtype="int16")[0]) == 32000  # the clip itself is whole
    assert len(sf.read(p.path("devices", "neg", b))[0]) == 16000
    rounds.set_trim(p, a, None)
    assert "trim" not in next(x for x in rounds.load(p) if x["id"] == a)
    rounds.apply(p, "2099-01-01T00:00")
    assert pos.is_symlink() and np.array_equal(sf.read(pos, dtype="int16")[0][:5], [0, 1, 2, 3, 4])


@pytest.mark.parametrize(
    ("start", "end", "says"),
    [(-0.1, 1.0, "within"), (1.0, 0.5, "within"), (0.0, 2.5, "within"), (1.0, 1.2, "at least")],
)
def test_a_bad_trim_is_refused(home, start, end, says):
    p = _word("hey jarvis", "hey_jarvis")
    rounds.import_folder(p, _tone_clips(home))
    cid = rounds.load(p)[0]["id"]
    with pytest.raises(ValueError, match=says):
        rounds.set_trim(p, cid, start, end)
    assert "trim" not in rounds.load(p)[0]
    with pytest.raises(KeyError):
        rounds.set_trim(p, "kitchen/nope.wav", 0, 1)


def test_trim_endpoint(client, home):
    p = _word("hey jarvis", "hey_jarvis")
    rounds.import_folder(p, _tone_clips(home))
    cid = rounds.load(p)[0]["id"]
    url = f"/api/projects/hey_jarvis/devices/trim?token={UI_TOKEN}"
    assert client.post(url, json={"id": cid, "start": "0"}).status_code == 400
    assert client.post(url, json={"id": cid, "start": True}).status_code == 400
    assert client.post(url, json={"id": "x/y.wav", "start": 0, "end": 1}).status_code == 404
    assert client.post(url, json={"id": cid, "start": 1.0, "end": 1.1}).status_code == 400
    with clipsync._locked(p):
        assert client.post(url, json={"id": cid, "start": 0.2, "end": 1.0}).status_code == 409
    r = client.post(url, json={"id": cid, "start": 0.2, "end": 1.0})
    assert r.status_code == 200 and r.json()["trim"] == [0.2, 1.0]
    r = client.post(url, json={"id": cid, "start": None})
    assert r.status_code == 200 and "trim" not in r.json()
    page = (Path(arrival.__file__).parent / "web" / "index.html").read_text()
    assert "/devices/trim" in page and "data-dtrim" in page


def test_a_new_version_keeps_deleted_clips_deleted_and_trims(home):
    from wakesmith import versions

    p = _word("hey jarvis", "hey_jarvis")
    src = _tone_clips(home)
    rounds.import_folder(p, src)
    a, b = (c["id"] for c in rounds.load(p))
    rounds.delete(p, [a])
    rounds.set_trim(p, b, 0.5, 1.5)
    q = versions.create(p, "next")
    assert rounds.deleted(q) == {a}
    assert rounds.import_folder(q, src)["added"] == 0
    assert rounds.load(q)[0]["trim"] == [0.5, 1.5]


# -- takes recorded on a device -------------------------------------------------------------------
def _take_meta(i, run="r1", word="hey_jarvis_v2"):
    return {"name": f"20261117-0600{i:02d}-000-take", "device": "kitchen", "kind": "take",
            "at": f"2026-11-17T06:00:{i:02d}Z", "word": word, "run": run, "index": i,
            "frame": "plain"}  # fmt: skip


def _voice(seconds=1.0):
    import numpy as np

    t = np.arange(int(16000 * seconds)) / 16000
    a = (8000 * np.sin(2 * np.pi * 220 * t)).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(a.tobytes())
    return buf.getvalue()


def test_takes_go_into_recordings_once_and_clips_stay_clips(home):
    import soundfile as sf

    from wakesmith import recordings, takes

    p = _word("hey jarvis", "hey_jarvis", inbox.folder(), "hey_jarvis")
    other = _word("alexa", "alexa", inbox.folder(), "alexa")
    for i in range(3):
        inbox.store(_voice(), json.dumps(_take_meta(i)).encode())
    inbox.store(_voice(), json.dumps(_take_meta(9, word="alexa_v2")).encode())
    inbox.store(_voice(), json.dumps(_meta()).encode())  # an ordinary wake clip
    r = rounds.import_folder(p, inbox.folder(), "hey_jarvis")
    assert (r["added"], r["takes"]) == (1, 3)
    assert [c["id"] for c in rounds.load(p)] == ["kitchen/20261117-035233-971-s1-p1.00-c0.85.wav"]
    listing = recordings.listing(p, "pos")
    rows = listing["takes"] if "takes" in listing else listing["files"]
    assert len(rows) == 3
    assert any("device take: kitchen, run r1, #2, plain" == x.get("condition") for x in rows)
    first = takes.done(p)["kitchen/20261117-060000-000-take.wav"]
    a, _ = sf.read(recordings.path(p, "pos", first), dtype="int16")
    b, _ = sf.read(io.BytesIO(_voice()), dtype="int16")
    assert (a == b).all()  # a 16 kHz take comes in sample for sample
    again = rounds.import_folder(p, inbox.folder(), "hey_jarvis")
    assert (again["added"], again["takes"]) == (0, 0)  # each comes in once
    assert rounds.import_folder(other, inbox.folder(), "alexa")["takes"] == 1


def test_a_silent_take_is_refused_once_and_a_broken_one_tried_again(home):
    from wakesmith import clipsync as cs
    from wakesmith import takes

    p = _word("hey jarvis", "hey_jarvis")
    inbox.store(_wav(1.0), json.dumps(_take_meta(0)).encode())  # silence: recordings refuses it
    r = rounds.import_folder(p, inbox.folder())
    assert (r["takes"], r["takes_refused"], r["unreadable"]) == (0, 1, 0)
    assert takes.done(p) == {"kitchen/20261117-060000-000-take.wav": ""}
    assert rounds.import_folder(p, inbox.folder())["takes_refused"] == 0  # remembered, not retried
    bad = inbox.folder() / "kitchen" / "take" / "20261117-060001-000-take.wav"
    bad.write_bytes(b"RIFF....not a wav")
    bad.with_suffix(".json").write_text(json.dumps(_take_meta(1)))
    for _ in range(2):  # unreadable: counted and tried again
        assert rounds.import_folder(p, inbox.folder())["unreadable"] == 1
    assert "1 take(s) left out (silent" in cs.describe(
        {"new": 0, "transcribed": 0, "takes_refused": 1, "to_check": 0}
    )
    assert "2 takes recorded on a device" in cs.describe(
        {"new": 0, "transcribed": 0, "takes": 2, "to_check": 0}
    )


@pytest.mark.parametrize(
    ("change", "says"),
    [
        ({"word": None}, "'word' is missing"),
        ({"index": "3"}, "'index' must be a whole number"),
        ({"index": True}, "'index' must be a whole number"),
        ({"run": 5}, "'run' must be a text"),
    ],
)
def test_take_meta_is_checked(home, change, says):
    m = {**_take_meta(0), **change}
    m = {k: v for k, v in m.items() if v is not None}
    with pytest.raises(inbox.BadUpload, match=says):
        inbox.check_meta(json.dumps(m).encode())


def test_a_take_upload_lands_in_the_take_folder_and_notes_its_word(client, monkeypatch):
    noted = []
    monkeypatch.setattr(client.app.state.arrivals, "note", lambda s, m: noted.append((s, m)))
    tok = client.post(f"/api/device-upload/token?token={UI_TOKEN}").json()["token"]
    files = {
        "audio": ("c.wav", _voice(), "audio/wav"),
        "meta": (None, json.dumps(_take_meta(4)), "application/json"),
    }
    r = client.post("/api/clips", files=files, headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200 and r.json()["stored"].startswith("kitchen/take/")
    assert noted[0][1] == "hey_jarvis_v2"


# -- review fixes (0.4.12 round 1) ----------------------------------------------------------------
def test_a_new_version_doesnt_bring_device_takes_in_again(home):
    from wakesmith import recordings, versions

    p = _word("hey jarvis", "hey_jarvis", inbox.folder(), "hey_jarvis")
    for i in range(2):
        inbox.store(_voice(), json.dumps(_take_meta(i)).encode())
    assert rounds.import_folder(p, inbox.folder(), "hey_jarvis")["takes"] == 2
    q = versions.create(p, "next")  # no device clips at all: only takes
    r = rounds.import_folder(q, inbox.folder(), "hey_jarvis")
    assert r["takes"] == 0
    listing = recordings.listing(q, "pos")
    assert len(listing["takes"] if "takes" in listing else listing["files"]) == 2


def test_deleting_a_used_clip_takes_it_out_of_training_and_nothing_else(home):
    import csv as _csv

    from wakesmith.real import RealData

    p = _word("hey jarvis", "hey_jarvis")
    rounds.import_folder(p, _tone_clips(home))
    a, b = (c["id"] for c in rounds.load(p))
    rounds.label(p, {a: "wake", b: "wake"})
    rounds.apply(p, "2026-11-17T03:52:01Z")  # b held out
    rounds.label(p, {a: "not_wake"})  # changed since, not used yet: stays as last used
    rounds.delete(p, [b])
    d = p.path("devices", "pos")
    assert not (d / b).exists() and not (d / b).is_symlink()
    rows = list(_csv.reader((d / "labels.csv").read_text(encoding="utf-8").splitlines()))
    assert [r[0] for r in rows[1:]] == [a]
    assert b not in (d / "HELDOUT.txt").read_text()
    assert (d / a).exists()  # a still trains as the phrase, as last used
    assert not p.path("devices", "neg").exists()
    assert RealData(p)  # still reads


def test_recordings_changes_happen_one_at_a_time(home):
    from concurrent.futures import ThreadPoolExecutor

    from wakesmith import recordings

    p = _word("hey jarvis", "hey_jarvis")
    with ThreadPoolExecutor(8) as ex:
        names = list(ex.map(lambda _: recordings.add(p, "pos", _voice())["file"], range(24)))
    assert len(set(names)) == 24
    listing = recordings.listing(p, "pos")
    assert len(listing["takes"] if "takes" in listing else listing["files"]) == 24


def test_a_steady_trickle_still_syncs_and_syncs_are_capped(home):
    _word("hey jarvis", "hey_jarvis", inbox.folder(), "hey_jarvis")
    ran = []
    a = arrival.Arrivals(run=ran.append, delay=0.3, max_wait=0.5)
    end = time.monotonic() + 1.2
    i = 0
    while time.monotonic() < end and not ran:  # a new clip every 0.1 s, more often than the delay
        a.note(_store(_meta(name=f"20261117-0353{i:02d}-971-s1-p1.00-c0.85")), "hey_jarvis_v2")
        i += 1
        time.sleep(0.1)
    assert ran, "a clip more often than the delay must not hold the sync off forever"
    seen = {}
    import wakesmith.arrival as ar

    orig = ar.subprocess.run
    try:
        ar.subprocess.run = lambda cmd, **kw: seen.update(cmd=cmd)
        ar._run_sync("/x/w")
    finally:
        ar.subprocess.run = orig
    assert seen["cmd"][-2:] == ["--limit", str(arrival.LIMIT)]


def test_delete_and_trim_refuse_odd_bodies(client, home):
    _word("hey jarvis", "hey_jarvis")
    for path in ("delete", "trim"):
        url = f"/api/projects/hey_jarvis/devices/{path}?token={UI_TOKEN}"
        assert client.post(url, content=b"not json").status_code == 400
        assert client.post(url, json=[1, 2]).status_code == 400
    url = f"/api/projects/hey_jarvis/devices/trim?token={UI_TOKEN}"
    huge = b'{"id": "x/y.wav", "start": 1' + b"0" * 400 + b"}"
    r = client.post(url, content=huge, headers={"content-type": "application/json"})
    assert r.status_code == 400
    inf = b'{"id": "x/y.wav", "start": Infinity}'
    assert (
        client.post(url, content=inf, headers={"content-type": "application/json"}).status_code
        == 400
    )


def test_a_label_set_while_whisper_runs_is_kept(home):
    p = _word("hey jarvis", "hey_jarvis")
    rounds.import_folder(p, _clips(home))
    ids = [c["id"] for c in rounds.load(p)]

    def whisper(paths):  # a person labels a clip while Whisper is busy
        rounds.label(p, {ids[2]: "not_wake"})
        return ["hey jarvis"] * len(paths)

    rounds.transcribe(p, whisper=whisper, batch=2)
    by = {c["id"]: c for c in rounds.load(p)}
    assert by[ids[2]]["label"] == "not_wake" and all(by[i]["text"] for i in ids)


def test_two_imports_at_once_bring_each_take_in_once(home):
    from concurrent.futures import ThreadPoolExecutor

    from wakesmith import recordings

    p = _word("hey jarvis", "hey_jarvis")
    for i in range(6):
        inbox.store(_voice(), json.dumps(_take_meta(i)).encode())
    with ThreadPoolExecutor(2) as ex:
        rs = list(ex.map(lambda _: rounds.import_folder(p, inbox.folder()), range(2)))
    assert sorted(r["takes"] for r in rs) == [0, 6]
    listing = recordings.listing(p, "pos")
    assert len(listing["takes"] if "takes" in listing else listing["files"]) == 6


def test_a_new_version_waits_for_an_import_in_progress(home):
    from wakesmith import versions

    p = _word("hey jarvis", "hey_jarvis")
    rounds.import_folder(p, _tone_clips(home))
    made, release = [], threading.Event()

    def holder():
        with rounds.index_lock(p):  # as an import in another process would
            release.wait(5)

    h = threading.Thread(target=holder)
    h.start()
    time.sleep(0.1)
    m = threading.Thread(target=lambda: made.append(versions.create(p, "next")))
    m.start()
    time.sleep(0.3)
    assert not made  # waiting for the lock
    release.set()
    h.join(5)
    m.join(10)
    assert made and len(rounds.load(made[0])) == 2


def test_the_two_locks_nest_in_one_thread(home):
    from wakesmith import recordings

    p = _word("hey jarvis", "hey_jarvis")
    with rounds.index_lock(p), rounds.index_lock(p), recordings.locked(p):
        assert recordings.add(p, "pos", _voice())["file"]  # takes the held lock again, no deadlock
        rounds.label(p, {})


# -- review fixes (0.4.12 round 2) ----------------------------------------------------------------
def test_a_broken_recordings_list_doesnt_refuse_takes_for_good(home):
    from wakesmith import recordings, takes

    p = _word("hey jarvis", "hey_jarvis")
    recordings.add(p, "pos", _voice())
    lab = recordings.folder(p, "pos") / "labels.csv"
    good = lab.read_text()
    lab.write_text(good + "take_999.wav,a,b,c,d,e\n")  # more cells than columns: edited by hand
    for i in range(2):
        inbox.store(_voice(), json.dumps(_take_meta(i)).encode())
    r = rounds.import_folder(p, inbox.folder())
    assert (r["takes"], r["takes_refused"], r["unreadable"]) == (0, 0, 2) and takes.done(p) == {}
    lab.write_text(good)  # fixed by the person
    assert rounds.import_folder(p, inbox.folder())["takes"] == 2


def test_a_sync_never_saves_the_project_for_a_take(home, monkeypatch):
    from wakesmith import recordings

    p = _word("hey jarvis", "hey_jarvis")
    inbox.store(_voice(), json.dumps(_take_meta(0)).encode())
    saved = []
    monkeypatch.setattr(recordings, "use", lambda *a: saved.append(a))
    assert rounds.import_folder(p, inbox.folder())["takes"] == 1
    assert saved == []


def test_use_the_labeled_clips_leaves_out_a_missing_clip(home):
    p = _word("hey jarvis", "hey_jarvis")
    rounds.import_folder(p, _tone_clips(home))
    a, b = (c["id"] for c in rounds.load(p))
    rounds.label(p, {a: "wake", b: "wake"})
    rounds.set_trim(p, a, 0.5, 1.5)
    rounds.file(p, a).unlink()  # gone by hand
    stats = rounds.apply(p, "2099-01-01T00:00")
    assert stats["pos"] == {"train": 1, "heldout": 0, "missing": 1}
    assert (p.path("devices", "pos") / b).exists() and (
        p.path("devices", "pos") / "labels.csv"
    ).exists()


def test_a_sync_that_stopped_at_the_limit_runs_again(home):
    p = _word("hey jarvis", "hey_jarvis", inbox.folder(), "hey_jarvis")
    ran = []

    def run(w):
        ran.append(w)
        left = 5 if len(ran) == 1 else 0
        f = Path(w) / "devices" / "sync.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"left": left, "at": time.time()}))

    a = arrival.Arrivals(run=run, delay=0.05)
    a.note(_store(_meta()), "hey_jarvis_v2")
    assert _until(lambda: len(ran) == 2)
    time.sleep(0.3)
    assert ran == [str(p.dir)] * 2


def test_locks_fall_back_when_the_share_has_no_flock(home, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from wakesmith import recordings

    p = _word("hey jarvis", "hey_jarvis")

    def no_flock(fd, op):
        raise OSError(95, "Operation not supported")

    monkeypatch.setattr(recordings.fcntl, "flock", no_flock)
    with ThreadPoolExecutor(6) as ex:
        names = list(ex.map(lambda _: recordings.add(p, "pos", _voice())["file"], range(12)))
    assert len(set(names)) == 12
    rounds.label(p, {})  # rounds' lock too


def test_a_stale_or_skipped_sync_result_doesnt_run_it_again(home):
    p = _word("hey jarvis", "hey_jarvis", inbox.folder(), "hey_jarvis")
    f = p.path("devices", "sync.json")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"left": 50, "at": time.time() - 3600}))  # an older run's
    ran = []
    a = arrival.Arrivals(run=ran.append, delay=0.05)  # this run writes nothing (skipped: busy)
    a.note(_store(_meta()), "hey_jarvis_v2")
    assert _until(lambda: len(ran) == 1)
    time.sleep(0.3)
    assert len(ran) == 1 and a.pending() == []
    f.write_text(json.dumps({"result": "skipped", "left": 9, "at": time.time() + 60}))
    a.note(_store(_meta(name="20261117-035254-929-s1-p1.00-c0.85")), "hey_jarvis_v2")
    assert _until(lambda: len(ran) == 2)
    time.sleep(0.3)
    assert len(ran) == 2


def test_labels_must_be_an_object(client, home):
    _word("hey jarvis", "hey_jarvis")
    url = f"/api/projects/hey_jarvis/devices/labels?token={UI_TOKEN}"
    for body in ({"labels": [1]}, {"labels": "abc"}, {"labels": 5}):
        assert client.post(url, json=body).status_code == 400
    deep = b"[" * 100000
    r = client.post(url, content=deep, headers={"content-type": "application/json"})
    assert r.status_code == 400


# -- review fixes (0.4.12 round 3) ----------------------------------------------------------------
def test_two_new_versions_at_once_make_two_whole_versions(home):
    from concurrent.futures import ThreadPoolExecutor

    from wakesmith import versions

    p = _word("hey jarvis", "hey_jarvis")
    rounds.import_folder(p, _tone_clips(home))
    with ThreadPoolExecutor(2) as ex:
        made = list(ex.map(lambda n: versions.create(p, n), ["a", "b"]))
    assert len({q.name for q in made}) == 2
    for q in made:
        assert len(rounds.load(q)) == 2 and q.path("project.yaml").exists()
    home_dir = made[0].dir.parent
    assert not [d for d in home_dir.iterdir() if ".making" in d.name]


def test_locks_make_no_folders_and_missing_clips_are_counted(home):
    from wakesmith import recordings

    p = _word("hey jarvis", "hey_jarvis")
    with rounds.index_lock(p), recordings.locked(p):
        pass
    assert not p.path("recordings").exists() and not p.path("devices").exists()
    rounds.import_folder(p, _tone_clips(home))
    a, _ = (c["id"] for c in rounds.load(p))
    rounds.label(p, {a: "wake"})
    assert "missing" not in rounds.summary(p)
    rounds.file(p, a).unlink()
    assert rounds.summary(p)["missing"] == 1


def test_new_version_refuses_a_body_that_isnt_an_object(client, home):
    _word("hey jarvis", "hey_jarvis")
    r = client.post(f"/api/projects/hey_jarvis/new-version?token={UI_TOKEN}", json=[1])
    assert r.status_code == 400


def test_new_version_sweeps_old_leftovers_and_says_when_it_lost_a_race(home, client, monkeypatch):
    import os as _os

    from wakesmith import versions

    p = _word("hey jarvis", "hey_jarvis")
    home_dir = p.dir.parent
    old = home_dir / ".hey_jarvis_v2.making.abc123"
    old.mkdir()
    _os.utime(old, (time.time() - 2 * 86400,) * 2)
    fresh = home_dir / ".hey_jarvis_v3.making.def456"
    fresh.mkdir()  # maybe another process at work: kept
    q = versions.create(p, "next")
    assert not old.exists() and fresh.exists()
    assert (q.dir / "project.yaml").exists() and not (q.dir / "clips").exists()

    def taken(*a, **kw):
        raise FileExistsError("hey_jarvis_v3 exists already; try again")

    monkeypatch.setattr(versions, "create", taken)
    r = client.post(f"/api/projects/hey_jarvis/new-version?token={UI_TOKEN}", json={})
    assert r.status_code == 409 and "try again" in r.json()["detail"]


def test_new_version_links_the_shared_clips(home):
    from wakesmith import versions

    p = _word("hey jarvis", "hey_jarvis")
    p.path("clips").mkdir(parents=True)
    q = versions.create(p, "next")
    assert q.path("clips").is_symlink() and q.path("clips").resolve() == p.path("clips").resolve()


# -- review fixes (0.4.12 round 4) ----------------------------------------------------------------
def test_a_dot_device_name_stays_inside_the_inbox_and_the_page_still_loads(home):
    p = _word("hey jarvis", "hey_jarvis")
    src = home / "odd"
    src.mkdir()
    (src / "a.wav").write_bytes(_wav(0.5))
    (src / "a.json").write_text(json.dumps({"device": "..", "at": "2026-11-17T03:52:00Z"}))
    rounds.import_folder(p, src)
    (c,) = rounds.load(p)
    assert c["id"] == "__/a.wav" and rounds.file(p, c["id"]).is_file()
    rounds.label(p, {c["id"]: "wake"})
    rounds.summary(p, "2099-01-01")


def test_missing_clips_are_left_out_of_the_counts(home):
    p = _word("hey jarvis", "hey_jarvis")
    rounds.import_folder(p, _tone_clips(home))
    a, b = (c["id"] for c in rounds.load(p))
    rounds.label(p, {a: "wake", b: "wake"})
    rounds.file(p, b).unlink()
    s = rounds.summary(p, "2000-01-01")  # everything held out
    assert s["missing_by"] == {"wake": 1, "not_wake": 0} and s["heldout"]["wake"] == 1


def test_a_new_version_folder_has_normal_permissions(home):
    import os as _os

    from wakesmith import versions

    p = _word("hey jarvis", "hey_jarvis")
    q = versions.create(p, "next")
    mask = _os.umask(0)
    _os.umask(mask)
    assert q.dir.stat().st_mode & 0o777 == 0o777 & ~mask


def test_a_hidden_device_folder_keeps_its_id(home):
    p = _word("hey jarvis", "hey_jarvis")
    src = home / "odd"
    (src / ".kitchen").mkdir(parents=True)
    (src / ".kitchen" / "a.wav").write_bytes(_wav(0.5))
    rounds.import_folder(p, src)
    assert [c["id"] for c in rounds.load(p)] == [".kitchen/a.wav"]  # as before 0.4.12
    assert rounds.import_folder(p, src)["added"] == 0


def test_new_version_works_where_flock_isnt_supported(home, monkeypatch):
    from wakesmith import versions

    def no_flock(fd, op):
        raise OSError(95, "Operation not supported")

    monkeypatch.setattr(versions.fcntl, "flock", no_flock)
    monkeypatch.setattr(rounds.fcntl, "flock", no_flock)
    p = _word("hey jarvis", "hey_jarvis")
    assert versions.create(p, "next").path("project.yaml").exists()


def test_importing_the_current_folder_keeps_ids_inside_the_project(home, monkeypatch):
    p = _word("hey jarvis", "hey_jarvis")
    src = home / "here"
    src.mkdir()
    (src / "a.wav").write_bytes(_wav(0.5))
    monkeypatch.chdir(src)
    rounds.import_folder(p, ".")
    (c,) = rounds.load(p)
    assert c["id"] == "_/a.wav"
    rounds.label(p, {c["id"]: "wake"})
    rounds.set_trim(p, c["id"], 0.1, 0.45)
    assert rounds.apply(p, "2099-01-01")["pos"]["train"] == 1
    assert (p.path("devices", "pos") / "_" / "a.wav").exists()
    # an id an older version made (absolute) is left out, never written or removed outside
    clips = rounds.load(p)
    clips.append({**clips[0], "id": "/a.wav"})
    rounds._save(p, clips)
    assert rounds.apply(p, "2099-01-01")["pos"].get("missing") == 1
    rounds._drop_used(p, "/a.wav")


def test_clip_ids_inside_the_inbox():
    ok, bad = (
        ["kitchen/a.wav", "./a.wav", "_/a.wav", ".kitchen/a.wav"],
        ["/a.wav", "../a.wav", "a/../b.wav", "a", "a/"],
    )
    assert all(rounds._inside(i) for i in ok) and not any(rounds._inside(i) for i in bad)


def test_a_clip_imported_from_dot_by_an_older_version_isnt_imported_again(home, monkeypatch):
    p = _word("hey jarvis", "hey_jarvis")
    src = home / "here"
    src.mkdir()
    (src / "a.wav").write_bytes(_wav(0.5))
    rounds._save(p, [{"id": "/a.wav", "device": "", "at": "2026-11-01T00:00:00Z", "model": "",
                      "score": None, "text": None, "suggestion": None, "label": None}])  # fmt: skip
    monkeypatch.chdir(src)
    assert rounds.import_folder(p, ".")["added"] == 0


# -- 0.4.13: device take runs as a test set ------------------------------------------------------
def _held_out(p):
    from wakesmith import recordings

    listing = recordings.listing(p, "pos")
    rows = listing["takes"] if "takes" in listing else listing["files"]
    return sorted((r.get("condition") or "").split(", ")[1] for r in rows if r.get("held"))


def test_take_runs_alternate_test_then_train_and_keep_their_role(home):
    from wakesmith import takes

    p = _word("hey jarvis", "hey_jarvis")
    for run, n in (("r1", 3), ("r2", 2), ("r3", 2)):
        for i in range(n):
            m = {**_take_meta(i, run=run), "name": f"20261117-0600{i:02d}-000-{run}"}
            inbox.store(_voice(), json.dumps(m).encode())
    assert rounds.import_folder(p, inbox.folder())["takes"] == 7
    assert takes.runs(p) == {"kitchen/r1": "test", "kitchen/r2": "train", "kitchen/r3": "test"}
    assert _held_out(p) == ["run r1"] * 3 + ["run r3"] * 2
    p.devices.takes = "train"  # changing the setting later leaves runs already seen as they were
    p.save()
    m = {**_take_meta(9, run="r1"), "name": "20261117-060009-000-r1"}
    inbox.store(_voice(), json.dumps(m).encode())
    rounds.import_folder(p, inbox.folder())
    assert _held_out(p).count("run r1") == 4


@pytest.mark.parametrize(("mode", "held"), [("train", 0), ("test", 4)])
def test_take_modes_train_or_test_every_run(home, mode, held):
    p = _word("hey jarvis", "hey_jarvis")
    p.devices.takes = mode
    p.save()
    for run in ("r1", "r2"):
        for i in range(2):
            m = {**_take_meta(i, run=run), "name": f"20261117-0600{i:02d}-000-{run}"}
            inbox.store(_voice(), json.dumps(m).encode())
    rounds.import_folder(p, inbox.folder())
    assert len(_held_out(p)) == held


def test_takes_without_a_run_id_group_by_gaps_not_the_clock(home):
    from wakesmith import takes

    s = {}

    def take(dev, at):
        k = takes.run_key(dev, {}, at, s)
        takes.note(s, dev, k, at)
        return k

    k1 = take("kitchen", "2026-11-17T14:58:00Z")
    assert take("kitchen", "2026-11-17T15:03:00Z") == k1  # across the hour
    k2 = take("kitchen", "2026-11-17T15:20:00Z")  # 17 minutes later: a new run
    assert k2 != k1 and take("hall", "2026-11-17T15:21:00Z").startswith("hall/")
    assert takes.run_key("kitchen", {"run": "abc"}, "x", s) == "kitchen/abc"


def test_a_late_take_joins_its_run_and_moves_nothing(home):
    from wakesmith import takes

    s = {}

    def take(at):
        k = takes.run_key("d", {}, at, s)
        takes.note(s, "d", k, at)
        return k

    k = take("2026-11-17T08:00:00Z")
    assert take("2026-11-17T08:06:00Z") == k
    assert take("2026-11-17T08:05:00Z") == k  # uploaded late (or retried): same run
    assert take("2026-11-17T08:07:00Z") == k and s["d"][-1]["last"] == "2026-11-17T08:07:00Z"
    assert takes.run_key("d", {}, "2026-11-17T07:55:00Z", s) == k  # a little before its start
    assert takes.run_key("d", {}, "2026-11-17T07:40:00Z", s) != k


def test_a_take_without_a_sidecar_is_timed_by_its_file_and_runs_split(home):
    import os as _os

    from wakesmith import takes

    p = _word("hey jarvis", "hey_jarvis")
    d = inbox.folder() / "kitchen" / "take"
    d.mkdir(parents=True)
    for i, ago in enumerate((7200, 7100, 60)):  # two takes two hours ago, one a minute ago
        f = d / f"t{i}.wav"
        f.write_bytes(_voice())
        _os.utime(f, (time.time() - ago,) * 2)
    assert rounds.import_folder(p, inbox.folder())["takes"] == 3
    assert list(takes.runs(p).values()) == ["test", "train"]


def test_a_run_whose_takes_all_fail_takes_no_turn(home):
    from wakesmith import takes

    p = _word("hey jarvis", "hey_jarvis")
    silent = {**_take_meta(0, run="r0"), "name": "20261117-060000-000-r0"}
    inbox.store(_wav(1.0), json.dumps(silent).encode())  # refused: silent
    for i in range(2):
        m = {**_take_meta(i, run="r1"), "name": f"20261117-0601{i:02d}-000-r1"}
        inbox.store(_voice(), json.dumps(m).encode())
    rounds.import_folder(p, inbox.folder())
    assert takes.runs(p) == {"kitchen/r1": "test"}  # the first run that landed is the test


def test_alternation_follows_the_last_role_after_a_mode_change(home):
    from wakesmith import takes

    p = _word("hey jarvis", "hey_jarvis")
    assert takes.next_role(p, {}) == "test"
    assert takes.next_role(p, {"a": "test", "b": "test"}) == "train"
    assert takes.next_role(p, {"a": "test", "b": "train"}) == "test"


def test_the_takes_setting_saves_and_is_checked(client, home):
    _word("hey jarvis", "hey_jarvis")
    url = f"/api/projects/hey_jarvis/devices/sync?token={UI_TOKEN}"
    assert client.put(url, json={"takes": "test"}).json()["takes"] == "test"
    assert client.put(url, json={"takes": "sometimes"}).status_code == 400
    page = (Path(arrival.__file__).parent / "web" / "index.html").read_text()
    assert 'id="synctakes"' in page


def test_a_word_whose_job_runs_syncs_once_the_job_ends(home):
    p = _word("hey jarvis", "hey_jarvis", inbox.folder(), "hey_jarvis")
    busy, ran = [True], []
    a = arrival.Arrivals(run=ran.append, delay=0.05, busy=lambda w: busy[0], recheck=0.1)
    a.note(_store(_meta()), "hey_jarvis_v2")
    time.sleep(0.4)
    assert ran == [] and a.pending() == [str(p.dir)]  # still waiting, not dropped
    busy[0] = False
    assert _until(lambda: ran == [str(p.dir)])


# -- 0.4.13: timing in the candidate pick --------------------------------------------------------
def test_speech_end_finds_the_last_loud_frame():
    import numpy as np

    from wakesmith import evaluate

    a = np.zeros(16000, np.float32)
    a[3200:9600] = 0.3 * np.sin(np.arange(6400) / 5)
    assert evaluate.speech_end(a) == 0.6
    assert evaluate.speech_end(np.zeros(16000, np.float32)) is None


def test_the_pick_passes_over_candidates_that_fire_early():
    from wakesmith import candidates

    base = {"positives": 100, "negatives": 400, "cutoff": 0.9}
    cands = [
        {**base, "n": 1, "caught": 90, "false": 2, "fires_ms": -20.0},
        {**base, "n": 2, "caught": 85, "false": 2, "fires_ms": 60.0},
        {**base, "n": 3, "caught": 80, "false": 2},
    ]
    best, why = candidates.choose(cands)
    assert best["n"] == 2 and "1 that fire before the wake word ends passed over" in why
    every = [{**c, "fires_ms": -5.0} for c in cands]  # all early: the usual pick, nothing skipped
    best, why = candidates.choose(every)
    assert best["n"] == 1 and "passed over" not in why


def test_timing_without_test_clips_is_none(home):
    from wakesmith import candidates

    p = _word("hey jarvis", "hey_jarvis")
    assert candidates.timing(p, Path("nope.tflite"), 0.9) is None


def test_a_weak_on_time_candidate_never_beats_a_good_early_one():
    from wakesmith import candidates

    base = {"positives": 50, "negatives": 400, "cutoff": 0.9}
    cands = [
        {**base, "n": 1, "caught": 45, "false": 2, "fires_ms": -10.0},
        {**base, "n": 2, "caught": 5, "false": 1},
        {"n": 3, "failed": "boom"},
    ]
    best, why = candidates.choose(cands)
    assert best["n"] == 1 and "passed over" not in why


def test_a_gpu_another_program_filled_is_left_out():
    from wakesmith import mww

    gpu = lambda d, gb=16: type("G", (), {"device": d, "memory_gb": gb})()
    two = [gpu("cuda:0"), gpu("cuda:1")]
    assert mww.train_devices(two, free_gb={"cuda:0": 0.8, "cuda:1": 15.0}) == ["cuda:1"]
    assert mww.train_devices(two, free_gb={"cuda:0": 0.8, "cuda:1": 2.0}) == ["cuda:1"]  # best left
    assert mww.train_devices(two, free_gb={}) == ["cuda:0", "cuda:1"]  # unknown: as before
    assert mww.train_devices([gpu("xpu:0")], free_gb={"cuda:0": 1.0}) == ["xpu:0"]


def test_gpu_free_follows_cuda_visible_devices(monkeypatch):
    from wakesmith import mww

    out = {"s": "0, NVIDIA RTX 5060 Ti, 900\n1, NVIDIA RTX 5060 Ti, 15000\n"}

    class R:
        @property
        def stdout(self):
            return out["s"]

    monkeypatch.setattr(mww.subprocess, "run", lambda *a, **kw: R())
    for k in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
        monkeypatch.delenv(k, raising=False)
    assert mww.gpu_free_gb() == {"cuda:0": 900 / 1024, "cuda:1": 15000 / 1024}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert mww.gpu_free_gb() == {"cuda:0": 15000 / 1024}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-8f2a,GPU-11c0")  # UUIDs: not matched
    assert mww.gpu_free_gb() == {}
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    out["s"] = "0, NVIDIA RTX 3060, 900\n1, NVIDIA RTX 4090, 15000\n"  # unlike cards
    assert mww.gpu_free_gb() == {}  # torch orders them fastest first: can't match by index
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    assert mww.gpu_free_gb() == {"cuda:0": 900 / 1024, "cuda:1": 15000 / 1024}


def test_the_estimate_counts_only_gpus_with_room(home, monkeypatch):
    from wakesmith import eta, mww

    p = _word("hey jarvis", "hey_jarvis")
    p.microwakeword.candidates = 4
    monkeypatch.setattr(eta, "_load", lambda: {"mww-train:slots": 2})
    monkeypatch.setattr(mww, "gpu_free_gb", lambda max_age=0: {"cuda:0": 0.5, "cuda:1": 14.0})
    assert eta._rounds(p) == 4  # one card with room: four rounds, not two
    monkeypatch.setattr(mww, "gpu_free_gb", lambda max_age=0: {})
    assert eta._rounds(p) == 2


def test_the_gpu_check_reads_the_trainers_mask_inside_a_stage(monkeypatch):
    from wakesmith import mww, pipeline

    class R:
        stdout = "0, NVIDIA RTX 5060 Ti, 900\n1, NVIDIA RTX 5060 Ti, 15000\n"

    monkeypatch.setattr(mww.subprocess, "run", lambda *a, **kw: R())
    for k in (
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "WAKESMITH_CUDA_DEVICES",
        "WAKESMITH_TF_GPU",
    ):
        monkeypatch.setenv(k, "x")  # recorded, so the test restores whatever _tf_off_gpu sets
        monkeypatch.delenv(k)
    pipeline._tf_off_gpu()  # as every training stage does: CUDA_VISIBLE_DEVICES=-1 for TensorFlow
    assert mww.gpu_free_gb() == {"cuda:0": 900 / 1024, "cuda:1": 15000 / 1024}
    gpu = lambda d: type("G", (), {"device": d, "memory_gb": 16})()
    assert mww.train_devices([gpu("cuda:0"), gpu("cuda:1")], free_gb=mww.gpu_free_gb()) == [
        "cuda:1"
    ]
    monkeypatch.setenv("WAKESMITH_CUDA_DEVICES", "1")  # the trainer gets card 1 only
    assert mww.gpu_free_gb() == {"cuda:0": 15000 / 1024}


def test_the_estimate_keeps_its_gpu_count_while_training_runs(home, monkeypatch):
    from wakesmith import eta, mww

    p = _word("hey jarvis", "hey_jarvis")
    p.microwakeword.candidates = 4
    monkeypatch.setattr(eta, "_load", lambda: {"mww-train:slots": 2})
    monkeypatch.setattr(mww, "gpu_free_gb", lambda max_age=0: {"cuda:0": 1.0, "cuda:1": 1.0})
    from wakesmith import candidates, jobs

    monkeypatch.setattr(jobs, "busy", lambda q: True)  # a page job or a command-line run
    f = candidates.progress_path(p)  # its own run, on 2 cards its trainers now fill
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"n": 4, "slots": 2, "done": [], "running": {}}))
    assert eta._rounds(p) == 2
    f.write_text(json.dumps({"n": 4, "slots": 1, "done": [], "running": {}}))
    assert eta._rounds(p) == 4  # a run that got one card: four rounds, whatever the last run had


def test_a_late_take_from_an_older_run_leaves_the_current_run_whole(home):
    from wakesmith import takes

    s = {}

    def take(at):
        k = takes.run_key("d", {}, at, s)
        takes.note(s, "d", k, at)
        return k

    r1 = take("2026-11-17T10:00:00Z")
    assert take("2026-11-17T10:04:00Z") == r1
    r2 = take("2026-11-17T10:30:00Z")
    assert r2 != r1
    assert take("2026-11-17T10:03:00Z") == r1  # a retried take of the first run
    assert take("2026-11-17T10:32:00Z") == r2  # the current run is still one run
    old = {
        "d": {"key": "d/2026-11-17T09:00", "last": "2026-11-17T09:05:00Z"}
    }  # first builds' shape
    assert takes.run_key("d", {}, "2026-11-17T09:07:00Z", old) == "d/2026-11-17T09:00"


def test_a_late_run_older_than_the_remembered_ones_stays_whole(home):
    from wakesmith import takes

    s = {}

    def take(at):
        k = takes.run_key("d", {}, at, s)
        takes.note(s, "d", k, at)
        return k

    for day in range(1, 22):  # 21 daily runs
        take(f"2026-10-{day:02d}T10:00:00Z")
    k = take("2026-09-30T10:00:00Z")  # a late run from before all of them
    assert take("2026-09-30T10:01:00Z") == k and take("2026-09-30T10:02:00Z") == k
    assert len(s["d"]) == takes.KEEP_SPANS


def test_a_stopped_runs_progress_file_steers_nothing(home, monkeypatch):
    from wakesmith import candidates, eta, jobs, mww

    p = _word("hey jarvis", "hey_jarvis")
    p.microwakeword.candidates = 4
    monkeypatch.setattr(eta, "_load", lambda: {"mww-train:slots": 2})
    monkeypatch.setattr(mww, "gpu_free_gb", lambda max_age=0: {"cuda:0": 1.0, "cuda:1": 14.0})
    f = candidates.progress_path(p)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"n": 4, "slots": 2, "done": [], "running": {}}))  # left by a kill
    assert eta._rounds(p) == 4  # idle: the GPUs with room now (one), not the dead run's two
    job = p.path("job.json")
    job.write_text(json.dumps({"pid": 1, "title": "train", "started": time.time()}))
    jobs._finish(job, json.loads(job.read_text()), -15)  # the launcher notes the stop
    assert not f.exists()


def test_a_command_line_runs_progress_file_is_left_alone(home, monkeypatch):
    from wakesmith import candidates, jobs

    p = _word("hey jarvis", "hey_jarvis")
    f = candidates.progress_path(p)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("{}")
    monkeypatch.setattr(jobs, "outside_run", lambda q: True)  # trained from the command line now
    jobs._drop_progress(p.dir)
    assert f.exists()
    monkeypatch.setattr(jobs, "outside_run", lambda q: False)
    jobs._drop_progress(p.dir)
    assert not f.exists()
