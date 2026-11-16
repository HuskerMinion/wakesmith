"""Device upload (inbox.py): devices POST their clips to /api/clips with a Bearer token."""

import io
import json
import wave

import pytest

from wakesmith import __version__, assistant, clipsync, inbox, project, rounds

UI_TOKEN = "ui-token-for-the-page-0123"
META = {
    "name": "20261012-213301-123-s1-p0.79-c0.90",
    "device": "kitchen",
    "kind": "nearmiss",
    "at": "2026-10-12T21:33:01Z",
    "slot": 1,
    "model": "hey_jarvis",
    "score": 0.79,
    "cutoff": 0.90,
}


def _wav(seconds=4.0, rate=16000, channels=1, width=2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * int(seconds * rate) * channels * (width // 2 or 1))
    return buf.getvalue()


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("WAKESMITH_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("WAKESMITH_PROJECTS", str(tmp_path / "data" / "projects"))
    return tmp_path


@pytest.fixture
def client(home):
    from fastapi.testclient import TestClient

    from wakesmith.web.server import create_app

    # as on the LAN: the page needs its own token, devices theirs
    app = create_app(UI_TOKEN, {"host": "0.0.0.0", "port": 8765, "https": True})
    return TestClient(app, base_url="https://192.0.2.20:8765")


@pytest.fixture
def token(client):
    r = client.post(f"/api/device-upload/token?token={UI_TOKEN}")
    assert r.status_code == 200 and r.json()["on"] is True
    return r.json()["token"]


def _send(client, token, meta=None, audio=None, raw_meta=None):
    meta = META if meta is None else meta
    files = {
        "audio": ("clip.wav", _wav() if audio is None else audio, "audio/wav"),
        "meta": (None, raw_meta if raw_meta is not None else json.dumps(meta), "application/json"),
    }
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.post("/api/clips", files=files, headers=headers)


def test_good_upload_stores_the_device_layout(client, token):
    r = _send(client, token)
    assert r.status_code == 200, r.text
    rel = "kitchen/nearmiss/20261012-213301-123-s1-p0.79-c0.90.wav"
    assert r.json() == {"ok": True, "stored": rel}
    root = inbox.folder()
    assert (root / rel).read_bytes() == _wav()
    assert json.loads((root / rel).with_suffix(".json").read_text()) == META  # as received
    assert sorted(p.name for p in (root / "kitchen" / "nearmiss").iterdir()) == [
        "20261012-213301-123-s1-p0.79-c0.90.json",
        "20261012-213301-123-s1-p0.79-c0.90.wav",
    ]  # no temp files left


def test_same_name_again_is_a_safe_duplicate(client, token):
    assert _send(client, token).status_code == 200
    r = _send(client, token, audio=_wav(seconds=1.0))
    assert r.status_code == 200
    assert r.json() == {
        "ok": True,
        "stored": "kitchen/nearmiss/20261012-213301-123-s1-p0.79-c0.90.wav",
        "duplicate": True,
    }
    stored = inbox.folder() / r.json()["stored"]
    assert stored.read_bytes() == _wav()  # the first one is kept
    # the same name from another device is a different clip
    other = _send(client, token, meta={**META, "device": "garage"})
    assert other.status_code == 200 and "duplicate" not in other.json()


def test_wrong_or_missing_token(client, token):
    for t in ("nope", token + "x", ""):
        r = _send(client, t)
        assert r.status_code == 401 and r.json()["ok"] is False
    r = client.post("/api/clips", headers={"Authorization": f"Basic {token}"})
    assert r.status_code == 401
    assert not inbox.folder().exists() or not any(inbox.folder().rglob("*.wav"))


def test_off_until_a_token_is_made_and_after_revoke(client):
    assert client.get("/api/clips/ping", headers={"Authorization": "Bearer x"}).status_code == 401
    assert "off" in _send(client, "anything").json()["error"]
    t = client.post(f"/api/device-upload/token?token={UI_TOKEN}").json()["token"]
    assert _send(client, t).status_code == 200
    assert client.delete(f"/api/device-upload/token?token={UI_TOKEN}").json()["on"] is False
    assert _send(client, t, meta={**META, "name": "another"}).status_code == 401


def test_a_new_token_replaces_the_old_one(client, token):
    new = client.post(f"/api/device-upload/token?token={UI_TOKEN}").json()["token"]
    assert new != token
    assert _send(client, token).status_code == 401
    assert _send(client, new).status_code == 200


@pytest.mark.parametrize(
    ("audio", "says"),
    [
        (_wav(channels=2), "channels 2"),
        (_wav(rate=44100), "sample rate 44100"),
        (_wav(width=1), "bits per sample 8"),
        (b"RIFF....WAVEnot really", "not a PCM WAV"),
        (b"", "not a PCM WAV"),
    ],
)
def test_bad_wav_is_400(client, token, audio, says):
    r = _send(client, token, audio=audio)
    assert r.status_code == 400 and says in r.json()["error"]
    assert not any(inbox.folder().rglob("*.wav")) if inbox.folder().exists() else True


@pytest.mark.parametrize(
    ("meta", "says"),
    [
        ({k: v for k, v in META.items() if k != "kind"}, "'kind'"),
        ({**META, "kind": "help"}, "kind must be one of"),
        ({**META, "at": "yesterday"}, "'at'"),
        ({**META, "score": "high"}, "'score'"),
        ({**META, "slot": 1.5}, "'slot'"),
        ({**META, "name": "///"}, "'name'"),
        ({**META, "at": "2026-10-12T23:33:01+02:00"}, "'at'"),
        ({**META, "at": "2026-10-12T21:33:01z"}, "'at'"),
        ({**META, "at": "20261012T213301Z"}, "'at'"),
        ({**META, "at": "2026-10-12Z"}, "'at'"),
        ({**META, "at": "2026-10-12 21:33:01Z"}, "'at'"),
        ({**META, "at": "2026-10-12T21:33Z"}, "'at'"),
        ({**META, "device": ""}, "'device'"),
    ],
)
def test_bad_meta_is_400_and_says_which(client, token, meta, says):
    r = _send(client, token, meta=meta)
    assert r.status_code == 400 and says in r.json()["error"], r.text


def test_meta_not_json_and_missing_parts(client, token):
    r = _send(client, token, raw_meta="{not json")
    assert r.status_code == 400 and "meta: not JSON" in r.json()["error"]
    h = {"Authorization": f"Bearer {token}"}
    r = client.post("/api/clips", files={"audio": ("a.wav", _wav(), "audio/wav")}, headers=h)
    assert r.status_code == 400 and "missing part: meta" in r.json()["error"]
    r = client.post("/api/clips", content=_wav(), headers={**h, "Content-Type": "audio/wav"})
    assert r.status_code == 400 and "multipart" in r.json()["error"]


@pytest.mark.parametrize(
    ("device", "name", "folder", "file"),
    [
        ("/etc", "/etc/passwd", "_etc", "_etc_passwd.wav"),
        ("a/../../b", "x\\..\\..\\y", "a_.._.._b", "x_.._.._y.wav"),
        (" kit chen ", "a b", "_kit_chen_", "a_b.wav"),
    ],
)
def test_separators_become_underscores_and_stay_in_the_folder(
    client, token, device, name, folder, file
):
    r = _send(client, token, meta={**META, "device": device, "name": name})
    assert r.status_code == 200, r.text
    assert r.json()["stored"] == f"{folder}/nearmiss/{file}"
    root = inbox.folder().resolve()
    assert (root / r.json()["stored"]).resolve().parent.parent.parent == root


@pytest.mark.parametrize(
    ("device", "name"),
    [
        ("../../outside", "ok"),
        ("ok", "../../escape"),
        ("..", "ok"),
        ("ok", ".."),
        (".hidden", "ok"),
        ("ok", ".bashrc"),
        ("ok", "x" * 200),
        ("///", "ok"),
    ],
)
def test_dot_names_and_traversal_are_refused(client, token, device, name):
    r = _send(client, token, meta={**META, "device": device, "name": name})
    assert r.status_code == 400 and "not start with a dot" in r.json()["error"], r.text
    assert not inbox.folder().exists() or not any(inbox.folder().rglob("*.wav"))


def test_too_large_is_413(client, token):
    big = _wav(seconds=70)  # 2.24 MB
    assert len(big) > inbox.AUDIO_MAX_BYTES
    assert _send(client, token, audio=big).status_code == 413
    # a body that says it's small but isn't
    h = {"Authorization": f"Bearer {token}", "Content-Type": "multipart/form-data; boundary=x"}

    def chunks():
        for _ in range(40):
            yield b"\0" * 65536

    assert client.post("/api/clips", content=chunks(), headers=h).status_code == 413
    assert not inbox.folder().exists() or not any(inbox.folder().rglob("*.wav"))


def test_audio_part_over_the_limit_is_413_even_inside_the_body_limit(home):
    with pytest.raises(inbox.TooLarge):
        inbox.store(b"\0" * (inbox.AUDIO_MAX_BYTES + 1), json.dumps(META).encode())


def test_ping(client, token):
    r = client.get("/api/clips/ping", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200 and r.json() == {"ok": True, "wakesmith": __version__}
    assert client.get("/api/clips/ping", headers={"Authorization": "Bearer no"}).status_code == 401
    assert client.get("/api/clips/ping").status_code == 401


def test_device_token_opens_only_the_inbox_and_the_ui_token_never_leaks(client, token):
    client.cookies.clear()  # the token fixture's page visit left the UI cookie
    h = {"Authorization": f"Bearer {token}"}
    assert client.get("/api/device-upload", headers=h).status_code == 401  # UI needs its own
    assert client.get("/api/projects", headers=h).status_code == 401
    # a ?token= on the device paths neither checks nor hands out the UI's cookie
    r = client.get(f"/api/clips/ping?token={UI_TOKEN}", headers=h)
    assert r.status_code == 200 and "set-cookie" not in r.headers
    r = client.get("/api/clips/ping?token=guess")
    assert r.status_code == 401 and "set-cookie" not in r.headers


def test_no_cors(client, token):
    pre = client.options(
        "/api/clips",
        headers={
            "Origin": "https://elsewhere.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    assert "access-control-allow-origin" not in pre.headers
    r = client.post(
        "/api/clips", headers={"Origin": "https://elsewhere.example", "Authorization": "Bearer x"}
    )
    assert r.status_code == 403 and "access-control-allow-origin" not in r.headers


def test_only_a_hash_of_the_token_is_kept(client, token):
    text = assistant.settings_path().read_text()
    assert token not in text
    saved = json.loads(text)["device_upload"]
    assert saved["token_sha256"] == inbox._hash(token)
    page = client.get(f"/api/device-upload?token={UI_TOKEN}").json()
    assert page["on"] is True and "token" not in page and "token_sha256" not in page
    assert oct(assistant.settings_path().stat().st_mode & 0o777) == "0o600"


def test_settings_show_the_url_and_folder(client, home):
    page = client.get(f"/api/device-upload?token={UI_TOKEN}").json()
    assert page["on"] is False and page["reachable"] is True
    assert page["url"].startswith("https://") and page["url"].endswith(":8765/api/clips")
    assert "0.0.0.0" not in page["url"]
    assert page["folder"] == str(inbox.default_folder())
    elsewhere = home / "incoming"
    r = client.put(f"/api/device-upload?token={UI_TOKEN}", json={"folder": str(elsewhere)})
    assert r.json()["folder"] == str(elsewhere)
    assert (
        client.put(f"/api/device-upload?token={UI_TOKEN}", json={"folder": "rel"}).status_code
        == 400
    )


def test_upload_url_for_each_way_of_listening(monkeypatch):
    monkeypatch.setattr(inbox, "lan_address", lambda: "192.0.2.20")
    assert inbox.upload_url("0.0.0.0", 8765, False)["url"] == "http://192.0.2.20:8765/api/clips"
    assert inbox.upload_url("10.0.0.5", 9000, True)["url"] == "https://10.0.0.5:9000/api/clips"
    local = inbox.upload_url("127.0.0.1", 8765, False)
    assert local["reachable"] is False and "--host 0.0.0.0" in local["note"]


def test_an_uploaded_clip_shows_up_in_sync_with_score_and_model(client, token, home):
    folder = home / "incoming"
    client.put(f"/api/device-upload?token={UI_TOKEN}", json={"folder": str(folder)})
    assert _send(client, token).status_code == 200
    other_word = {**META, "name": "20261012-213400-001-s2-p0.95-c0.90", "model": "okay_nabu"}
    assert _send(client, token, meta=other_word).status_code == 200
    assert (
        _send(client, token, meta={**META, "device": "Shop Bench", "kind": "wake"}).status_code
        == 200
    )

    p = project.create("hey jarvis")
    p.devices.source, p.devices.model_prefix = str(folder), "hey_jarvis"
    p.save()

    class Whisper:
        def __call__(self, paths):
            return ["hey jarvis" for _ in paths]

    out = clipsync.sync(p, device="cpu", whisper=Whisper())
    assert out["new"] == 2 and out["result"] == "ok"
    clips = {c["id"]: c for c in rounds.load(p)}
    assert set(clips) == {
        "kitchen/20261012-213301-123-s1-p0.79-c0.90.wav",
        "Shop_Bench/20261012-213301-123-s1-p0.79-c0.90.wav",
    }
    for c in clips.values():
        assert c["model"] == "hey_jarvis" and c["score"] == 0.79 and c["at"] == META["at"]
    assert rounds.file(p, "kitchen/20261012-213301-123-s1-p0.79-c0.90.wav").read_bytes() == _wav()
    assert clipsync.sync(p, device="cpu", whisper=Whisper())["new"] == 0  # seen once only


def test_a_name_sent_before_as_another_kind_is_a_duplicate(client, token):
    assert _send(client, token, meta={**META, "kind": "wake"}).status_code == 200
    r = _send(client, token)  # nearmiss now: the import knows clips by device and name only
    assert r.json() == {
        "ok": True,
        "stored": "kitchen/wake/20261012-213301-123-s1-p0.79-c0.90.wav",
        "duplicate": True,
    }
    assert not (inbox.folder() / "kitchen" / "nearmiss").exists()


def test_a_sidecar_already_there_is_kept(client, token):
    d = inbox.folder() / "kitchen" / "nearmiss"
    d.mkdir(parents=True)
    side = d / "20261012-213301-123-s1-p0.79-c0.90.json"
    side.write_text('{"from": "the puller"}')  # its .wav not written yet
    assert _send(client, token).json() == {
        "ok": True,
        "stored": f"kitchen/nearmiss/{side.stem}.wav",
    }
    assert side.read_text() == '{"from": "the puller"}' and side.with_suffix(".wav").exists()


def test_a_write_error_leaves_no_temp_files(client, token, monkeypatch):
    import errno

    real = inbox._write_new
    calls = []

    def full(dst, data):
        calls.append(dst)
        if len(calls) == 2:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(dst, data)

    monkeypatch.setattr(inbox, "_write_new", full)
    r = _send(client, token)
    assert r.status_code == 500 and "No space left" in r.json()["error"]
    assert [p.name for p in inbox.folder().rglob("*") if p.is_file()] == [".inbox.lock"]


def test_write_new_removes_its_partial_file(home, monkeypatch, tmp_path):
    def broken(fd):
        raise OSError("fsync failed")

    monkeypatch.setattr(inbox.os, "fsync", broken)
    with pytest.raises(OSError):
        inbox._write_new(tmp_path / "a.wav", b"x")
    assert list(tmp_path.iterdir()) == []


def test_a_header_that_claims_more_audio_than_it_holds_is_400(client, token):
    wav = bytearray(_wav(seconds=0.5))
    wav[40:44] = (0xFFFFFFFF).to_bytes(4, "little")  # a streaming writer's "data" size
    r = _send(client, token, audio=bytes(wav))
    assert r.status_code == 400 and "the header says" in r.json()["error"]


@pytest.mark.parametrize("raw", ["[" * 100000, '{"n": ' + "9" * 5000 + "}"])
def test_meta_too_deep_or_too_big_a_number_is_400(client, token, raw):
    r = _send(client, token, raw_meta=raw)
    assert r.status_code == 400 and "meta: not JSON" in r.json()["error"]


def test_at_with_a_fraction_of_a_second_is_fine(client, token):
    assert _send(client, token, meta={**META, "at": "2026-10-12T21:33:01.123Z"}).status_code == 200


def test_multipart_with_bare_line_feeds(client, token):
    body = (
        b"--xyz\nContent-Disposition: form-data; name=meta\n\n"
        + json.dumps(META).encode()
        + b'\n--xyz\nContent-Disposition: form-data; name="audio"; filename="a.wav"\n\n'
        + _wav()
        + b"\n--xyz--\n"
    )
    h = {"Authorization": f"Bearer {token}", "Content-Type": "multipart/form-data; boundary=xyz"}
    r = client.post("/api/clips", content=body, headers=h)
    assert r.status_code == 200, r.text
    assert (inbox.folder() / r.json()["stored"]).read_bytes() == _wav()


def test_trailing_slash_paths_answer_like_the_inbox(client, token):
    client.cookies.clear()
    h = {"Authorization": f"Bearer {token}"}
    assert client.get("/api/clips/ping/", headers=h).json()["ok"] is True
    files = {"audio": ("a.wav", _wav(), "audio/wav"), "meta": (None, json.dumps(META))}
    assert client.post("/api/clips/", files=files, headers=h).status_code == 200
    r = client.get("/api/clips/ping/", headers={"Authorization": "Bearer no"})
    assert r.status_code == 401 and r.json()["ok"] is False


def test_off_and_wrong_token_answer_the_same(client):
    off = _send(client, "x").json()
    client.post(f"/api/device-upload/token?token={UI_TOKEN}")
    assert _send(client, "x").json() == off


def test_device_paths_get_the_inbox_answer_shape_when_refused(client, token):
    client.app.state.updater.frozen = lambda: True
    r = _send(client, token)
    assert r.status_code == 503 and r.json()["ok"] is False and "updating" in r.json()["error"]
    client.app.state.updater.frozen = lambda: False
    r = client.post("/api/clips", headers={"Origin": "https://elsewhere.example"})
    assert r.status_code == 403 and r.json() == {"ok": False, "error": "cross-site request refused"}


def test_too_large_says_both_limits(client, token):
    r = _send(client, token, audio=_wav(seconds=70))
    assert r.status_code == 413 and "2000000" in r.json()["error"]
    assert str(inbox.BODY_MAX_BYTES) in r.json()["error"]


def test_folder_setting_refuses_bad_input(client):
    put = lambda **kw: client.put(f"/api/device-upload?token={UI_TOKEN}", **kw)
    assert put(json=["/x"]).status_code == 400
    assert (
        put(content=b"{not json", headers={"Content-Type": "application/json"}).status_code == 400
    )
    r = put(json={"folder": "/tmp/a\u0000b"})
    assert r.status_code == 400 and "NUL" in r.json()["detail"]


def test_audio_holding_the_boundary_bytes_stays_whole(client, token):
    audio = _wav(seconds=0.5)
    audio = audio[:44] + b"--X--Xa--X" + audio[54:]  # a short fixed boundary inside the samples
    body = (
        b"--X\r\nContent-Disposition: form-data; name=meta\r\n\r\n"
        + json.dumps(META).encode()
        + b'\r\n--X\r\nContent-Disposition: form-data; name="audio"\r\n\r\n'
        + audio
        + b"\r\n--X--\r\n"
    )
    h = {"Authorization": f"Bearer {token}", "Content-Type": "multipart/form-data; boundary=X"}
    r = client.post("/api/clips", content=body, headers=h)
    assert r.status_code == 200, r.text
    assert (inbox.folder() / r.json()["stored"]).read_bytes() == audio


def test_a_part_without_headers_is_passed_over():
    body = b"--b\r\n\r\nno headers\r\n--b\r\nContent-Disposition: form-data; name=meta\r\n\r\n{}\r\n--b--"
    assert inbox.parse_multipart("multipart/form-data; boundary=b", body) == {"meta": b"{}"}


def test_the_lock_never_follows_a_symlink(home, tmp_path, monkeypatch):
    root = tmp_path / "clips"
    root.mkdir()
    (root / ".inbox.lock").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(OSError), inbox._locked(root):
        pass
    assert not (tmp_path / "elsewhere").exists()


def _parts(body, boundary="b"):
    return inbox.parse_multipart(f"multipart/form-data; boundary={boundary}", body)


def test_a_bare_lf_sender_keeps_a_final_cr_in_its_audio():
    body = b"--b\nContent-Disposition: form-data; name=audio\n\nabc\r\n--b--\n"
    assert _parts(body) == {"audio": b"abc\r"}


def test_a_body_without_its_closing_boundary_ends_the_last_part_at_the_break():
    body = b"--b\r\nContent-Disposition: form-data; name=meta\r\n\r\n{}\r\n"
    assert _parts(body) == {"meta": b"{}"}


def test_a_line_starting_with_the_boundary_and_more_stays_in_the_part():
    body = (
        b"--XX\r\nContent-Disposition: form-data; name=audio\r\n\r\nab\r\n--XXcd\r\nef"
        b"\r\n--XX--\r\n"
    )
    assert _parts(body, "XX") == {"audio": b"ab\r\n--XXcd\r\nef"}
