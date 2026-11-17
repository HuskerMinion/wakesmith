"""Device upload: an HTTP inbox devices push their wake word clips to, instead of someone copying them off.

POST /api/clips (Authorization: Bearer <token>, multipart/form-data with an "audio" WAV part and a
"meta" JSON part) stores <folder>/<device>/<kind>/<name>.wav and <name>.json, the same layout and
names as a device's own clip folder (characters other than letters, digits, _ . - become _, as the
import does), so Device clips' import and Sync automatically read them unchanged (point a wake word's
Device clips folder here). GET /api/clips/ping checks a token.

The inbox is off until a token is created. Only the token's SHA-256 is kept (settings.json
"device_upload"); the token itself is shown once, when it is made. Tokens are 256 random bits, so a
plain hash is enough, and it is compared in constant time.
"""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import io
import json
import os
import re
import secrets
import socket
import struct
import time
import wave
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from . import assets

AUDIO_MAX_BYTES = 2_000_000  # about 4 s of 16 kHz 16-bit mono is ~128 KB
BODY_MAX_BYTES = AUDIO_MAX_BYTES + 64_000  # the meta part and multipart framing on top
KINDS = ("wake", "nearmiss", "mistake", "noise", "take")  # take: recorded as the phrase (takes.py)
RATE, CHANNELS, WIDTH = 16000, 1, 2
KEY = "device_upload"


class BadUpload(ValueError):
    """400: the message says what's wrong."""


class TooLarge(ValueError):
    """413."""


# -- settings --------------------------------------------------------------------------------------
def default_folder() -> Path:
    return assets.home() / "device-clips"


def _saved() -> dict:
    from . import assistant  # settings.json's reader and locked writer live there

    s = assistant._read_all().get(KEY)
    return s if isinstance(s, dict) else {}


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def settings() -> dict:
    """What the settings page shows: never the token or its hash."""
    s = _saved()
    return {
        "on": bool(s.get("token_sha256")),
        "created": s.get("created"),
        "folder": s.get("folder") or str(default_folder()),
    }


def folder() -> Path:
    return Path(settings()["folder"]).expanduser()


def _change(fn) -> None:
    from . import assistant

    def apply(data: dict) -> None:
        cur = data.get(KEY)
        data[KEY] = fn(dict(cur) if isinstance(cur, dict) else {})

    assistant.update_all(apply)


def create_token() -> str:
    """A new token (it replaces any earlier one); returned once, only its hash is kept."""
    token = secrets.token_urlsafe(32)

    def put(s: dict) -> dict:
        s.update(token_sha256=_hash(token), created=time.time())
        s.setdefault("folder", str(default_folder()))
        return s

    _change(put)
    return token


def revoke() -> None:
    _change(lambda s: {k: v for k, v in s.items() if k not in ("token_sha256", "created")})


def set_folder(path: str) -> None:
    path = str(path or "").strip()
    if not path:
        raise ValueError("choose a folder for the clips")
    if "\0" in path:
        raise ValueError("the folder name has a NUL character")
    p = Path(path).expanduser()
    if not p.is_absolute():
        raise ValueError(f"use a full path, not {path!r}")
    if p.exists() and not p.is_dir():
        raise ValueError(f"not a folder: {path}")

    def put(s: dict) -> dict:
        s["folder"] = str(p)
        return s

    _change(put)


def check_auth(header: str | None) -> bool:
    """The Authorization header carries the current token ("Bearer <token>")."""
    want = _saved().get("token_sha256")
    scheme, _, given = (header or "").strip().partition(" ")
    if not want or scheme.lower() != "bearer" or not given.strip():
        # still hash and compare, so a missing token takes as long as a wrong one
        hmac.compare_digest(_hash(given.strip() or "-"), "0" * 64)
        return False
    return hmac.compare_digest(_hash(given.strip()), str(want))


# -- the address devices send to -------------------------------------------------------------------
def lan_address() -> str:
    """This machine's address on the LAN: the source address of the route out (no packet is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))  # TEST-NET-1: only picks the outgoing interface
            return s.getsockname()[0]
    except OSError:
        return ""


def upload_url(host: str, port: int, https: bool) -> dict:
    """{"url", "reachable", "note"} for where wakesmith listens (`serve --host/--port/--https`)."""
    scheme = "https" if https else "http"
    local = host in ("127.0.0.1", "localhost", "::1")
    if host in ("0.0.0.0", "::", ""):
        host = lan_address() or socket.gethostname()
    shown = f"[{host}]" if ":" in host else host
    out = {"url": f"{scheme}://{shown}:{port}/api/clips", "reachable": not local, "note": ""}
    if local:
        out["note"] = (
            "wakesmith listens on this computer only, so devices can't reach it: "
            "start it with wakesmith serve --host 0.0.0.0"
        )
    elif https:
        out["note"] = (
            "HTTPS with wakesmith's own self-signed certificate: the device must accept it"
        )
    return out


# -- one upload -------------------------------------------------------------------------------------
def parse_multipart(content_type: str, body: bytes) -> dict[str, bytes]:
    """The parts of a multipart/form-data body by their form name (the first of each name).

    A boundary counts only at the start of a line and followed by "--", padding or a line break, so
    audio holding the boundary's bytes stays whole. Line breaks are CR LF (the standard) or, when the
    first boundary line ends in one, a bare LF (some hand-written senders); only that one kind is
    taken as the break before a boundary, so a bare-LF sender's audio keeps a final CR. A part
    without headers has no name and is passed over; a body missing its closing boundary still ends
    its last part at the line break before the end."""
    m = re.match(r"\s*multipart/form-data\s*;(.*)$", content_type or "", re.IGNORECASE | re.DOTALL)
    b = re.search(r'(?:^|;)\s*boundary=(?:"([^"]+)"|([^;\s]+))', m[1], re.IGNORECASE) if m else None
    if not b:
        raise BadUpload("send multipart/form-data with an audio part and a meta part")
    dash = b"--" + (b[1] or b[2]).encode("latin-1")
    first = re.search(rb"(?:\A|\n)" + re.escape(dash) + rb"[ \t]*(\r?\n)", body)
    if not first:
        raise BadUpload("the multipart body has no parts")
    nl = first[1]
    delim = rb"(?:\A|" + nl + rb")" + re.escape(dash) + rb"(?=--|[ \t]*" + nl + rb")"
    segments = re.split(delim, body)[1:]
    closed = False
    found: list[tuple[str | None, bytes]] = []
    for seg in segments:
        if seg.startswith(b"--"):
            closed = True  # the closing boundary
            break
        seg = seg.lstrip(b" \t")[len(nl) :]  # transport padding, then the boundary line's break
        if seg.startswith(nl):
            found.append((None, b""))  # no headers: no name
            continue
        i = seg.find(nl + nl)
        if i < 0:
            raise BadUpload("a multipart part is malformed")
        head, content = seg[:i], seg[i + 2 * len(nl) :]
        disp = ""
        for line in head.split(nl):
            k, _, v = line.decode("latin-1").partition(":")
            if k.strip().lower() == "content-disposition":
                disp = v
        name = re.search(r'(?:^|;)\s*name=(?:"([^"]*)"|([^;\s]+))', disp, re.IGNORECASE)
        found.append(
            (name[1] if name and name[1] is not None else name[2] if name else None, content)
        )
    if not closed and found and found[-1][1].endswith(nl):  # cut off: the break before the end
        found[-1] = (found[-1][0], found[-1][1][: -len(nl)])
    parts: dict[str, bytes] = {}
    for name, content in found:
        if name is not None:
            parts.setdefault(name, content)
    return parts


def check_wav(data: bytes) -> float:
    """The clip's length in seconds; BadUpload unless it is a 16 kHz mono 16-bit PCM WAV that holds
    the samples its header says (a streaming header claims about 4 GB)."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            rate, ch, width, frames = (
                w.getframerate(),
                w.getnchannels(),
                w.getsampwidth(),
                w.getnframes(),
            )
            held = len(w.readframes(frames)) // max(1, ch * width)
    except (wave.Error, EOFError, ValueError, RuntimeError, struct.error) as e:
        raise BadUpload(f"audio: not a PCM WAV file ({e or type(e).__name__})") from None
    wrong = [
        f"{what} {got} (need {need})"
        for what, got, need in (
            ("sample rate", rate, RATE),
            ("channels", ch, CHANNELS),
            ("bits per sample", width * 8, WIDTH * 8),
        )
        if got != need
    ]
    if wrong:
        raise BadUpload("audio: " + ", ".join(wrong))
    if not frames:
        raise BadUpload("audio: the WAV file has no samples")
    if held != frames:
        raise BadUpload(f"audio: the header says {frames} samples, the file holds {held}")
    return frames / rate


NAME_MAX = 120
AT = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?Z")


def file_name(s: str) -> str:
    """The import's own rule (rounds.import_folder): anything but letters, digits, _ . - becomes _.
    The same rule on the device name gives the folder the import takes the device from, so a clip's
    folder and its import id always agree."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


def _usable(k: str, raw: str) -> str:
    s = file_name(raw)
    if not s.strip("_") or s.startswith(".") or len(s) > NAME_MAX:
        raise BadUpload(
            f"meta: {k!r} must be 1-{NAME_MAX} letters, digits, _ . - (others become _) "
            f"and not start with a dot: {raw[:60]!r}"
        )
    return s


def check_meta(raw: bytes) -> dict:
    try:
        meta = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as e:  # nested too deep, huge numbers
        raise BadUpload(f"meta: not JSON ({str(e)[:120] or type(e).__name__})") from None
    if not isinstance(meta, dict):
        raise BadUpload("meta: must be a JSON object")
    if meta.get("kind") == "take" and "model" not in meta:  # a take names its word as "word"
        need = ("name", "device", "kind", "at", "word")
    else:
        need = ("name", "device", "kind", "at", "model")
    for k in need:
        if not isinstance(meta.get(k), str) or not meta[k].strip():
            raise BadUpload(f"meta: {k!r} is missing or not a text")
    if meta["kind"] not in KINDS:
        raise BadUpload(f"meta: kind must be one of {', '.join(KINDS)}, not {meta['kind']!r}")
    at = meta["at"]
    try:  # one form, UTC with Z: the import sorts and filters times as text
        ok = bool(AT.fullmatch(at)) and datetime.fromisoformat(at[:-1] + "+00:00") is not None
    except ValueError:
        ok = False
    if not ok:
        raise BadUpload(
            f"meta: 'at' must be an RFC 3339 UTC time like 2026-10-12T21:33:01Z: {at[:60]!r}"
        )
    for k in ("score", "cutoff"):
        v = meta.get(k)
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))):
            raise BadUpload(f"meta: {k!r} must be a number")
    for k in ("slot", "index"):
        v = meta.get(k)
        if v is not None and (isinstance(v, bool) or not isinstance(v, int)):
            raise BadUpload(f"meta: {k!r} must be a whole number")
    for k in ("word", "run", "frame", "front"):
        v = meta.get(k)
        if v is not None and not isinstance(v, str):
            raise BadUpload(f"meta: {k!r} must be a text")
    _usable("device", meta["device"])
    _usable("name", _stem(meta["name"]))
    return meta


def _stem(name: str) -> str:
    return name[:-4] if name.lower().endswith(".wav") else name


def _write_new(dst: Path, data: bytes) -> Path:
    """data in a temp file next to dst (not named *.wav or *.json: an import running now skips it)."""
    tmp = dst.with_name(f".{dst.name}.{secrets.token_hex(4)}.part")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return tmp


def _put(tmp: Path, dst: Path) -> bool:
    """tmp in as dst unless dst is there already (then dst is left as it is); True when it went in."""
    try:
        os.link(tmp, dst)
        return True
    except FileExistsError:
        return False
    except OSError:  # a file system without hard links: a rename, after a look (under the lock)
        if dst.exists():
            return False
        os.replace(tmp, dst)
        return True


@contextmanager
def _locked(root: Path):
    """One upload at a time checks and writes (uploads are small; the check must see the last one)."""
    root.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)  # never through a symlink
    with os.fdopen(os.open(root / ".inbox.lock", flags, 0o600), "a") as fd:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)


def store(audio: bytes, meta_raw: bytes) -> dict:
    """Check and store one clip: {"ok", "stored"}, or with "duplicate" when this device already sent a
    clip of that name (of any kind: the import knows a clip by device and name). The sidecar goes in
    before the WAV, so an import never sees a WAV without one, and neither ever replaces a file that
    is there: a second upload changes nothing."""
    if len(audio) > AUDIO_MAX_BYTES:
        raise TooLarge(f"audio: {len(audio)} bytes, the limit is {AUDIO_MAX_BYTES}")
    meta = check_meta(meta_raw)
    check_wav(audio)
    device, name = file_name(meta["device"]), file_name(_stem(meta["name"])) + ".wav"
    root = folder()
    with _locked(root):
        for kind in KINDS:
            if (root / device / kind / name).exists():
                return {"ok": True, "stored": f"{device}/{kind}/{name}", "duplicate": True}
        dst = root / device / meta["kind"] / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmps: list[Path] = []
        try:
            tmps.append(_write_new(dst.with_suffix(".json"), meta_raw))
            tmps.append(_write_new(dst, audio))
            _put(tmps[0], dst.with_suffix(".json"))  # one there already (the puller's) stays
            _put(tmps[1], dst)
        finally:
            for t in tmps:
                t.unlink(missing_ok=True)
    return {"ok": True, "stored": f"{device}/{meta['kind']}/{name}"}
