"""Record a whole room in one go: one long phone recording, split into takes for every wake word.

The phone lies next to the device (so the takes sound like what the device hears) while someone walks
around the room and follows a script: quiet first (room sound), a few claps (the room's echo), then
each wake word a few times from each spot. The page uploads the audio in pieces as it records, so
nothing is lost if the page dies. After Stop:

1. split(): finds the loud stretches (frame loudness over an adaptive noise floor), tells claps
   (short, sudden, fast decay) from speech, transcribes the speech with Whisper (rounds._Whisper)
   and labels each stretch: a wake word's project, a spot marker ("across the room": later takes
   inherit it as their condition), other speech, or unclear. Long quiet stretches are room sound.
2. The review page shows the guesses; the person changes what's wrong.
3. send(): each wake-word take goes into its own project's recordings (recordings.add, every
   fourth one kept aside to test, as on the Record page); room sound, cut into 3-5 s pieces, and
   other speech go to every included project as not-the-phrase clips. A wake word is never sent
   as a not-the-phrase clip to another wake word's project. Claps are cut out as room impulse
   responses into <data>/rooms/<room>/claps/, which microWakeWord's room-echo augmentation uses
   when a project turns on `microwakeword.house_rirs`. What was sent is remembered (sent.json), so
   sending twice adds nothing twice, also after splitting again.

<data>/sessions/<id>/
  session.json    room, speakers, wake words (project, phrase), spots, times, device
  chunks/         the uploaded pieces (16 kHz mono 16-bit PCM), joined into audio.wav at Stop
  audio.wav       the whole recording
  segments.json   what split() found, with the person's changes
  sent.json       what went where
  split.json      the split job's state; split.log its output
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from . import assets, recordings
from . import project as projects

SR = 16000
HOP = 160  # 10 ms loudness frames
MERGE_GAP_S = 0.4  # loud stretches closer than this are one segment
PAD_S = 0.25  # kept before and after each segment
SPLIT_PAUSE_S = 0.25  # a pause this long inside a segment can separate repeated phrases
ROOM_MIN_S = 3.0  # quiet stretches at least this long are room sound
ROOM_PIECE_S = (3.0, 5.0)  # room sound is sent in pieces of about this length
CLAP_MAX_S = 0.3  # a clap's loud core; its echo tail may last up to CLAP_TAIL_S
CLAP_TAIL_S = 1.5
NEAR_TAKE_S = 4.0  # speech this close to a wake-word take is "unclear", not "other speech"
CLAP_BEFORE_S, CLAP_AFTER_S = 0.02, 1.0  # a clap's impulse response: just before onset to 1 s after
HELD_EVERY = 4  # every fourth take of a project kept aside to test (as on the Record page)
MAX_CHUNK_S = 60
SPOTS = [  # the script's spots and the words that mark them when said out loud
    ("close to the device", ["close"]),
    ("across the room", ["across"]),
    ("doorway", ["doorway", "door"]),
    ("next room, calling out", ["next room", "other room"]),
    ("quietly", ["quiet", "quietly", "softly"]),
    ("TV or music on", ["tv", "television", "music", "telly"]),
]
# Whisper hears a German session in German: its spot words too (only there: "nah" is English as well)
SPOT_WORDS = {
    "de": {
        "close to the device": ["nah", "nahe", "dicht"],
        "across the room": ["gegenüber", "anderen ende"],
        "doorway": ["tür", "türrahmen"],
        "next room, calling out": ["nebenzimmer", "nebenraum"],
        "quietly": ["leise"],
        "TV or music on": ["fernseher", "fernsehen", "musik"],
    }
}
FIXED_LABELS = ("spot", "other", "unclear", "room", "clap", "skip")
_ID = re.compile(r"^[0-9]{8}-[0-9]{6}(-[a-z0-9_]+)?$")
_WORD = re.compile(r"[a-zäöüß']+")


# -- where things live ----------------------------------------------------------------------------
def home() -> Path:
    return assets.home() / "sessions"


def rooms_home() -> Path:
    """House room echoes: <data>/rooms/<room>/claps/*.wav (microwakeword.house_rirs)."""
    return assets.home() / "rooms"


def house_rirs() -> list[Path]:
    d = rooms_home()
    return sorted(d.glob("*/claps/*.wav")) if d.is_dir() else []


def folder(sid: str) -> Path:
    if not _ID.match(sid or ""):
        raise ValueError("not a session id")
    d = home() / sid
    if not (d / "session.json").exists():
        raise FileNotFoundError(f"no session {sid}")
    return d


def _read(f: Path, default):
    try:
        return json.loads(f.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def _write(f: Path, data) -> None:
    tmp = f.with_name(f.name + ".part")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, f)


def load(sid: str) -> dict:
    return _read(folder(sid) / "session.json", {})


def _save(sid: str, s: dict) -> None:
    _write(folder(sid) / "session.json", s)


def segments(sid: str) -> list[dict]:
    return _read(folder(sid) / "segments.json", [])


def _save_segments(sid: str, segs: list[dict]) -> None:
    _write(folder(sid) / "segments.json", segs)


def sent(sid: str) -> list[dict]:
    return _read(folder(sid) / "sent.json", [])


# -- wake words and rooms to offer ----------------------------------------------------------------
def _worked_on(p: projects.Project) -> float:
    return max(
        (f.stat().st_mtime for f in (p.path("state.json"), p.path("project.yaml")) if f.exists()),
        default=0.0,
    )


def choices() -> list[dict]:
    """Every project, with `default` on one per distinct phrase: the newest with a live model, else
    the newest worked on."""
    from . import models

    home_ = projects.projects_home()
    out = []
    for f in projects.project_files(home_):
        try:
            p = projects.load(f.parent)
        except (OSError, ValueError, TypeError):
            continue
        try:
            live = bool(models.live(p))
        except Exception:  # noqa: BLE001 - a broken models.json only loses the preference
            live = False
        out.append(
            {
                "name": p.name,
                "label": p.title,
                "phrase": p.phrase,
                "live": live,
                "t": _worked_on(p),
            }
        )
    best: dict[str, dict] = {}
    for c in out:
        key = " ".join(_words(c["phrase"]))
        cur = best.get(key)
        if cur is None or (c["live"], c["t"]) > (cur["live"], cur["t"]):
            best[key] = c
    for c in out:
        c["default"] = best.get(" ".join(_words(c["phrase"]))) is c
    return out


def listing() -> list[dict]:
    d = home()
    out = []
    for f in sorted(d.glob("*/session.json"), reverse=True) if d.is_dir() else []:
        s = _read(f, None)
        if not isinstance(s, dict):
            continue
        segs = _read(f.parent / "segments.json", [])
        out.append(
            {
                **{k: s.get(k) for k in ("id", "room", "speakers", "started", "stopped", "status")},
                "seconds": s.get("seconds", 0),
                "phrases": [x["phrase"] for x in s.get("phrases", [])],
                "segments": len(segs),
                "split": split_status(s["id"])["status"] if s.get("id") else None,
            }
        )
    return out


def rooms() -> list[str]:
    """Room names used before, newest first (suggestions for the room field)."""
    seen = []
    for s in listing():
        r = (s.get("room") or "").strip()
        if r and r.lower() not in [x.lower() for x in seen]:
            seen.append(r)
    return seen


# -- recording --------------------------------------------------------------------------------------
def create(
    room: str,
    speakers: list[str],
    project_names: list[str],
    spots: list[str] | None = None,
    device: str = "",
) -> dict:
    room = (room or "").strip()[:60]
    if not room:
        raise ValueError("name the room")
    if not project_names:
        raise ValueError("pick at least one wake word")
    phrases, families = [], set()
    for n in project_names:
        p = projects.load(n)
        if p.family in families:  # two versions of one word would split its takes between them
            raise ValueError(
                f"pick one version of {p.display}: takes of a phrase go to one version"
            )
        families.add(p.family)
        phrases.append(
            {
                "project": p.name,
                "phrase": p.phrase,
                "label": p.title,
                "spellings": p.heard_forms[1:],
                "language": p.language,
            }
        )
    if len({x["language"] for x in phrases}) > 1:  # Whisper listens for one language per session
        raise ValueError("pick wake words of one language: Whisper listens for one per session")
    t = time.localtime()
    base = time.strftime("%Y%m%d-%H%M%S", t) + "-" + projects.slug(room)[:30]
    sid, n = base, 1
    home().mkdir(parents=True, exist_ok=True)
    while True:  # two sessions started in the same second
        try:
            (home() / sid).mkdir()
            break
        except FileExistsError:
            n += 1
            sid = f"{base}_{n}"
    d = home() / sid
    (d / "chunks").mkdir()
    s = {
        "id": sid,
        "room": room,
        "speakers": [x.strip()[:40] for x in speakers if x.strip()],
        "phrases": phrases,
        "spots": [x.strip() for x in (spots or [n for n, _ in SPOTS]) if x.strip()],
        "started": time.strftime("%Y-%m-%d %H:%M:%S", t),
        "stopped": None,
        "status": "recording",
        "device": device[:300],
        "seconds": 0.0,
    }
    _write(d / "session.json", s)
    return s


def add_chunk(sid: str, seq: int, data: bytes) -> dict:
    """One uploaded piece: 16 kHz mono WAV, or raw 16-bit little-endian PCM. Sending the same piece
    again replaces it (a retried upload)."""
    d = folder(sid)
    if load(sid).get("status") != "recording":
        raise PermissionError("this session is finished")
    if not 0 <= seq < 1_000_000:
        raise ValueError("bad piece number")
    if data[:4] == b"RIFF":
        a, sr = sf.read(io.BytesIO(data), dtype="int16", always_2d=True)
        if sr != SR:
            raise ValueError(f"pieces must be {SR} Hz, got {sr}")
        pcm = a[:, 0].tobytes()
    else:
        if len(data) % 2:
            raise ValueError("raw pieces are 16-bit samples")
        pcm = data
    if len(pcm) > MAX_CHUNK_S * SR * 2:
        raise ValueError(f"a piece can be at most {MAX_CHUNK_S} s")
    f = d / "chunks" / f"{seq:06d}.pcm"
    tmp = f.with_suffix(".part")
    tmp.write_bytes(pcm)
    os.replace(tmp, f)
    return {"seq": seq, "seconds": round(len(pcm) / 2 / SR, 2)}


def finish(sid: str) -> dict:
    """Join the pieces in order into audio.wav (also for a session whose page died: nothing else is
    needed). Missing piece numbers are noted."""
    d = folder(sid)
    s = load(sid)
    parts = sorted((d / "chunks").glob("*.pcm"))
    seqs = [int(f.stem) for f in parts]
    missing = sorted(set(range(max(seqs) + 1)) - set(seqs)) if seqs else []
    audio = np.concatenate(
        [np.frombuffer(f.read_bytes(), dtype="<i2") for f in parts] or [np.zeros(0, "<i2")]
    )
    sf.write(d / "audio.wav", audio, SR, subtype="PCM_16")
    s.update(
        status="stopped",
        stopped=s.get("stopped") or time.strftime("%Y-%m-%d %H:%M:%S"),
        seconds=round(len(audio) / SR, 2),
        missing=missing,
        pieces=len(parts),
    )
    _save(sid, s)
    return s


def delete(sid: str) -> None:
    import shutil

    shutil.rmtree(folder(sid))


# -- finding segments -------------------------------------------------------------------------------
def _frame_rms(a: np.ndarray, hop: int = HOP) -> np.ndarray:
    n = len(a) // hop
    if not n:
        return np.zeros(0)
    return np.sqrt((a[: n * hop].astype(np.float64).reshape(n, hop) ** 2).mean(axis=1) + 1e-12)


def noise_floor(rms: np.ndarray, block: int = 50, reach: int = 6) -> np.ndarray:
    """Per frame: the lowest 20th-percentile loudness of the half-second blocks within ~3 s. Follows
    a room that gets louder (TV on) without counting speech as the room."""
    if not len(rms):
        return rms
    nb = -(-len(rms) // block)
    lows = np.array([np.percentile(rms[i * block : (i + 1) * block], 20) for i in range(nb)])
    near = np.array([lows[max(0, i - reach) : i + reach + 1].min() for i in range(nb)])
    return np.repeat(near, block)[: len(rms)]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of each run of True."""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(m))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def loud_regions(a: np.ndarray, factor: float = 3.0, floor_min: float = 3e-4) -> tuple:
    """Loud stretches as [start, end) frame ranges (gaps < MERGE_GAP_S merged), with the frame
    loudness and threshold."""
    rms = _frame_rms(a)
    thr = np.maximum(noise_floor(rms) * factor, floor_min)
    runs = [r for r in _runs(rms > thr) if r[1] - r[0] >= 3]  # >= 30 ms
    merged: list[list[int]] = []
    gap = int(MERGE_GAP_S * SR / HOP)
    for s, e in runs:
        if merged and s - merged[-1][1] < gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [tuple(r) for r in merged], rms, thr


def is_clap(a: np.ndarray, start: int, end: int, thr: float) -> bool:
    """A clap: a sudden onset (within 6 dB of the peak 10 ms after crossing the threshold), a fast decay (10 dB
    down 80 ms after the peak) and its loud core (within 20 dB of the peak) shorter than CLAP_MAX_S.
    The room's echo tail may run on longer; speech starts softer and holds its level."""
    if (end - start) * HOP / SR >= CLAP_TAIL_S:
        return False
    lo = max(0, start * HOP - int(0.02 * SR))
    seg = a[lo : min(len(a), end * HOP + int(0.1 * SR))]
    fine = _frame_rms(seg, 16)  # 1 ms
    if len(fine) < 90:
        return False
    above = np.flatnonzero(fine > thr)
    if not len(above):
        return False
    onset, peak = int(above[0]), int(np.argmax(fine))
    if int(np.flatnonzero(fine >= fine[peak] * 0.5)[0]) - onset > 10:  # within 6 dB of the peak
        return False
    after = fine[peak + 80] if peak + 80 < len(fine) else fine[-1]
    core = np.flatnonzero(fine > fine[peak] * 0.1)
    return after <= fine[peak] * 10 ** (-10 / 20) and (core[-1] - core[0]) / 1000 < CLAP_MAX_S


def find(a: np.ndarray) -> list[dict]:
    """Segments of a recording (float32, 16 kHz): speech and clap stretches padded by PAD_S, and
    room sound for the quiet stretches of at least ROOM_MIN_S between them."""
    regions, _, thr = loud_regions(a)
    total = len(a) / SR
    out = []
    for s, e in regions:
        t0, t1 = s * HOP / SR, e * HOP / SR
        clap = is_clap(a, s, e, float(thr[s]))
        seg = {
            "kind": "clap" if clap else "speech",
            "start": round(max(0.0, t0 - PAD_S), 3),
            "end": round(min(total, t1 + PAD_S), 3),
            "loud": [round(t0, 3), round(t1, 3)],
        }
        if clap:
            fine = _frame_rms(a[s * HOP : e * HOP], 16)
            above = np.flatnonzero(fine > thr[s])
            seg["onset"] = round(t0 + (int(above[0]) if len(above) else 0) * 16 / SR, 4)
        out.append(seg)
    for i in range(1, len(out)):  # padding of neighbors: meet halfway
        if out[i]["start"] < out[i - 1]["end"]:
            mid = round((out[i - 1]["loud"][1] + out[i]["loud"][0]) / 2, 3)
            out[i - 1]["end"], out[i]["start"] = mid, mid
    quiet, last = [], 0.0
    for seg in out + [{"start": total}]:
        if seg["start"] - last >= ROOM_MIN_S + 2 * PAD_S:
            quiet.append(
                {
                    "kind": "room",
                    "start": round(last + PAD_S, 3),
                    "end": round(seg["start"] - PAD_S, 3),
                }
            )
        last = max(last, seg.get("end", total))
    return sorted(out + quiet, key=lambda x: x["start"])


def pauses(rms: np.ndarray, thr: np.ndarray, seg: dict) -> list[tuple[float, float, float]]:
    """Pauses longer than SPLIT_PAUSE_S inside a segment's loud part (frame loudness and threshold
    from loud_regions): (start s, end s, length s)."""
    s, e = int(seg["loud"][0] * SR / HOP), int(seg["loud"][1] * SR / HOP)
    quiet = rms[s:e] <= thr[s:e]
    out = []
    for qs, qe in _runs(quiet):
        if qs == 0 or qe == e - s:
            continue
        if (qe - qs) * HOP / SR > SPLIT_PAUSE_S:
            out.append(((s + qs) * HOP / SR, (s + qe) * HOP / SR, (qe - qs) * HOP / SR))
    return out


# -- what was said ----------------------------------------------------------------------------------
def _words(text: str) -> list[str]:
    t = re.sub(r"\b([a-z])\.(?=[a-z]\b)", r"\1", (text or "").lower())  # t.v. -> tv
    return _WORD.findall(t.replace(".", " "))


def _find(seq: list, want: list[str]) -> int:
    for i in range(len(seq) - len(want) + 1):
        if seq[i : i + len(want)] == want:
            return i
    return -1


def match(text: str, phrases: list[dict]) -> tuple[dict[str, int], list[str]]:
    """How often each project's phrase (or extra spelling) is said, longest phrases first so "help"
    inside "hey jarvis" doesn't count for a "jarvis" project; and the words left over."""
    words: list = _words(text)
    forms = [
        (x["project"], _words(f))
        for x in phrases
        for f in [x["phrase"], *(x.get("spellings") or [])]
        if _words(f)
    ]
    counts: dict[str, int] = {}
    for proj, want in sorted(forms, key=lambda f: -len(f[1])):
        while (i := _find(words, want)) >= 0:
            words[i : i + len(want)] = [None] * len(want)
            counts[proj] = counts.get(proj, 0) + 1
    return counts, [w for w in words if w]


def spot_of(words: list[str], spots: list[str], lang: str = "en") -> str | None:
    """The spot a spot marker names ("across the room" -> across the room), or None. English words
    always count; a session in another language adds its own."""
    extra = SPOT_WORDS.get(lang, {})
    keys = {name: words_ + extra.get(name, []) for name, words_ in SPOTS}
    for name in spots:
        for k in keys.get(name, [name.lower()]):
            if _find(words, _words(k)) >= 0:
                return name
    return None


def label_text(text: str, s: dict) -> tuple[str, int, str | None, str]:
    """(label, times said, spot named, note) for a transcript."""
    counts, rest = match(text, s["phrases"])
    spot = spot_of(rest, s.get("spots", []), language(s))
    if len(counts) > 1:
        return "unclear", 0, spot, "more than one wake word in one stretch: listen and pick"
    if counts:
        ((proj, n),) = counts.items()
        return proj, n, spot, ""
    if spot:
        return "spot", 0, spot, ""
    phrase_words = {
        w
        for x in s["phrases"]
        for f in [x["phrase"], *(x.get("spellings") or [])]
        for w in _words(f)
    }
    if not rest:
        return "unclear", 0, None, "no words heard"
    if phrase_words & set(rest):
        return "unclear", 0, None, "part of a wake word"
    return "other", 0, None, ""


MAX_TOKENS = 64  # a stretch is a few words; noise can make Whisper repeat itself up to 448 tokens


class _Capped:
    """The device-clips Whisper (rounds._Whisper) with its output capped at MAX_TOKENS. A door or a
    tone can make Whisper repeat one word up to its limit, which took minutes per stretch on a CPU."""

    def __init__(self, w):
        self.w = w

    def __call__(self, paths: list[Path]) -> list[str]:
        w = self.w
        if w.fw is not None:
            out = []
            for f in paths:
                segs, _ = w.fw.transcribe(
                    str(f),
                    language=w.language,
                    beam_size=5,
                    condition_on_previous_text=False,
                    max_new_tokens=MAX_TOKENS,
                )
                out.append(" ".join(x.text.strip() for x in segs))
            return out
        audio = [sf.read(str(f), dtype="float32")[0] for f in paths]
        feats = w.proc(audio, sampling_rate=SR, return_tensors="pt").input_features
        with w.torch.inference_mode():
            ids = w.model.generate(
                feats.to(w.device, w.dtype),
                language=w.language,
                task="transcribe",
                num_beams=5,
                do_sample=False,
                max_new_tokens=MAX_TOKENS,
            )
        return [t.strip() for t in w.proc.batch_decode(ids, skip_special_tokens=True)]


def _transcriber(device: str, model: str = "turbo", language: str = "en"):
    from .rounds import _Whisper

    return _Capped(_Whisper(device, model, language))


def auto_device() -> str:
    """The GPU only while no wakesmith training job runs (it may need all of it), else the CPU."""
    from . import hardware, jobs, mww

    if jobs.running_all():
        return "cpu"
    return mww.train_device(hardware.detect_safe()[0].gpus)


def language(s: dict) -> str:
    """The language the session's words are in (sessions from before languages: English)."""
    return next((x.get("language", "en") for x in s.get("phrases", [])), "en")


def _clip(a: np.ndarray, t0: float, t1: float) -> np.ndarray:
    return a[max(0, round(t0 * SR)) : min(len(a), round(t1 * SR))]


def _fresh_spellings(s: dict) -> None:
    """The wake words' spellings as they are now: one added since the recording was made (Also written
    as) counts when it is split again."""
    for x in s.get("phrases", []):
        try:
            x["spellings"] = projects.load(x["project"]).heard_forms[1:]
        except (OSError, ValueError, TypeError):
            pass  # a deleted or unreadable wake word keeps what the session saved


def split(sid: str, device: str = "cpu", model: str = "turbo", transcribe=None, log=print) -> dict:
    """Find, transcribe and label the segments of a finished session (replaces segments.json; what
    was sent stays sent)."""
    d = folder(sid)
    s = load(sid)
    if not (d / "audio.wav").exists():
        s = finish(sid)
    a, sr = sf.read(d / "audio.wav", dtype="float32")
    if sr != SR:
        raise ValueError(f"audio.wav is {sr} Hz")
    found = find(a)
    _, rms, thr = loud_regions(a)
    log(
        f"{len(found)} stretches: "
        + ", ".join(f"{k} {sum(x['kind'] == k for x in found)}" for k in ("speech", "clap", "room"))
    )
    speech = [x for x in found if x["kind"] == "speech"]
    _fresh_spellings(s)
    texts: list[str] = []
    if speech:
        transcribe = transcribe or _transcriber(device, model, language(s))
        tmp = d / "tmp"
        tmp.mkdir(exist_ok=True)
        paths = []
        for i, x in enumerate(speech):
            f = tmp / f"seg{i:04d}.wav"
            sf.write(f, _clip(a, x["start"], x["end"]), SR, subtype="PCM_16")
            paths.append(f)
        for i in range(0, len(paths), 8):
            texts += transcribe(paths[i : i + 8])
            log(f"transcribed {min(i + 8, len(paths))}/{len(paths)}")
        for f in paths:
            f.unlink()
        tmp.rmdir()
    segs: list[dict] = []
    spot = ""
    speaker = s["speakers"][0] if len(s.get("speakers", [])) == 1 else ""
    it = iter(texts)
    for x in found:
        seg = {**x, "text": None, "label": x["kind"], "note": "", "spot": spot, "speaker": speaker}
        if x["kind"] == "speech":
            text = next(it)
            label, n, named, note = label_text(text, s)
            if named:
                spot = named
            seg.update(text=text, label=label, note=note, spot=spot)
            if n > 1:
                gaps = sorted(pauses(rms, thr, x), key=lambda g: -g[2])[: n - 1]
                if len(gaps) == n - 1:
                    cuts = sorted((g[0] + g[1]) / 2 for g in gaps)
                    edges = [x["start"], *cuts, x["end"]]
                    for j in range(n):
                        segs.append(
                            {
                                **seg,
                                "start": round(edges[j], 3),
                                "end": round(edges[j + 1], 3),
                                "note": f"part {j + 1} of {n} said in one go",
                            }
                        )
                    continue
                seg.update(
                    label="unclear",
                    note=f"said {n} times without clear pauses: leave it out or send it as is",
                )
        segs.append(seg)
    # Speech next to wake-word takes that matched nothing is more likely a take Whisper misheard than
    # a remark: sending it as a not-the-phrase clip would teach the model to ignore real calls.
    words = {x["project"] for x in s["phrases"]}
    sp = [x for x in segs if x["kind"] == "speech"]
    for k, seg in enumerate(sp):
        if seg["label"] != "other":
            continue
        near = [
            o
            for o in sp[max(0, k - 1) : k] + sp[k + 1 : k + 2]
            if o["label"] in words
            and min(abs(o["start"] - seg["end"]), abs(seg["start"] - o["end"])) < NEAR_TAKE_S
        ]
        if near:
            seg.update(
                label="unclear",
                note="next to wake-word takes: listen, Whisper may have misheard one",
            )
    for i, seg in enumerate(segs):
        seg["i"] = i
        seg["auto"] = seg["label"]
        seg.pop("loud", None)
    _save_segments(sid, segs)
    labels = {}
    for seg in segs:
        labels[seg["label"]] = labels.get(seg["label"], 0) + 1
    return {"segments": len(segs), "labels": labels}


def set_labels(sid: str, changes: dict) -> list[dict]:
    """changes: {i: {"label"?, "spot"?, "speaker"?}}."""
    s = load(sid)
    allowed = {x["project"] for x in s["phrases"]} | set(FIXED_LABELS)
    segs = segments(sid)
    by_i = {seg["i"]: seg for seg in segs}
    for k, ch in changes.items():
        seg = by_i.get(int(k))
        if seg is None:
            raise ValueError(f"no segment {k}")
        if "label" in ch:
            if ch["label"] not in allowed:
                raise ValueError(f"unknown label {ch['label']}")
            seg["label"] = ch["label"]
        for f in ("spot", "speaker"):
            if f in ch:
                seg[f] = str(ch[f] or "").strip()[:60]
    _save_segments(sid, segs)
    return segs


# -- the split job ----------------------------------------------------------------------------------
def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        if os.waitpid(pid, os.WNOHANG) != (0, 0):
            return False
    except ChildProcessError:
        pass
    return True


def split_status(sid: str) -> dict:
    st = _read(folder(sid) / "split.json", {"status": None})
    if st.get("status") == "running":
        pid = st.get("pid")
        if (pid and not _alive(pid)) or (not pid and time.time() - st.get("started", 0) > 30):
            st["status"] = "stopped"
    return st


def start_split(sid: str, device: str = "auto") -> dict:
    d = folder(sid)
    if split_status(sid).get("status") == "running":
        raise RuntimeError("this session is being split already")
    _write(d / "split.json", {"status": "running", "pid": None, "started": time.time()})
    with open(d / "split.log", "a") as f:
        f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} split\n")
        f.flush()
        from .jobs import detach

        detach([sys.executable, "-m", "wakesmith", "room", "split", sid, "--device", device], f)
    return split_status(sid)


def run_split(sid: str, device: str = "auto", model: str = "turbo") -> dict:
    """The split job's body (`wakesmith room split`): notes its state in split.json."""
    d = folder(sid)
    st = {"status": "running", "pid": os.getpid(), "started": time.time()}
    _write(d / "split.json", st)
    try:
        dev = auto_device() if device == "auto" else device
        print(f"Whisper on {dev}", flush=True)
        r = split(sid, dev, model, log=lambda m: print(m, flush=True))
    except Exception as e:
        _write(
            d / "split.json",
            {**st, "status": "failed", "error": str(e)[:500], "finished": time.time()},
        )
        raise
    _write(d / "split.json", {**st, "status": "done", "device": dev, "finished": time.time(), **r})
    return r


# -- sending ----------------------------------------------------------------------------------------
def _wav(a: np.ndarray) -> bytes:
    b = io.BytesIO()
    sf.write(b, a, SR, format="WAV", subtype="PCM_16")
    return b.getvalue()


def _is_sent(log: list[dict], seg: dict, target: str) -> bool:
    """Sent before to this target (a project, or "claps")? Matched by time overlap, so splitting
    again doesn't send the same audio twice."""
    for e in log:
        if e["target"] != target:
            continue
        both = min(e["end"], seg["end"]) - max(e["start"], seg["start"])
        if both > 0.5 * min(e["end"] - e["start"], seg["end"] - seg["start"]):
            return True
    return False


def room_pieces(t0: float, t1: float) -> list[tuple[float, float]]:
    """A quiet stretch cut into equal pieces of about 4 s (3-5 s where it is long enough)."""
    length = t1 - t0
    if length < ROOM_PIECE_S[0]:
        return []
    n = max(1, round(length / 4.0))
    while length / n > ROOM_PIECE_S[1]:
        n += 1
    step = length / n
    return [(round(t0 + i * step, 3), round(t0 + (i + 1) * step, 3)) for i in range(n)]


def clap_rir(a: np.ndarray, onset: float) -> np.ndarray:
    """A clap as a room impulse response: 20 ms before its onset to 1 s after, peak at 0.9."""
    x = _clip(a, onset - CLAP_BEFORE_S, onset + CLAP_AFTER_S).astype(np.float32)
    peak = float(np.abs(x).max()) if len(x) else 0.0
    return x * (0.9 / peak) if peak > 0 else x


def send(sid: str, busy=None) -> dict:
    """Send the labeled segments to the wake words (see the module doc). busy(project) -> True
    skips a project (a job running on it); it is sent next time."""
    from . import jobs

    busy = busy or (lambda p: jobs.status(p)["running"])
    d = folder(sid)
    s = load(sid)
    a, _ = sf.read(d / "audio.wav", dtype="float32")
    segs = segments(sid)
    log = sent(sid)
    names = [x["project"] for x in s["phrases"]]
    room = s["room"]
    added = {n: {"pos": 0, "neg": 0} for n in names}
    errors: dict[str, str] = {}
    notes: dict[str, str] = {}
    loaded: dict[str, projects.Project] = {}
    for n in names:
        try:
            p = projects.load(n)
        except (OSError, ValueError) as e:
            errors[n] = f"can't open: {e}"
            continue
        if busy(p):
            errors[n] = "a job is running on it: send again when it has finished"
            continue
        loaded[n] = p

    def put(n: str, kind: str, audio: np.ndarray, seg: dict, condition: str) -> bool:
        p = loaded[n]
        held = False
        if kind == "pos":
            held = (len(recordings.listing(p, "pos")["takes"]) + 1) % HELD_EVERY == 0
        try:
            r = recordings.add(
                p,
                kind,
                _wav(audio),
                condition,
                held,
                seg.get("speaker", "") if kind == "pos" else "",
                keep_unused=True,
            )
        except (PermissionError, ValueError) as e:
            errors.setdefault(n, str(e))
            return False
        log.append(
            {
                "target": n,
                "kind": kind,
                "start": seg["start"],
                "end": seg["end"],
                "file": r["file"],
                "at": time.time(),
            }
        )
        added[n][kind] += 1
        return True

    claps = 0
    for seg in segs:
        where = ", ".join(x for x in (room, seg.get("spot", "")) if x)
        lab = seg["label"]
        if lab in loaded:
            if not _is_sent(log, seg, lab):
                put(lab, "pos", _clip(a, seg["start"], seg["end"]), seg, where)
        elif lab == "room":
            for n in loaded:
                if _is_sent(log, seg, n):
                    continue
                for t0, t1 in room_pieces(seg["start"], seg["end"]):
                    put(
                        n,
                        "neg",
                        _clip(a, t0, t1),
                        {**seg, "start": t0, "end": t1},
                        f"{room}, room sound",
                    )
        elif lab == "other":
            for n in loaded:
                if not _is_sent(log, seg, n):
                    put(n, "neg", _clip(a, seg["start"], seg["end"]), seg, f"{where}, other speech")
        elif lab == "clap" and not _is_sent(log, seg, "claps"):
            onset = seg.get("onset", seg["start"] + PAD_S)
            out = rooms_home() / projects.slug(room) / "claps"
            out.mkdir(parents=True, exist_ok=True)
            f = out / f"{sid}_{seg['i']:03d}.wav"
            sf.write(f, clap_rir(a, onset), SR, subtype="FLOAT")
            log.append(
                {
                    "target": "claps",
                    "kind": "rir",
                    "start": seg["start"],
                    "end": seg["end"],
                    "file": str(f),
                    "at": time.time(),
                }
            )
            claps += 1
    _write(d / "sent.json", log)
    return {"added": added, "claps": claps, "errors": errors, "notes": notes}


def sent_marks(sid: str) -> dict[int, list[str]]:
    """For the review page: which targets each segment went to."""
    log = sent(sid)
    out = {}
    for seg in segments(sid):
        targets = sorted({e["target"] for e in log if _is_sent([e], seg, e["target"])})
        if targets:
            out[seg["i"]] = targets
    return out
