"""Optional AI help from a model server the user runs: explain a failed job, advise on a round.

Off unless turned on. It talks to any OpenAI-compatible chat endpoint (llama.cpp's llama-server,
Ollama's /v1, LM Studio, vLLM): POST <base>/v1/chat/completions. Settings are app-wide, in
<data home>/settings.json under "assistant" (file mode 0600: it may hold an API key, which the web
API never returns in full).

What gets sent: log excerpts, project settings (paths shortened to ~), the hardware summary and
result numbers. Never audio. The model only gives advice: nothing here changes a project.

The quick checks (QUICK_RULES) need no model: they spot the commonest failures in the log and are
shown whether or not the assistant is on.

The prompt builders (failure_messages, advice_numbers, advice_messages, ...) are pure functions of
their inputs, so they can be tested without a project or a model.
"""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path

from . import assets

DEFAULTS = {
    "enabled": False,
    "base_url": "http://127.0.0.1:8080",
    "model": "",  # empty: the server's default (llama-server serves one model anyway)
    "api_key": "",
    "timeout": 120,  # seconds; a local model on a busy machine can take a while
}
LOG_LINES = 150
MAX_LINE = 400  # characters kept of one log line (progress bars run to thousands)


class AssistantError(RuntimeError):
    """Something the person can act on: the model server is down, refused, timed out..."""


# -- settings ------------------------------------------------------------------------------------
def settings_path() -> Path:
    return assets.home() / "settings.json"


def _read_all() -> dict:
    f = settings_path()
    try:
        data = json.loads(f.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def load_settings() -> dict:
    s = _read_all().get("assistant") or {}
    return {**DEFAULTS, **{k: v for k, v in s.items() if k in DEFAULTS}}


def save_settings(changes: dict) -> dict:
    """Merge `changes` into the saved settings. An absent api_key keeps the saved one; "" clears it."""
    new = {}
    if "enabled" in changes:
        new["enabled"] = bool(changes["enabled"])
    if "base_url" in changes:
        url = str(changes["base_url"] or "").strip().rstrip("/")
        if not re.match(r"^https?://[^/\s]+", url):
            raise ValueError("the server address must start with http:// or https://")
        new["base_url"] = url
    if "model" in changes:
        new["model"] = str(changes["model"] or "").strip()
    if "api_key" in changes and changes["api_key"] is not None:
        new["api_key"] = str(changes["api_key"]).strip()
    if "timeout" in changes:
        try:
            new["timeout"] = max(5, min(900, int(float(changes["timeout"]))))
        except (TypeError, ValueError):
            raise ValueError("the timeout must be a number of seconds") from None
    out = {}

    def merge(data: dict) -> None:  # onto what is saved now, under the lock: no change is lost
        old = data.get("assistant") or {}
        out.update({**DEFAULTS, **{k: v for k, v in old.items() if k in DEFAULTS}, **new})
        data["assistant"] = out

    update_all(merge)
    return out


_settings_lock = threading.Lock()


def update_all(change) -> None:
    """Read settings.json, apply change(data) and write it back atomically, readable by its owner only
    (it may hold the API key). One writer at a time, so the AI settings and the language switch
    don't overwrite each other."""
    import fcntl

    f = settings_path()
    f.parent.mkdir(parents=True, exist_ok=True)
    # the thread lock for the web server's threads, the file lock for the CLI writing at the same time
    lock_fd = os.open(f.with_name(f.name + ".lock"), os.O_WRONLY | os.O_CREAT, 0o600)
    with _settings_lock, os.fdopen(lock_fd, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = _read_all()
        change(data)
        tmp = f.with_suffix(".tmp")
        tmp.unlink(missing_ok=True)  # a stale one may have looser permissions
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as out:
            json.dump(data, out, indent=1)
        tmp.replace(f)


def public(s: dict) -> dict:
    """Settings for the web page: the API key only as a hint of its last characters."""
    key = s.get("api_key") or ""
    return {
        **{k: v for k, v in s.items() if k != "api_key"},
        "has_key": bool(key),
        "key_hint": f"…{key[-4:]}" if len(key) >= 8 else ("set" if key else ""),
    }


# -- talking to the model server -----------------------------------------------------------------
def endpoint(base_url: str) -> str:
    """The chat completions URL for a base address, with or without /v1 at the end."""
    base = base_url.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if not base.endswith("/v1"):
        base += "/v1"
    return base + "/chat/completions"


_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def chat(messages: list[dict], s: dict | None = None) -> dict:
    """One chat completion. Returns {"answer", "model"}; raises AssistantError with a plain message."""
    s = s or load_settings()
    if not s.get("enabled"):
        raise AssistantError("the AI assistant is off: turn it on under Hardware")
    url = endpoint(s["base_url"])
    body: dict = {"messages": messages, "temperature": 0.2, "stream": False}
    if s.get("model"):
        body["model"] = s["model"]
    headers = {"Content-Type": "application/json"}
    if s.get("api_key"):
        headers["Authorization"] = f"Bearer {s['api_key']}"
    req = urllib.request.Request(url, json.dumps(body).encode(), headers, method="POST")
    timeout = float(s.get("timeout") or DEFAULTS["timeout"])
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300].strip()
        hint = {
            401: " (check the API key)",
            403: " (check the API key)",
            404: " (check the address; it should be the server's base, e.g. http://127.0.0.1:8080)",
        }.get(e.code, "")
        raise AssistantError(
            f"the model server at {s['base_url']} answered HTTP {e.code}{hint}: {detail}"
        ) from None
    except TimeoutError:
        raise AssistantError(
            f"no answer from {s['base_url']} within {timeout:.0f} s: the model may still be "
            "loading or too slow; try again or raise the timeout"
        ) from None
    except urllib.error.URLError as e:
        reason = e.reason
        if isinstance(reason, TimeoutError):
            raise AssistantError(
                f"no answer from {s['base_url']} within {timeout:.0f} s: the model may still be "
                "loading or too slow; try again or raise the timeout"
            ) from None
        raise AssistantError(
            f"couldn't reach a model server at {s['base_url']} ({reason}). Is it running, "
            "and is the address right?"
        ) from None
    except OSError as e:
        raise AssistantError(f"couldn't talk to {s['base_url']}: {e}") from None
    try:
        data = json.loads(raw)
        msg = data["choices"][0]["message"]
        text = (msg.get("content") or "").strip()
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        raise AssistantError(
            f"{url} didn't answer like an OpenAI-compatible chat server: {raw[:200]!r}"
        ) from None
    text = _THINK.sub("", text).strip()  # reasoning models: keep the answer, not the musing
    if not text:
        raise AssistantError("the model sent back an empty answer; try again or another model")
    return {"answer": text, "model": data.get("model") or s.get("model") or ""}


TEST_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant. Answer in one short sentence."},
    {"role": "user", "content": "Say hello and name the model you are."},
]


# -- quick checks: no model needed -----------------------------------------------------------------
QUICK_RULES: list[tuple[re.Pattern, str, str]] = [
    (
        re.compile(
            r"CUDA out of memory|OutOfMemoryError|XPU out of memory|UR_RESULT_ERROR_OUT_OF_"
            r"(DEVICE|HOST)_MEMORY|out of memory on device|PI_ERROR_OUT_OF_RESOURCES|"
            r"RESOURCE_EXHAUSTED|failed to allocate memory",
            re.IGNORECASE,
        ),
        "The GPU ran out of memory",
        (
            "Close other programs that use the GPU (a local AI model, a game, another training run), "
            "then press Train to resume. If it happens again, start wakesmith with a smaller speech "
            "batch (WAKESMITH_TTS_BATCH=50 wakesmith serve) or train on the CPU (wakesmith serve --cpu)."
        ),
    ),
    (
        re.compile(r"No space left on device|Errno 28|disk quota exceeded", re.IGNORECASE),
        "The disk is full",
        (
            "Free some space (each project needs about 23 GB; delete wake words or old runs you no "
            "longer need), then press Train to resume."
        ),
    ),
    (
        re.compile(
            r"\bKilled\b|exit(?: code| status)?:? ?(?:-9|137)\b|signal 9\b|SIGKILL|"
            r"oom-kill|Out of memory: Killed process|\bMemoryError",
            re.IGNORECASE,
        ),
        "The system ran out of memory and stopped the job",
        (
            "Close other big programs (browsers with many tabs, a local AI model) and press Train to "
            "resume. microWakeWord training needs about 16 GB of free RAM."
        ),
    ),
    (
        re.compile(
            r"No such file or directory|FileNotFoundError|isn't downloaded|not downloaded|"
            r"HTTP Error 40[34]|HTTPError|URLError|Connection (?:refused|reset)|"
            r"Temporary failure in name resolution|checksum|sha256 mismatch",
            re.IGNORECASE,
        ),
        "A file or download is missing",
        (
            "Check the log line for the file it wanted. For downloads, check the internet connection "
            "and run wakesmith download (or wakesmith doctor) again; for your own recordings, check "
            "the folders under Real voices and that every file listed in HELDOUT.txt exists."
        ),
    ),
    (
        re.compile(
            r"UR_RESULT_ERROR_DEVICE_LOST|DEVICE_LOST|ZE_RESULT_ERROR|level_zero|"
            r"CUDA error|cudaErrorLaunchFailure|illegal memory access|no kernel image",
            re.IGNORECASE,
        ),
        "The GPU runtime failed",
        (
            "A crashed GPU job can leave the driver stuck: reboot, then run wakesmith doctor. On Intel "
            "GPUs the compute runtime must be 26.35 or newer. To go on without the GPU, start "
            "wakesmith with --cpu."
        ),
    ),
    (
        re.compile(
            r"max\(\) (?:iterable argument|arg) is (?:an )?empty|no held-out|lists \d+ files",
            re.IGNORECASE,
        ),
        "Too few or missing held-out recordings",
        (
            'The held-out test needs real recordings kept aside: mark some takes as "test" under '
            "Real voices (or list them in HELDOUT.txt), and check every file listed there exists."
        ),
    ),
]


def quick_causes(text: str) -> list[dict]:
    """Known failure patterns found in `text`, most specific first, each with what to try."""
    out = []
    for rx, title, advice in QUICK_RULES:
        m = rx.search(text)
        if m:
            line = next((ln for ln in text.splitlines() if rx.search(ln)), m.group(0))
            out.append({"title": title, "advice": advice, "evidence": line.strip()[:MAX_LINE]})
    return out


# -- shaping what gets sent ------------------------------------------------------------------------
_SECRET = re.compile(
    r"\b(?:hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"(?:token|api[_-]?key|password|secret)\s*[=:]\s*\S+)",
    re.IGNORECASE,
)


def scrub(text: str, home: str | None = None) -> str:
    """Shorten the home folder to ~ and blank anything that looks like a token or password."""
    home = home if home is not None else str(Path.home())
    if home and home != "/":
        text = text.replace(home, "~")
    return _SECRET.sub("[redacted]", text)


def tail_lines(text: str, n: int = LOG_LINES) -> list[str]:
    """The last n meaningful lines: \\r progress redraws collapsed to their last state, long lines cut."""
    lines = []
    for ln in text.splitlines():
        ln = ln.rsplit("\r", 1)[-1].rstrip()
        if ln:
            lines.append(ln if len(ln) <= MAX_LINE else ln[:MAX_LINE] + " …")
    return lines[-n:]


def referenced_logs(text: str) -> list[str]:
    """Names of log files under logs/ that an error points to ("see logs/generate-negative.log")."""
    names = re.findall(r"logs/([A-Za-z0-9_.-]+\.log)\b", text)
    return list(dict.fromkeys(names))


def last_section(text: str, marker: str) -> str:
    """What follows the last line matching the regex `marker` (that line included): the latest
    run in a log that runs append to. All of `text` when no line matches."""
    starts = [m.start() for m in re.finditer(rf"(?m)^.*{marker}", text)]
    return text[starts[-1] :] if starts else text


def _read_tail(f: Path, nbytes: int = 300_000) -> str:
    try:
        with open(f, "rb") as fh:
            fh.seek(max(0, f.stat().st_size - nbytes))
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def failure_logs(logs: Path, error: str = "") -> list[str]:
    """The log excerpt for a failed job, at most LOG_LINES lines: the logs an error points to
    (their tails first), then the latest run of the job's own log."""
    out: list[str] = []
    train = last_section(_read_tail(logs / "train.log"), r"\d\d:\d\d:\d\d hardware: ")
    job_f = logs / "job.log"
    job = last_section(_read_tail(job_f), r"===== \d{4}-") if job_f.exists() else ""
    tf = logs / "train.log"
    # a job started from the web UI logs everything (tracebacks included) to job.log; from the
    # command line only train.log is written
    fresh = job and (not tf.exists() or job_f.stat().st_mtime >= tf.stat().st_mtime - 120)
    main_name, main = ("job.log", job) if fresh else ("train.log", train)
    for name in referenced_logs(error + "\n" + main)[:2]:
        f = logs / name
        if name != main_name and f.is_file():
            lines = tail_lines(_read_tail(f, 60_000), 40)
            out += [f"--- logs/{name} (last {len(lines)} lines)", *lines]
    lines = tail_lines(main, max(60, LOG_LINES - len(out) - 1))
    out += [f"--- logs/{main_name} (last {len(lines)} lines)", *lines]
    return out[-LOG_LINES:]


SETTING_KEYS = {
    "samples": None,
    "real": ("copies", "negative_copies", "sampling_weight"),
    "commonvoice": ("enabled", "train"),
    "microwakeword": (
        "enabled",
        "steps",
        "finetune_steps",
        "candidates",
        "partial_weight",
        "adversarial_weight",
        "cutoff",
        "frontend",
    ),
    "openwakeword": ("enabled", "steps", "threshold"),
}


def project_summary(data: dict) -> dict:
    """The settings that matter for a diagnosis, from project.yaml's data. Folder paths become
    yes/no; the sound-alikes become a count."""
    out: dict = {"phrase": data.get("phrase", ""), "sound_alikes": len(data.get("negatives") or [])}
    for sect, keys in SETTING_KEYS.items():
        v = data.get(sect) or {}
        out[sect] = dict(v) if keys is None else {k: v[k] for k in keys if k in v}
    real = data.get("real") or {}
    out["real"]["phrase_recordings"] = bool(real.get("positives"))
    out["real"]["not_phrase_recordings"] = bool(real.get("negatives"))
    out["data_profile"] = data.get("data_profile", "")
    if data.get("hardware"):
        out["hardware_overrides"] = data["hardware"]
    return out


def hardware_summary(hw: dict) -> dict:
    """The hardware dict from the server's /api/hardware, cut to what helps a diagnosis."""
    keys = (
        "accel",
        "cpus",
        "ram_gb",
        "ram_free_gb",
        "disk_free_gb",
        "mww_device",
        "tts_batch",
        "problem",
        "disk_warning",
        "ram_warning",
    )
    out = {k: hw[k] for k in keys if hw.get(k) not in (None, "")}
    out["gpus"] = [f"{g.get('name')} ({g.get('memory_gb')} GB)" for g in hw.get("gpus") or []]
    return out


FAILURE_SYSTEM = """You help people who use wakesmith, a local app that trains wake-word models \
(microWakeWord, openWakeWord). A training job failed; explain why in plain words for \
someone who is not an expert, and say what to try.

wakesmith's stages, in order: generate (synthetic speech of the phrase and sound-alikes, on the \
GPU), real (the person's own recordings: training takes trimmed, held-out takes kept aside), \
commonvoice (phrase cut out of Mozilla Common Voice takes), mww-features / mww-train \
(microWakeWord: augmented features, then training; about 16 GB RAM), oww-features / oww-train \
(openWakeWord), test (scores held-out recordings), export (writes the model files). Train resumes from the failed stage; "Retrain from recordings" starts \
at real.

Common causes: GPU out of memory (other programs on the GPU, batch too big), system out of memory \
(the job is killed: exit -9/137, "Killed"), GPU runtime problems (Intel compute runtime older than \
26.35, a wedged driver after a crash: reboot, wakesmith doctor, or --cpu), missing downloads \
(wakesmith download), disk full (about 23 GB per project), recording folders that moved or a \
HELDOUT.txt naming missing files, too few held-out recordings for the test.

"How the job ended" comes from the process that ran it. A job that ended with no error line in \
the log was most often stopped by the person or by a restart: only call it out of memory when the \
log or the job's end shows it (Killed, exit -9 or 137, signal 9, out of memory). Never guess a \
cause the log doesn't show.

The only fixes you may name are these: close other programs, press Train to resume, reboot, \
wakesmith doctor, wakesmith download, starting wakesmith with --cpu, the WAKESMITH_TTS_BATCH=50 \
environment variable (smaller speech batches), freeing disk space, fixing recording folders or \
HELDOUT.txt under Real voices. Don't invent other settings, files or commands.

Answer in this shape, short:
Cause: one or two sentences, quoting the log line that shows it. If nothing in the log shows a \
cause, say "The log doesn't show why it stopped" and what that usually means.
What to try: up to four numbered steps, the most likely fix first."""


_ERROR_LINE = re.compile(
    r"Traceback|\bError\b|\w+Error:|Exception|FAILED|fatal|\bKilled\b|exit(?: code)? -?\d+|signal \d+|"
    r"out of memory|abort",
    re.IGNORECASE,
)
NO_CAUSE = (
    "Cause: The log doesn't show why it stopped: there is no error line, so the run most likely "
    "was stopped (the Stop button, a closed terminal) or the machine restarted.\n"
    "What to try:\n1. Press Train: it goes on from this stage.\n"
    "2. If it stops again on its own, look at the log's last lines then (or ask again): a real failure "
    "leaves an error there."
)


def shows_a_cause(error: str, log: list[str], quick: list[dict], ended: str) -> bool:
    """Whether there is anything for a model to explain: an error line, a matched check or a recorded
    failure. wakesmith's own "stopped before it finished" note doesn't count."""
    from . import jobs

    if quick or ended.startswith("failed"):
        return True
    return bool(_ERROR_LINE.search(error.replace(jobs.STOPPED, "") + "\n" + "\n".join(log)))


def failure_messages(
    summary: dict,
    stage: str,
    error: str,
    log: list[str],
    hardware: dict,
    quick: list[dict] | None = None,
    home: str | None = None,
    ended: str = "",
) -> list[dict]:
    """The chat for "Explain this failure". Pure: everything comes in as arguments."""
    parts = [
        f"Failed stage: {stage or 'unknown'}",
        f"Error: {error or '(none recorded)'}",
        f"How the job ended: {ended or 'not recorded (started by an older wakesmith)'}",
        "Project settings:\n" + json.dumps(summary, indent=1),
        "Hardware:\n" + json.dumps(hardware, indent=1),
    ]
    if quick:
        parts.append(
            "Patterns wakesmith's own checks spotted (may be right or wrong):\n"
            + "\n".join(f"- {q['title']}: {q['evidence']}" for q in quick)
        )
    parts.append(f"Last {len(log)} log lines:\n" + "\n".join(log))
    return [
        {"role": "system", "content": FAILURE_SYSTEM},
        {"role": "user", "content": scrub("\n\n".join(parts), home)},
    ]


# -- round advisor -----------------------------------------------------------------------------------
MODEL_LABELS = {
    "mww": "microWakeWord",
    "oww": "openWakeWord",
    "mww-cv": "microWakeWord on Common Voice speakers",
    "oww-cv": "openWakeWord on Common Voice speakers",
}


def advice_numbers(
    phrase: str,
    results: dict,
    chosen: dict,
    candidates: dict | None = None,
    devices: dict | None = None,
    runs: list[dict] | None = None,
) -> dict:
    """Everything the advisor is told, as one dict (also shown to the person next to the answer).

    results: {kind: {"table": [{threshold, caught, false}], "positives", "negatives"}}
    chosen: {"mww": cutoff, "oww": threshold}; candidates: pick.json;
    devices: rounds.summary(); runs: runs.listing() (newest first)."""
    out: dict = {"wake_word": phrase, "models": {}}
    for kind, r in results.items():
        out["models"][kind] = {
            "model": MODEL_LABELS.get(kind, kind),
            "held_out_phrase_clips": r["positives"],
            "held_out_other_clips": r["negatives"],
            "chosen_cutoff": chosen.get(kind.split("-")[0]),
            "by_cutoff": [
                {"cutoff": row["threshold"], "caught": row["caught"], "false": row["false"]}
                for row in r["table"]
            ],
        }
    if candidates and candidates.get("candidates"):
        out["mww_candidates"] = {
            "kept": candidates.get("picked"),
            "why": candidates.get("why", ""),
            "runs": [
                {
                    k: c[k]
                    for k in (
                        "n",
                        "validation",
                        "cutoff",
                        "caught",
                        "positives",
                        "false",
                        "negatives",
                        "failed",
                    )
                    if k in c
                }
                for c in candidates["candidates"]
            ],
        }
    if devices and devices.get("total"):
        out["device_clips"] = {
            k: devices[k] for k in ("total", "labels", "unlabelled", "heldout") if k in devices
        }
    if runs and len(runs) > 1:
        past = []
        for run in runs[:5]:
            row = {"run": run.get("label") or run.get("id"), "id": run.get("id")}
            for kind, r in (run.get("results") or {}).items():
                cut = chosen.get(kind.split("-")[0])
                hit = next(
                    (x for x in r["table"] if cut is not None and abs(x["threshold"] - cut) < 1e-9),
                    None,
                )
                if hit:
                    row[kind] = (
                        f"{hit['caught']}/{r['positives']} caught, "
                        f"{hit['false']}/{r['negatives']} false at {cut}"
                    )
            if len(row) > 2:
                past.append(row)
        if len(past) > 1:
            out["recent_runs_at_chosen_cutoffs"] = past
    return out


ADVICE_SYSTEM = """You advise someone training their own wake word with wakesmith. They give \
you the held-out test numbers of their latest training round. "caught" is how many of their \
held-out real recordings of the phrase the model fired on (higher is better); "false" is how \
many of the other held-out clips (talk, TV, sound-alikes, real false triggers from their devices) \
it fired on (lower is better). The cutoff is the score a sound must reach to count; higher means \
fewer false triggers and more misses.

Recommend, briefly and in plain words:
1. Which cutoff to use for each model, and the trade-off at the next cutoff up and down.
2. Whether this model looks ready to put on a device, or what makes it doubtful.
3. What data to add next round (more held-out recordings, other voices, far-away takes, more \
real false triggers, etc.).

Rules: quote the exact numbers you rely on (e.g. "41/45 caught, 3/3990 false at 0.70"). When \
there are few held-out clips (under about 30 of the phrase, or under a few hundred others), say \
the numbers are too thin to be sure and that one clip changes the picture. Don't compare runs \
tested on different clip counts as if they were the same test. This is advice only: never claim \
to have changed anything."""


def advice_messages(numbers: dict) -> list[dict]:
    return [
        {"role": "system", "content": ADVICE_SYSTEM},
        {"role": "user", "content": "The numbers:\n" + json.dumps(numbers, indent=1)},
    ]
