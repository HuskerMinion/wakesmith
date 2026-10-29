"""Everyday sentences as not-the-phrase clips (experimental; conversational.clips in project.yaml).

An AI assistant on an OpenAI-compatible server writes short everyday sentences that use words or sounds of
the wake phrase without saying the whole phrase in order ("can you hand Jarvis the dishes" for "hey
jarvis"). They are asked for in batches, cleaned (numbering stripped, too short or long ones dropped, repeats
and any sentence holding the phrase removed) and kept in <project>/conversational/sentences.txt, so a rerun
asks only for what is missing. Each sentence is then spoken once by the synthetic voices (every TTS model
takes an equal share) into <project>/conversational/clips, and microWakeWord trains on them as their own
negative set (conversational.weight); openWakeWord adds them to its negatives.

The endpoint comes from the project (conversational.llm_url, .model), WAKESMITH_LLM_URL, WAKESMITH_LLM_MODEL
and WAKESMITH_LLM_KEY, or else the AI assistant's settings (Hardware page) when it is turned on.
"""

from __future__ import annotations

import functools
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
import zlib
from collections.abc import Callable
from pathlib import Path

from . import languages
from .project import Project

DEFAULT_URL = "http://127.0.0.1:8080"
BATCH = 50  # sentences asked for per request
MIN_WORDS, MAX_WORDS = 4, 20  # asked for 6 to 14; a little slack, since models miscount
CACHE = "sentences.txt"


# --- minimal OpenAI-compatible client --------------------------------------------------------------


def chat(
    messages: list[dict],
    base_url: str = DEFAULT_URL,
    model: str = "",
    api_key: str = "",
    timeout: float = 300.0,
    temperature: float = 1.0,
) -> str:
    """The assistant's reply to `messages` (POST {base_url}/v1/chat/completions). Raises ConnectionError
    when the server can't be reached or answers with an error."""
    url = base_url.rstrip("/").removesuffix("/v1") + "/v1/chat/completions"
    body = {"messages": messages, "temperature": temperature, "stream": False}
    if model:
        body["model"] = model
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, json.dumps(body).encode(), headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            reply = json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise ConnectionError(f"the AI assistant at {base_url} answered {e.code}: {detail}") from e
    except (urllib.error.URLError, OSError, ValueError) as e:
        reason = getattr(e, "reason", e)
        raise ConnectionError(f"can't reach the AI assistant at {base_url}: {reason}") from e
    try:
        text = reply["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        raise ConnectionError(f"the AI assistant at {base_url} sent no reply text") from e
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)  # reasoning models' thinking


def endpoint(p: Project) -> tuple[str, str, str]:
    """(base URL, model, API key) for this project: the project's own settings, then the environment, then
    the app's AI assistant settings (Hardware page) when the assistant is turned on."""
    from . import assistant

    c = p.conversational
    a = assistant.load_settings()
    a = a if a.get("enabled") else {}
    return (
        c.llm_url or os.environ.get("WAKESMITH_LLM_URL") or a.get("base_url") or DEFAULT_URL,
        c.model or os.environ.get("WAKESMITH_LLM_MODEL") or a.get("model") or "",
        os.environ.get("WAKESMITH_LLM_KEY") or a.get("api_key") or "",
    )


# --- sentences ---------------------------------------------------------------------------------------


def normalize(text: str) -> str:
    """Lower case words only (umlauts kept): "Hey, Jarvis!" and "hey-jarvis" both give "hey jarvis"."""
    text = unicodedata.normalize("NFC", text)  # "ü" as u + a combining mark is still one letter
    return " ".join(re.sub(r"[\W_]+", " ", text.lower().replace("'", "")).split())


def says_phrase(sentence: str, phrases: list[str]) -> bool:
    """Does the sentence say any of the phrases, whole words in order?"""
    s = f" {normalize(sentence)} "
    return any(f" {normalize(ph)} " in s for ph in phrases if normalize(ph))


def parse(reply: str) -> list[str]:
    """Sentences from a reply: one per line, list numbering, bullets and quotes stripped."""
    out = []
    for line in reply.splitlines():
        line = (
            re.sub(r"^\s*(?:[-*•]|\d+[.)]|\(\d+\))\s*", "", line).strip().strip("\"'“”‘’").strip()
        )
        line = re.sub(
            r"[\"“”«»]", "", line
        )  # quotes inside: nothing for the voice to say, often unbalanced
        if line:
            out.append(line)
    return out


def keep(candidates: list[str], phrases: list[str], seen: set[str]) -> list[str]:
    """The usable new sentences: right length, not saying the phrase, not seen before (`seen` holds
    normalized text and is updated)."""
    out = []
    for s in candidates:
        key = normalize(s)
        if not MIN_WORDS <= len(key.split()) <= MAX_WORDS or key in seen or says_phrase(s, phrases):
            continue
        seen.add(key)
        out.append(s)
    return out


IN_LANGUAGE = {"de": "German"}  # what to ask for when the wake word isn't English


def messages(
    phrase: str, n: int, hints: list[str], avoid: list[str], language: str = "en"
) -> list[dict]:
    lang = f" in {IN_LANGUAGE[language]}" if language in IN_LANGUAGE else ""
    ask = (
        f"Write {n} different short everyday sentences{lang}, 6 to 14 words each, the kind people say at home "
        f'or on the phone. Each must use a word or sound from the phrase "{phrase}", or a word that '
        f'sounds close to it, but must never contain the whole phrase "{phrase}" with its words in order.'
    )
    if hints:
        ask += " Words that sound close: " + ", ".join(hints) + "."
    ask += " Vary the topics, speakers and sentence shapes. One sentence per line, no numbering, nothing else."
    if avoid:
        ask += "\nDon't repeat these:\n" + "\n".join(avoid)
    return [
        {
            "role": "system",
            "content": "You write natural example sentences for training a speech model.",
        },
        {"role": "user", "content": ask},
    ]


def cache_path(p: Project) -> Path:
    return p.path("conversational", CACHE)


def clip_dir(p: Project) -> Path:
    return p.path("conversational", "clips")


def collect(
    p: Project,
    n: int,
    ask: Callable[[list[dict]], str] | None = None,
    log: Callable[[str], None] = print,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[str]:
    """n sentences for the project: the cached ones first, then the assistant asked for the rest in
    batches of BATCH. Gives up after three requests in a row add nothing (the run goes on with fewer)."""
    phrases = [p.phrase, *p.positive_spellings]
    f = cache_path(p)
    seen: set[str] = set()
    have = keep(f.read_text().splitlines(), phrases, seen) if f.exists() else []
    if len(have) >= n:
        return have[:n]
    if ask is None:
        url, model, key = endpoint(p)
        ask = functools.partial(chat, base_url=url, model=model, api_key=key)

    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("".join(s + "\n" for s in have))
    hints = [x for x in p.negatives if not says_phrase(x, phrases)][:12]
    dry = 0
    while len(have) < n and dry < 3:
        want = min(BATCH, n - len(have))
        new = keep(
            parse(ask(messages(p.phrase, want, hints, have[-20:], p.language))), phrases, seen
        )
        dry = 0 if new else dry + 1
        new = new[: n - len(have)]
        have += new
        with open(f, "a") as out:
            out.writelines(s + "\n" for s in new)
        if on_progress:
            on_progress(len(have), n)
    if len(have) < n:
        log(
            f"warning: the AI assistant gave {len(have)} usable sentences of {n}; some are said twice"
        )
    return have


def clips(p: Project) -> list[str]:
    """The clips training uses (the first conversational.clips; a lowered count leaves extra files)."""
    files = sorted(str(x) for x in clip_dir(p).glob("clip_[0-9][0-9][0-9][0-9][0-9][0-9].wav"))
    return files[: max(0, p.conversational.clips)]


def complete(p: Project) -> bool:
    from .tts.generate import count_clips

    return count_clips(clip_dir(p)) >= p.conversational.clips


def run(p: Project, device: str, batch_size: int, log=print, ask=None) -> dict:
    """Sentences (cached or asked for), then each spoken once by the project's TTS models in turn."""
    from . import assets
    from .tts.generate import generate

    n = p.conversational.clips
    if n <= 0:
        return {"skipped": "off"}
    if complete(p):
        return {"skipped": "all clips present"}
    progress = p.path("logs", "generate-conversational.json")
    progress.parent.mkdir(parents=True, exist_ok=True)

    def report(step: str, done: int, total: int) -> None:
        progress.write_text(
            json.dumps({"split": step, "done": done, "total": total, "time": time.time()})
        )

    try:
        texts = collect(p, n, ask, log, lambda d, t: report("sentences", d, t))
    except ConnectionError as e:
        raise RuntimeError(
            f"{e}. Start the AI assistant (or set conversational.llm_url), or set the everyday "
            "sentences to 0 to train without them"
        ) from None
    if not texts:
        raise RuntimeError(
            "the AI assistant wrote no usable sentences; see conversational/sentences.txt"
        )
    log(f"{len(texts)} everyday sentences, e.g. {texts[0]!r}")
    models = [assets.ASSETS[assets.TTS_MODELS[m]] for m in p.tts_models]
    stats = {"sentences": len(texts), "written": 0, "failed": 0}
    for i, a in enumerate(models):
        assets.fetch(a)
        assets.fetch(assets.ASSETS[a.name + "-config"])
        r = generate(
            texts,
            clip_dir(p),
            n * (i + 1) // len(models),  # cumulative target after model i: clip k says sentence k
            a.local,
            device=device,
            batch_size=batch_size,
            seed=zlib.crc32(f"{p.name}/conversational/{a.name}".encode()),
            on_progress=lambda d, t: report("clips", d, n),
            in_order=True,
            espeak=languages.get(p.language).espeak,
        )
        stats["written"] += r["written"]
        stats["failed"] += r["failed"]
    return stats
