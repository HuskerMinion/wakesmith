"""Languages a wake word can be trained in. English is the default; others are experimental and stay
hidden until turned on (the Data page, settings.json "languages": {"experimental": true}, or
WAKESMITH_EXPERIMENTAL_LANGUAGES=1).

A language brings its espeak-ng voice (phonemes for the speech models and for the sound-alike check),
its Piper voices, and the language Whisper listens for.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from . import assets


@dataclass(frozen=True)
class Language:
    code: str
    name: str  # in the language itself
    espeak: str
    tts_models: tuple[str, ...]  # keys of assets.TTS_MODELS
    experimental: bool = False


LANGUAGES = {
    "en": Language("en", "English", "en-us", ("libritts-high", "libritts-r-medium")),
    "de": Language(
        "de",
        "Deutsch",
        "de",
        tuple(f"de-{v}" for v in assets.GERMAN_VOICES),
        experimental=True,
    ),
}


def experimental_on() -> bool:
    from . import assistant

    env = os.environ.get("WAKESMITH_EXPERIMENTAL_LANGUAGES", "").strip().lower()
    if env not in ("", "0", "false", "no", "off"):
        return True
    return bool((assistant._read_all().get("languages") or {}).get("experimental"))


def set_experimental(on: bool) -> None:
    from . import assistant

    assistant.update_all(lambda data: data.__setitem__("languages", {"experimental": bool(on)}))


def available() -> list[Language]:
    """The languages offered for a new wake word."""
    on = experimental_on()
    return [lang for lang in LANGUAGES.values() if on or not lang.experimental]


def get(code: str) -> Language:
    if code not in LANGUAGES:
        raise ValueError(f"unknown language {code!r} (known: {', '.join(LANGUAGES)})")
    return LANGUAGES[code]
