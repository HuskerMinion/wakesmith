"""Text → Piper phoneme ids, using the `espeak-ng` program.

Matches Piper's own phonemizer (espeak-ng IPA, decomposed to NFD code points, clause punctuation
kept) without linking to espeak-ng: running it as a separate program keeps wakesmith free of GPL code.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import unicodedata
from functools import cache

# Clause terminators Piper keeps as "phonemes"; , : ; are followed by a space (not a sentence end).
_CLAUSE = re.compile(r"([,:;.!?])")


def check_espeak() -> str:
    path = shutil.which("espeak-ng")
    if path is None:
        from ..setup_env import install_hint

        raise RuntimeError(
            f"espeak-ng not found: install it (e.g. `{install_hint(['espeak-ng'])}`)"
        )
    return path


@cache
def _ipa(text: str, voice: str) -> str:
    out = subprocess.run(
        ["espeak-ng", "-q", "-v", voice, "--ipa", text],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return " ".join(out.split())


@cache
def phonemes(text: str, voice: str = "en-us") -> tuple[str, ...]:
    """Phonemes of `text` as single code points, as Piper's training data had them."""
    out = ""
    parts = _CLAUSE.split(text)
    for chunk, term in zip(parts[::2], parts[1::2] + [""]):
        chunk = chunk.strip()
        if chunk:
            out += _ipa(chunk, voice)
        if term:
            out += term + (" " if term in ",:;" else "")
    return tuple(unicodedata.normalize("NFD", out.strip()))


def phoneme_ids(text: str, id_map: dict[str, list[int]], voice: str = "en-us") -> list[int]:
    """Piper's id layout: ^ _ (phoneme _)* $, skipping phonemes the model doesn't know."""
    ids = list(id_map["^"]) + list(id_map["_"])
    for p in phonemes(text, voice):
        if p in id_map:
            ids += id_map[p] + id_map["_"]
    return ids + list(id_map["$"])
