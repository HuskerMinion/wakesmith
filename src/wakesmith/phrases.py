"""Suggest sound-alike phrases the model should learn to reject.

For each word of the wake phrase, CMUdict words a short weighted phoneme distance away (similar
sounds swap cheaply), stress ignored; words CMUdict lacks get spelling variants instead. Phrases are
built by swapping one word at a time, plus partial phrases ("jarvis" alone, "hey" + a sound-alike).
Anything espeak-ng pronounces exactly like the wake phrase is dropped: it would teach the model to
reject the wake word itself.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from . import assets

VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}


@lru_cache(maxsize=1)
def lexicon() -> tuple[dict[str, tuple[str, ...]], dict[int, list[tuple[str, tuple[str, ...]]]]]:
    path = assets.fetch(assets.ASSETS["cmudict"], progress=False)
    words: dict[str, tuple[str, ...]] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split("#")[0].split()
        if len(parts) < 2 or not re.fullmatch(r"[a-z']+", parts[0]):
            continue  # skip alternates "word(2)" and odd entries
        words.setdefault(parts[0], tuple(re.sub(r"\d", "", p) for p in parts[1:]))
    by_len: dict[int, list] = defaultdict(list)
    for w, ph in words.items():
        by_len[len(ph)].append((w, ph))
    return words, by_len


# Consonants that sound alike (voicing / place / liquids / nasals): swapping within a group is cheap.
_GROUPS = [
    {"P", "B"},
    {"T", "D"},
    {"K", "G"},
    {"F", "V"},
    {"S", "Z"},
    {"SH", "ZH"},
    {"CH", "JH"},
    {"TH", "DH"},
    {"M", "N", "NG"},
    {"L", "R"},
    {"W", "Y"},
    {"HH"},
]
_GROUP = {p: i for i, g in enumerate(_GROUPS) for p in g}


def _sub(a: str, b: str) -> float:
    if a == b:
        return 0.0
    va, vb = a in VOWELS, b in VOWELS
    if va and vb:
        return 0.5
    if va != vb:
        return 1.5
    return 0.5 if _GROUP.get(a, -1) == _GROUP.get(b, -2) else 1.0


def distance(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    """Weighted phoneme edit distance (insert/delete 1, similar-sound swaps 0.5)."""
    prev = [float(j) for j in range(len(b) + 1)]
    for i, x in enumerate(a, 1):
        cur = [float(i)] + [0.0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + _sub(x, y))
        prev = cur
    return prev[-1]


def _contains(big: tuple[str, ...], small: tuple[str, ...]) -> bool:
    n = len(small)
    return any(big[i : i + n] == small for i in range(len(big) - n + 1))


def neighbors(word: str, limit: int = 40, seed: int = 0) -> list[str]:
    """Dictionary words that sound close to `word` (closest first), never containing it whole."""
    words, by_len = lexicon()
    ph = words.get(word.lower())
    if ph is None:
        return []
    budget = 1.0 if len(ph) <= 4 else 1.5 if len(ph) <= 6 else 2.0
    scored = []
    for n in range(len(ph) - 2, len(ph) + 3):
        for w, p in by_len.get(n, []):
            if p == ph or w == word or (len(p) > len(ph) and _contains(p, ph)):
                continue
            d = distance(ph, p)
            if d <= budget:
                scored.append((d, w))
    rng = random.Random(seed)
    rng.shuffle(scored)
    scored.sort(key=lambda t: t[0])
    return [w for _, w in scored[:limit]]


_VOWEL_LETTERS = "aeiou"
_SIMILAR_LETTERS = {
    "b": "pd",
    "p": "bt",
    "d": "tb",
    "t": "dk",
    "g": "kd",
    "k": "gt",
    "v": "fb",
    "f": "vth",
    "s": "zsh",
    "z": "s",
    "j": "chg",
    "m": "n",
    "n": "m",
    "l": "r",
    "r": "lw",
    "w": "rv",
    "h": "",
    "c": "kg",
}


def spelling_variants(word: str, limit: int = 12, seed: int = 0) -> list[str]:
    """Pseudo-words one letter-sound away ("nabu" -> "nebu", "mabu"), for words CMUdict lacks.
    The TTS reads them like real words, which is all a negative example needs."""
    out = set()
    for i, ch in enumerate(word):
        if ch in _VOWEL_LETTERS:
            out.update(word[:i] + v + word[i + 1 :] for v in _VOWEL_LETTERS if v != ch)
        for alt in _SIMILAR_LETTERS.get(ch, ""):
            out.add(word[:i] + alt + word[i + 1 :])
    out.discard(word)
    items = sorted(out)
    random.Random(seed).shuffle(items)
    return items[:limit]


def homophones(word: str) -> set[str]:
    words, by_len = lexicon()
    ph = words.get(word.lower())
    if ph is None:
        return set()
    return {w for w, p in by_len[len(ph)] if p == ph}


def _ipa_key(text: str) -> str:
    """espeak-ng pronunciation without stress marks, spaces or punctuation."""
    from .tts.phonemize import phonemes

    return "".join(c for c in phonemes(text) if c not in "ˈˌ ,.!?;:")


def suggest(phrase: str, n: int = 60, seed: int = 0) -> list[str]:
    """Up to `n` sound-alike phrases for `phrase`, most useful kinds first."""
    toks = re.findall(r"[a-z']+", phrase.lower())
    if not toks:
        return []
    banned = {phrase.lower(), " ".join(toks)}
    rng = random.Random(seed)
    swaps: list[list[str]] = []
    for i, t in enumerate(toks):
        cands = [c for c in neighbors(t, limit=60, seed=seed + i) if c not in homophones(t)]
        if len(cands) < 8:
            cands += spelling_variants(t, limit=12, seed=seed + i)
        swaps.append([" ".join(toks[:i] + [c] + toks[i + 1 :]) for c in cands])
    picks: list[str] = []
    if len(toks) > 1:
        picks.append(" ".join(toks[1:]))  # the name without "hey": the classic false trigger
        picks.append(" ".join(toks[:-1]))
    # Swaps of the last (usually distinctive) word first, then the others, round-robin
    order = list(range(len(toks)))[::-1]
    while any(swaps[i] for i in order) and len(picks) < n * 2:
        for i in order:
            if swaps[i]:
                picks.append(swaps[i].pop(0))
    if len(toks) > 1:  # partials: a sound-alike of the last word on its own
        last = neighbors(toks[-1], limit=6, seed=seed + 99) or spelling_variants(toks[-1], 4, seed)
        picks += [w for w in last if w not in homophones(toks[-1])]
    target = _ipa_key(phrase)
    seen, out = set(), []
    for p in picks:
        if p not in seen and p not in banned and _ipa_key(p) != target:
            seen.add(p)
            out.append(p)
    head, tail = out[:4], out[4:]
    rng.shuffle(tail)
    return (head + tail)[:n]
