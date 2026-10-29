"""Single-clip previews for the web UI, in a separate CPU process.

One clip takes well under a second on the CPU, so previews never touch the GPU (training may be
using it), and a crash in synthesis can't take the web server down: the pool is rebuilt.
"""

from __future__ import annotations

import io
import multiprocessing as mp
import random
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

_gens: list = []
_pools: dict[str, ProcessPoolExecutor] = {}  # one per language
_lock = threading.Lock()
_fetch_lock = threading.Lock()
# The voices a preview picks from: English's 904-speaker voice; for German a man's and a woman's (CC0)
PREVIEW_VOICES = {"en": ["libritts-r-medium"], "de": ["de-thorsten-medium", "de-kerstin-low"]}


def _init(model_paths: list[str]) -> None:
    import torch

    from .generate import Generator, OnnxGenerator

    torch.set_num_threads(2)
    _gens[:] = [
        OnnxGenerator(Path(m)) if m.endswith(".onnx") else Generator(m, "cpu") for m in model_paths
    ]


def _synth(text: str) -> bytes:
    from .generate import write_wav

    buf = io.BytesIO()
    write_wav(buf, random.choice(_gens).batch([text])[0])
    return buf.getvalue()


def wav(text: str, language: str = "en") -> bytes:
    """WAV bytes of `text` in a random synthetic voice of `language`."""
    from .. import assets

    voices = [assets.ASSETS[assets.TTS_MODELS[m]] for m in PREVIEW_VOICES[language]]
    paths = [str(a.local) for a in voices]
    if language not in _pools:
        # a first German preview downloads its voices: under its own lock, so previews in a
        # language that is ready don't wait for it
        with _fetch_lock:
            for a in voices:  # another process downloading it: say so instead of waiting
                assets.fetch(a, wait=False)
                assets.fetch(assets.ASSETS[a.name + "-config"], wait=False)
    with _lock:
        if language not in _pools:
            _pools[language] = ProcessPoolExecutor(
                1, mp_context=mp.get_context("spawn"), initializer=_init, initargs=(paths,)
            )
        pool = _pools[language]
    try:
        return pool.submit(_synth, text).result(timeout=120)
    except BrokenProcessPool:
        with _lock:
            _pools.pop(language, None)
        raise RuntimeError("the preview process crashed; try again") from None
