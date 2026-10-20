"""Single-clip previews for the web UI, in a separate CPU process.

One clip takes well under a second on the CPU, so previews never touch the GPU (training may be
using it), and a crash in synthesis can't take the web server down: the pool is rebuilt.
"""

from __future__ import annotations

import io
import multiprocessing as mp
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

_gen = None
_pool: ProcessPoolExecutor | None = None
_lock = threading.Lock()


def _init(model_path: str) -> None:
    global _gen
    import torch

    from .generate import Generator

    torch.set_num_threads(2)
    _gen = Generator(model_path, "cpu")


def _synth(text: str) -> bytes:
    from .generate import write_wav

    buf = io.BytesIO()
    write_wav(buf, _gen.batch([text])[0])
    return buf.getvalue()


def wav(text: str) -> bytes:
    """WAV bytes of `text` in a random synthetic voice."""
    global _pool
    from .. import assets

    with _lock:
        if _pool is None:
            a = assets.ASSETS["tts-libritts-r-medium"]
            assets.fetch(a)
            assets.fetch(assets.ASSETS[a.name + "-config"])
            _pool = ProcessPoolExecutor(
                1, mp_context=mp.get_context("spawn"), initializer=_init, initargs=(str(a.local),)
            )
        pool = _pool
    try:
        return pool.submit(_synth, text).result(timeout=120)
    except BrokenProcessPool:
        with _lock:
            _pool = None
        raise RuntimeError("the preview process crashed; try again") from None
