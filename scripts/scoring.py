"""Score a whole recording with a trained model the way a live listener would, returning the peak.
  .onnx    livekit-wakeword / openWakeWord-style model (run in .venv): a 2 s window slides in 80 ms
           hops and the peak score counts.
  .tflite  microWakeWord streaming model (run in .venv-mww): one probability per 30 ms, and a
           detection is the mean of the last `window` probabilities, as ESPHome does.
Each clip gets 1 s of silence before and 0.5 s after, so the model's state starts clean and a phrase
at the very end is still seen whole."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

LEAD, TAIL = 16000, 8000


def _load(path: Path, dtype: str) -> np.ndarray:
    a, _ = sf.read(path, dtype=dtype, always_2d=True)
    return np.concatenate([np.zeros(LEAD, dtype), a[:, 0], np.zeros(TAIL, dtype)])


def scorer(model: str | Path, window: int = 5) -> Callable[[Path], float]:
    model = str(model)
    if model.endswith(".onnx"):
        from livekit.wakeword import WakeWordModel

        m = WakeWordModel(models=[model])  # stateless: each call scores one window
        frame, win = 1280, 32000

        def peak(path: Path) -> float:
            a = _load(path, "int16")
            return max(max(m.predict(a[i - win : i]).values()) for i in range(win, len(a) + 1, frame))

        return peak
    if model.endswith(".tflite"):
        from microwakeword.inference import Model

        def peak(path: Path) -> float:
            # fresh streaming state per clip
            p = np.asarray(Model(model).predict_clip(_load(path, "float32"), step_ms=10), np.float32)
            if len(p) < window:
                return float(p.max(initial=0))
            return float(np.convolve(p, np.ones(window) / window, mode="valid").max())

        return peak
    raise ValueError(f"don't know how to score {model} (want .onnx or a microWakeWord .tflite)")
