"""Real check of the combined model (ensemble.py) on a trained project's best-of-N candidates.

Needs TensorFlow, microWakeWord and a project trained with candidates > 1 (its features and
training.yaml), so it only runs when asked:
    WAKESMITH_COMBINED=<project name or folder> pytest -s tests/test_combined_real.py
It writes only <project>/mww/combined/. The combined file must take int8 [1, 3, 40] and give uint8, and on
20 random inputs streamed in sequence its output must equal, within 2/255, the mean of the candidates' own
streaming models run the same way.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from wakesmith import ensemble
from wakesmith import project as projects

WHERE = os.environ.get("WAKESMITH_COMBINED", "")
pytestmark = pytest.mark.skipif(
    not WHERE, reason="real combined-model build: set WAKESMITH_COMBINED=<project>"
)


def _stream(path, inputs_f32):
    """uint8 outputs of a streaming .tflite fed the float inputs in order (state kept between calls)."""
    import tensorflow as tf

    it = tf.lite.Interpreter(model_path=str(path))
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    scale, zero = i["quantization"]
    out = []
    for x in inputs_f32:
        q = np.clip(np.round(x / scale + zero), -128, 127).astype(np.int8)
        it.set_tensor(i["index"], q)
        it.invoke()
        out.append(int(it.get_tensor(o["index"]).ravel()[0]))
    return np.array(out), i, o


def test_combined_matches_mean_of_candidates():
    import tensorflow as tf

    p = projects.load(WHERE)
    cands = ensemble.available(p)
    assert len(cands) >= 2, "the project needs at least 2 trained candidates"
    model = ensemble.combine(p)
    info = ensemble.info(p)
    assert info and info["candidates"] == [c["n"] for c in cands]

    it = tf.lite.Interpreter(model_path=str(model))
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    assert i["dtype"] == np.int8 and list(i["shape"]) == [1, 3, 40]
    assert o["dtype"] == np.uint8

    rng = np.random.default_rng(0)
    raw = rng.integers(-128, 128, size=(20, 1, 3, 40)).astype(np.int8)
    scale, zero = i["quantization"]
    inputs = [(r.astype(np.float32) - zero) * scale for r in raw]  # the same audio for every model

    combined, _, _ = _stream(model, inputs)
    singles = []
    for c in cands:
        tfl = c["weights"].parent / ensemble.TFLITE
        y, ci, co = _stream(tfl, inputs)
        assert ci["dtype"] == np.int8 and co["dtype"] == np.uint8
        singles.append(y)
    mean = np.mean(singles, axis=0)
    diff = np.abs(combined - mean)
    print(
        f"\n{p.name}: combined of {ensemble.note(info)}, {info['bytes']} bytes, {info['seconds']} s;"
        f" input scale {scale:.5f} zero {zero}; outputs (x/255) combined {combined.tolist()}"
        f"\n mean of candidates {np.round(mean, 2).tolist()}"
        f"\n |diff| max {diff.max():.2f}/255, mean {diff.mean():.2f}/255"
    )
    assert diff.max() <= 2.0


def test_combined_on_real_clips():
    """The same comparison on held-out recordings, the way evaluate.py scores them: per-frame
    probabilities and each clip's score (peak of the 5-frame mean)."""
    import soundfile as sf
    from microwakeword.inference import Model

    p = projects.load(WHERE)
    model = ensemble.model_path(p)
    if not model.exists():
        ensemble.combine(p)
    cands = [
        ensemble.folder(p).parent / "candidates" / f"s{n}" / ensemble.TFLITE
        for n in ensemble.info(p)["candidates"]
    ]
    clips = sorted(p.path("real", "test").glob("*.wav"))[:10]
    clips += sorted(p.path("real_neg", "test").glob("*.wav"))[:10]
    assert clips, "the project has no held-out recordings"
    frame, peak = [], []
    for c in clips:
        a, _ = sf.read(str(c), dtype="float32", always_2d=True)
        a = np.concatenate([np.zeros(16000, np.float32), a[:, 0], np.zeros(8000, np.float32)])
        y = np.asarray(Model(str(model)).predict_clip(a, step_ms=10), np.float32)
        m = np.mean([Model(str(t)).predict_clip(a, step_ms=10) for t in cands], axis=0)
        frame.append(np.abs(y - m))
        win = np.ones(5) / 5
        peak.append(abs(np.convolve(y, win, "valid").max() - np.convolve(m, win, "valid").max()))
    frame = np.concatenate(frame)
    print(
        f"\n{len(clips)} held-out clips, {frame.size} frames: |combined - mean of candidates| per frame"
        f" max {frame.max():.4f}, mean {frame.mean():.4f}; clip score max {max(peak):.4f},"
        f" mean {np.mean(peak):.4f}"
    )
    assert np.mean(peak) < 0.01
