"""The feature frontend of zserge/microwakeword (pure Go), which some devices run instead of
TFLite Micro's microfrontend. Ported to numpy from its audiofrontend package (MIT License,
Copyright (c) 2025 Serge Zaitsev).

It is a float approximation of the microfrontend: same window, mel range and int8 scaling, but its
noise reduction smooths by "signal above estimate" instead of by channel parity, PCAN is
x / (noise + 80)^0.95 * 80 without the gain table, and the log is natural log * 64. Features differ
enough that a model can score very differently on it, so models for these devices should be
scored (and trained) on it.
"""

from __future__ import annotations

import numpy as np

RATE = 16000
WINDOW = 480  # 30 ms
FFT = 512
CHANNELS = 40


def _mel_filters() -> np.ndarray:
    def mel(hz):
        return 1127.0 * np.log(1.0 + hz / 700.0)

    lo, hi = mel(125.0), mel(7500.0)
    pts = lo + np.arange(CHANNELS + 2) * (hi - lo) / (CHANNELS + 1)
    bins = 700.0 * (np.exp(pts / 1127.0) - 1.0) * FFT / RATE
    b = np.arange(FFT // 2 + 1, dtype=np.float64)
    out = np.zeros((CHANNELS, FFT // 2 + 1))
    for ch in range(CHANNELS):
        left, center, right = bins[ch : ch + 3]
        up = (b >= left) & (b <= center)
        down = (b > center) & (b <= right)
        out[ch, up] = (b[up] - left) / (center - left)
        out[ch, down] = (right - b[down]) / (right - center)
    return out


_HANN = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WINDOW) / WINDOW)
_MEL = _mel_filters()


def features(audio: np.ndarray, step_ms: int = 10) -> np.ndarray:
    """int16 (or float in [-1, 1]) 16 kHz audio -> int8 features [frames, 40], as the device
    computes them from the start of a stream."""
    if audio.dtype != np.int16:
        audio = np.clip(np.round(audio * 32768), -32768, 32767).astype(np.int16)
    step = RATE * step_ms // 1000
    n = (len(audio) - WINDOW) // step + 1
    if n <= 0:
        return np.zeros((0, CHANNELS), np.int8)
    idx = np.arange(WINDOW)[None, :] + step * np.arange(n)[:, None]
    spec = np.fft.rfft(audio[idx].astype(np.float64) * _HANN, FFT)
    energy = np.sqrt(np.maximum((spec.real**2 + spec.imag**2) @ _MEL.T, 0))

    noise = energy[0].copy()
    out = np.empty_like(energy)
    for i, e in enumerate(energy):
        if i:
            smooth = np.where(e > noise, 0.06, 0.025)
            noise = noise * (1 - smooth) + e * smooth
        sig = np.maximum(np.maximum(e - noise, e * 0.05), 0)
        out[i] = sig / np.power(noise + 80.0, 0.95) * 80.0

    raw = np.clip(np.log(np.maximum(out, 1)) * 64, 0, 65535).astype(np.int32)  # uint16 in Go
    return np.clip((raw * 256 + 333) // 666 - 128, -128, 127).astype(np.int8)
