"""microWakeWord's trainer, with training batches prepared ahead in a background process.

microWakeWord builds each batch (128 random picks from the feature sets) in a pure-Python loop between training
steps: ~12 s of every ~45 s per 1000 steps. A forked producer keeps a few batches ready instead; it only reads
the memory-mapped features (no TensorFlow), and validation/testing batches are still built in place.

usage: python -m wakesmith.mww_runner <microwakeword.model_train_eval arguments>
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
import runpy
import sys

from wakesmith.hardware import worker_start

AHEAD = 4  # batches kept ready (~4 MB each)


def _producer(handler, orig, args, kwargs, q) -> None:
    import numpy as np

    random.seed(os.getpid())
    np.random.seed(os.getpid() % 2**32)
    while True:
        q.put(orig(handler, *args, **kwargs))


def _install() -> None:
    from microwakeword import data

    orig = data.FeatureHandler.get_data

    def get_data(self, mode, *args, **kwargs):
        # no forked producer where workers must spawn (macOS, see hardware.worker_start): built in place
        if mode != "training" or worker_start() == "spawn":
            return orig(self, mode, *args, **kwargs)
        key = repr((args, sorted(kwargs.items())))
        st = self.__dict__.setdefault("_ahead", {})
        if st.get("key") != key:  # first call, or new settings (a new training phase)
            if st.get("proc"):
                st["proc"].terminate()
            q = mp.get_context("fork").Queue(AHEAD)
            proc = mp.get_context("fork").Process(
                target=_producer, args=(self, orig, ("training", *args), kwargs, q), daemon=True
            )
            proc.start()
            st.update(key=key, proc=proc, q=q)
        return st["q"].get()

    data.FeatureHandler.get_data = get_data


if __name__ == "__main__":
    _install()
    sys.argv = ["microwakeword.model_train_eval", *sys.argv[1:]]
    runpy.run_module("microwakeword.model_train_eval", run_name="__main__", alter_sys=True)
