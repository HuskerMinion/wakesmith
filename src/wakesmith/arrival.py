"""Sync on arrival: when the device upload (inbox.py) stores a clip, the wake words that take it are
synced soon after, instead of at the daily timer, so a new clip is ready to label within a minute or two.

A wake word takes a clip when its Device clips folder (devices.source) holds the stored file and the
clip's "model" starts with its devices.model_prefix (empty: every clip), and devices.on_arrival is on.
A burst of clips is one sync: each word's sync waits until DELAY_S after its newest clip (at most
MAX_WAIT_S after its first), and brings in up to LIMIT clips (the rest the next time). Syncs run one
at a time, each as its own `wakesmith clips sync` process (as the timer runs it, outside the job queue,
so a training job never waits for one and Whisper never loads inside the server), with output in
clips-sync.log. Such a sync skips a word whose own job is running, and transcribes on the CPU while
another job holds the GPU (clipsync.py); the clips it skipped come in with the next one.
This runs only while `wakesmith serve` does, since that's where the upload is received.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import assets, clipsync
from . import project as projects

DELAY_S = 60.0  # after a word's newest clip, so a burst is one sync (and one Whisper load)
MAX_WAIT_S = 600.0  # but never longer than this after its first waiting clip (a steady trickle)
LIMIT = 200  # clips per arrival sync, oldest first; a flood comes in over several syncs
SYNC_TIMEOUT_S = 3600.0


def takers(stored: Path, model: str) -> list[str]:
    """The project folders whose sync would bring in this stored clip."""
    try:
        clip = stored.resolve()
    except OSError:
        return []
    out = []
    for f in projects.project_files(projects.projects_home()):
        try:
            p = projects.load(f.parent)
        except (OSError, ValueError, TypeError):
            continue
        d = p.devices
        if not (d.source and d.on_arrival and str(model or "").startswith(d.model_prefix)):
            continue
        try:
            if clip.is_relative_to(Path(d.source).expanduser().resolve()):
                out.append(str(p.dir))
        except OSError:
            continue
    return out


def _run_sync(project_dir: str) -> None:
    """One `wakesmith clips sync` of a word, as its own process (low priority), output to the sync log."""
    cmd = [sys.executable, "-m", "wakesmith", "clips", "sync", project_dir, "--limit", str(LIMIT)]
    if shutil.which("nice"):
        cmd = ["nice", "-n", "10", *cmd]
    log = clipsync.log_file()
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as out:
        subprocess.run(
            cmd,
            stdout=out,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            timeout=SYNC_TIMEOUT_S,
            check=False,
        )


class Arrivals:
    """The words waiting for a sync, and the one thread that runs them in turn."""

    def __init__(
        self,
        run=_run_sync,
        delay: float = DELAY_S,
        frozen=lambda: False,
        max_wait: float = MAX_WAIT_S,
    ):
        self._run, self._delay, self._frozen, self._max = run, delay, frozen, max_wait
        self._due: dict[str, float] = {}  # project folder -> when its sync may start
        self._first: dict[str, float] = {}  # project folder -> its first waiting clip
        self._cv = threading.Condition()
        self._thread: threading.Thread | None = None

    def note(self, stored: Path, model: str) -> list[str]:
        """A clip was stored: (re)start the wait of each word that takes it. Returns those words' folders."""
        words = takers(stored, model)
        if not words:
            return []
        with self._cv:
            now = time.monotonic()
            for w in words:
                first = self._first.setdefault(w, now)
                self._due[w] = min(now + self._delay, first + self._max)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._loop, name="wakesmith-arrival", daemon=True
                )
                self._thread.start()
            self._cv.notify()
        return words

    def pending(self) -> list[str]:
        with self._cv:
            return sorted(self._due)

    def _next(self) -> str:
        with self._cv:
            while True:
                now = time.monotonic()
                ready = [w for w, t in self._due.items() if t <= now]
                if ready and not self._frozen() and not assets.update_underway():
                    w = min(ready, key=self._due.__getitem__)
                    del self._due[w]
                    self._first.pop(w, None)
                    return w
                wait = min(self._due.values()) - now if self._due and not ready else self._delay
                self._cv.wait(timeout=max(wait, 0.05))

    def _again_if_left(self, w: str, started: float) -> None:
        """The sync stopped at LIMIT clips: the rest come in with another one, not at the next clip.
        Only for a result this sync wrote: a skipped or failed one leaves an older file behind."""
        try:
            r = json.loads((Path(w) / "devices" / "sync.json").read_text())
            left, at = r.get("left"), float(r.get("at") or 0)
        except (OSError, ValueError, AttributeError, TypeError):
            return
        if r.get("result") in ("failed", "skipped") or at < started:
            return
        if isinstance(left, int) and left > 0:
            with self._cv:
                self._due.setdefault(w, time.monotonic() + self._delay)
                self._first.setdefault(w, time.monotonic())
                self._cv.notify()

    def _loop(self) -> None:
        while True:
            w = self._next()
            try:
                started = time.time()
                self._run(w)
                self._again_if_left(w, started)
            except Exception as e:  # noqa: BLE001 - note it; the next clip or the daily sync catches up
                try:
                    clipsync.log(f"{Path(w).name}: sync on arrival failed: {type(e).__name__}: {e}")
                except OSError:
                    pass
