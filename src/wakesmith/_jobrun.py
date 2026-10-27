"""Becomes a background job: `python -m wakesmith._jobrun JOB_JSON ARGS...`.

Notes its own pid in JOB_JSON, runs `python -m wakesmith ARGS...` in the same session and waits for it.
The job (and the workers it starts, which inherit the setting) is the kernel's first pick when memory runs
out; this small launcher is not, so it is still there to note how the job ended (exit code, or killed) in
JOB_JSON and the jobs history, and then to start the next queued run (jobs.finished). jobs.start launches
it outside the server's process tree, so stopping or restarting the server leaves the job running.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys

OOM_SCORE_ADJ = (
    500  # the job, not the desktop or other services, goes first (raising it needs no privileges)
)


def _oom_first() -> None:
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write(str(OOM_SCORE_ADJ))
    except OSError:
        pass


def main() -> None:
    job_file, args = sys.argv[1], sys.argv[2:]
    with open(job_file) as f:
        job = json.load(f)
    job["pid"] = os.getpid()
    tmp = f"{job_file}.part"
    with open(tmp, "w") as f:
        json.dump(job, f)
    os.replace(tmp, job_file)
    # Stop signals the whole session or unit: the job ends, the launcher stays to note it. A handler,
    # not SIG_IGN, so the job itself keeps the default action (handlers reset on exec).
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: None)
    try:
        code = subprocess.call([sys.executable, "-m", "wakesmith", *args], preexec_fn=_oom_first)
    except OSError:
        code = 127
    from . import jobs  # only now: the launcher stays small while the job runs

    jobs.finished(job_file, code)


if __name__ == "__main__":
    main()
