## What's new in 0.4.2

**macOS fixes.**
- Training starts on macOS. Jobs were launched with the `setsid` command, which macOS doesn't have; they now
  start in a session of their own without it.
- The Training page no longer crashes on macOS. It read memory use from `/proc`, which only Linux has; where
  that isn't there, the page skips the memory bars.

Both were checked on macOS 26 (the server, starting and stopping a job, the Training page). Training itself
is still untested on a Mac and runs on the CPU only; reports are welcome.

To update: `uv tool install --python 3.12 --force ./wakesmith-0.4.2-py3-none-any.whl`.

Install fixes for Python and macOS came in [0.4.1](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.1),
German wake words (experimental) in [0.4.0](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.0).
