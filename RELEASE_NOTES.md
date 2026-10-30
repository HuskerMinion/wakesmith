## What's new in 0.4.3

**More macOS fixes.**
- Training no longer stops at "microWakeWord features 0/14" on a Mac. The feature and training workers
  were started by forking a process that had already used the GPU, which hangs on macOS; there they now
  start fresh (`spawn`). Linux is unchanged.
- The RAM check reads the Mac's memory. It used to fall back to 8 GB on macOS and warn that
  microWakeWord training needs 16 GB.
- microWakeWord training on macOS no longer fails when raising the open-file limit, which macOS caps
  lower than Linux.
- The Training page says "under a minute left in this step" instead of "about under a minute".

Training on a Mac (Apple Silicon, on its GPU via MPS) is still new; reports are welcome.

To update: `uv tool install --python 3.12 --force ./wakesmith-0.4.3-py3-none-any.whl`.

Earlier: macOS job start and Training page fixes in
[0.4.2](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.2), install fixes in
[0.4.1](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.1), German wake words (experimental) in
[0.4.0](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.0).
