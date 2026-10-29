## What's new in 0.4.1

**Install fixes.**
- `wakesmith setup` stops at once, with the command to fix it, when wakesmith was installed on a Python
  it can't use. wakesmith needs Python 3.10 to 3.12; uv can set it up on a newer one from a local wheel,
  and setup then failed minutes later in a dependency's build. Install with
  `uv tool install --python 3.12 ./wakesmith-0.4.1-py3-none-any.whl` (the README now says so).
- On macOS, setup suggests `brew install` for missing tools instead of apt. macOS is still untested and
  runs on the CPU only; reports are welcome.

To update: `uv tool install --python 3.12 --force ./wakesmith-0.4.1-py3-none-any.whl`, then
`wakesmith setup`.

The German wake words (experimental) came in [0.4.0](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.0).
