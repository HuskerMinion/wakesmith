## What's new in 0.4.5

**Wake words spelled the way they sound.**
- A new **Also written as…** item in a wake word's ⋯ menu. If you typed the phrase the way it sounds so
  the synthetic voices say it right (e.g. "hey hanoomahn"), add how speech recognition writes it
  ("hey hanuman"). Record a room then finds your takes (use Split again on a recording you already
  made), and so do device clips. The synthetic voices still say what you typed.
- Takes marked "part of a wake word" on Record a room come with a hint pointing to it.

**Play buttons show what's playing.** The button shows a stop square while loading and playing, with a
ring that fills as the clip plays, and a second click stops it. One clip plays at a time, and leaving
the page stops it.

**Where a version's files are.** The Training page's Log section shows the version's folder (the log is
`logs/train.log` in it, the models are in `export`).

To update: `uv tool install --python 3.12 --force ./wakesmith-0.4.5-py3-none-any.whl`.

Earlier: Wake words page menu fixes in [0.4.4](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.4),
macOS training fixes (workers, RAM, open-file limit) in
[0.4.3](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.3), macOS job start and Training page fixes in
[0.4.2](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.2), install fixes in
[0.4.1](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.1), German wake words (experimental) in
[0.4.0](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.0).
