## What's new in 0.4.9

**Candidates train on every GPU at once.** A Standard run trains microWakeWord several times with
different seeds and keeps the best one. With two or more graphics cards, those runs now go side by side,
one per card, as many at once as free memory allows (about 8 GB each), instead of one after another on
the first card: on two RTX 5060 Ti, four candidates took 4 minutes instead of about 7. The Training page counts every candidate in its progress and says how
many train at once, and the time estimate learns from it after the first run. With one card, or on the
CPU, nothing changes.

**The kept candidate catches calls.** Choosing between candidates looked at false triggers first, so a
seed that barely fires could win: in one short run the kept candidate caught none of the held-out takes.
Now only candidates that catch at least 70% as many held-out takes as the best one compete; among those,
the fewest false triggers still decide. Results says when a candidate was left out for that.

To update: the Update button on the Hardware page (from 0.4.8; nothing else to do), or by hand:
`uv tool install --python 3.12 --force ./wakesmith-0.4.9-py3-none-any.whl`, then `wakesmith setup`.

Earlier: the listening check, the Real voices guide and updating from the page in
[0.4.8](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.8), a smith at the forge while you train and `test --model` keeping the project's results in
[0.4.7](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.7), Macs with Apple silicon and the folder summary in
[0.4.6](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.6), Also written as, play buttons that show what's playing and the version's folder in
[0.4.5](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.5), Wake words page menu fixes in
[0.4.4](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.4), macOS training fixes (workers, RAM, open-file limit) in
[0.4.3](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.3), macOS job start and Training page fixes in
[0.4.2](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.2), install fixes in
[0.4.1](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.1), German wake words (experimental) in
[0.4.0](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.0).
