## What's new in 0.4.6

**Runs on Macs with Apple silicon.** Training on a Mac uses its GPU for every heavy stage: a user's M1
trained a full-size model start to finish. The README and the German guide no longer call macOS untested,
and `wakesmith setup` on a Mac no longer reinstalls PyTorch every time it runs.

**The summary counts your own folder.** When a wake word trains on a folder of recordings you brought,
the "So far" box on the Real voices page now counts that folder too: its clips, how many train, and how
many its `HELDOUT.txt` keeps aside to test. Speakers named in a `speaker` column of its `labels.csv`
count toward the speaker check. The warnings cover the folder too: with nothing kept aside, they point to
`HELDOUT.txt`. A `HELDOUT.txt` that lists missing files, a `labels.csv` without a `file` column, one
that isn't UTF-8 or has a row without a file name, and a folder that's gone are shown there, not as a
broken page. Files saved by a spreadsheet (with a byte-order mark, blank rows below the data) are read
as they are. The "What it learns from" panel on the same page has a bigger arrow, so it reads as one that opens.

**A hand-edited `labels.csv` of the takes recorded here is named, not a dead end.** If it can't be read
(not UTF-8, a row without a file name, or a row with more cells than the header), or its `HELDOUT.txt`
isn't UTF-8, the Real voices page still opens and says so, and the take buttons name the file. A take
file cut short by a copy is listed as "can't be read", and the summary asks you to delete it before
training. Deleting the wake word still works. A room recording's send still gives the other wake words
their takes and lists the problem, and what went out is logged even if a send stops part-way, so sending
again adds nothing twice.

**microWakeWord feature workers share the cores.** Each worker now runs one OpenBLAS thread instead of
one per core, so on many-core Linux machines the workers no longer compete for the cores.

To update: `uv tool install --python 3.12 --force ./wakesmith-0.4.6-py3-none-any.whl`.

Earlier: Also written as, play buttons that show what's playing and the version's folder in
[0.4.5](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.5), Wake words page menu fixes in
[0.4.4](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.4), macOS training fixes (workers, RAM, open-file limit) in
[0.4.3](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.3), macOS job start and Training page fixes in
[0.4.2](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.2), install fixes in
[0.4.1](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.1), German wake words (experimental) in
[0.4.0](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.0).
