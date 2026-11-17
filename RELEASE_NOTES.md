## What's new in 0.4.13

**A fair test from your devices.** Takes recorded on a device come in a run at a time, and each run is
now trained on or held out for the test as a whole: by default the first run is held out, the next
trained on, and so on (*Takes recorded on a device*, under Device clips > Sync automatically). A run
recorded on purpose keeps every capture with speech in it, so a held-out run counts missed calls
honestly, which clips saved when a model fired can't.

**Candidates that fire too soon are passed over.** Each microWakeWord candidate is now timed on the
synthetic test clips at the project's cutoff, and one that typically fires before the wake word has
ended (which would cut the start of the command) is passed over while another candidate is left.

**Training leaves a full GPU alone.** An NVIDIA GPU another program has nearly filled (under 5 GB
free: another trainer, a local LLM) is no longer given microWakeWord candidates, openWakeWord training
or Whisper, which ran out of memory there; if every GPU is that full, the one with the most room is
used. Unlike NVIDIA cards are checked only with CUDA_DEVICE_ORDER=PCI_BUS_ID set.

**Clips arriving during training.** A wake word whose own training is running now syncs its newly
arrived device clips when the job ends, instead of waiting for the next clip or the daily sync.

To update: the Update button on the Hardware page, or by hand:
`uv tool install --python 3.12 --force ./wakesmith-0.4.13-py3-none-any.whl`, then `wakesmith setup`.

Earlier: device clips as they arrive, sorting, deleting and trimming them, and takes recorded on
the device in [0.4.12](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.12),
a training recipe that catches about twice as many real calls, recordings at any sample rate and the
"misses hide" note in
[0.4.11](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.11), the device inbox in
[0.4.10](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.10), candidates on every GPU at once and a kept candidate that catches calls in
[0.4.9](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.9), the listening check, the Real voices guide and updating from the page in
[0.4.8](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.8), a smith at the forge while you train and `test --model` keeping the project's results in
[0.4.7](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.7), Macs with Apple silicon and the folder summary in
[0.4.6](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.6), Also written as, play buttons that show what's playing and the version's folder in
[0.4.5](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.5), Wake words page menu fixes in
[0.4.4](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.4), macOS training fixes (workers, RAM, open-file limit) in
[0.4.3](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.3), macOS job start and Training page fixes in
[0.4.2](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.2), install fixes in
[0.4.1](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.1), German wake words (experimental) in
[0.4.0](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.0).
