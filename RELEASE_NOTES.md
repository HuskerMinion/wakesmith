## What's new in 0.4.8

**Listening check: sound-alikes that are really the phrase are left out.** A sound-alike one letter
away from the phrase ("hey jarvus" for "hey jarvis") is often spoken exactly like the phrase by a
synthetic voice, and trained as "not the phrase" it teaches the model to ignore its own wake word. A
new step after the synthetic speech has Whisper listen to every sound-alike take and leaves out those
whose whole transcript is the phrase (one that only contains it, like "hey jarvis is", stays). In a
test on one wake word Whisper heard 7% of them as the phrase; three models trained without those caught
71-76% of real calls at up to 3 false triggers, against 41-71% for three trained with them, with the
same or fewer false triggers in hours of talk. It needs a GPU and Whisper's first download (without them
it's skipped and tried again on the next run) and takes
about half an hour for a Standard set on one card, once: retrains and new versions reuse it, and older
projects get it on their next run. The Training page shows how many takes were left out; turn it off
with `samples: check_negatives: false` in `project.yaml`. Whisper now comes with wakesmith (it was the
`whisper` extra), so device clips are transcribed without installing anything else.

**How to get good real voices.** The Real voices page has a short guide in six steps: where to record,
varied takes, everyone who'll use it, a quarter kept aside, room sound and echoes, and feeding back the
model's mistakes. It's open for a wake word without real voices until you close it.

**Update from the page.** When a new version is out, the sidebar says so and the Hardware page has an
Update button: wakesmith downloads the release's wheel, checks it against the sha256 GitHub lists, installs it into the
environment it runs from (PyTorch and the data stay, no `wakesmith setup` needed), checks that it
starts, and restarts itself; if the new version doesn't work the old one goes back. It waits while
anything is training or queued. To know a new version is out it asks GitHub when you open the page,
about twice a day; the same page turns that off. Works from the next release on: this one still installs by hand.

To update: `uv tool install --python 3.12 --force ./wakesmith-0.4.8-py3-none-any.whl`, then `wakesmith setup`.

Earlier: a smith at the forge while you train and `test --model` keeping the project's results in
[0.4.7](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.7), Macs with Apple silicon and the folder summary in
[0.4.6](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.6), Also written as, play buttons that show what's playing and the version's folder in
[0.4.5](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.5), Wake words page menu fixes in
[0.4.4](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.4), macOS training fixes (workers, RAM, open-file limit) in
[0.4.3](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.3), macOS job start and Training page fixes in
[0.4.2](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.2), install fixes in
[0.4.1](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.1), German wake words (experimental) in
[0.4.0](https://github.com/HuskerMinion/wakesmith/releases/tag/v0.4.0).
