## What's new in 0.4.12

**Device clips as they arrive.** With *As clips arrive* on (Device clips > Sync automatically, on by
default), a wake word syncs a minute after a device sends it a clip through the device upload, instead
of waiting for the daily sync, while wakesmith is running. Two wake words on one device share one
folder: each takes only the clips whose `model` starts with its own name.

**Sort, delete and trim device clips.** With clips from more than one device, a *Device* menu narrows
the list and Quick check to one. The trash button (press twice) deletes a clip for good: it leaves
training at once, and the next sync doesn't bring it back. The scissors button trims one: drag where
it starts and ends, play the part kept, save. The clip itself stays whole; *Use the labeled clips*
trains and tests on the part kept (press it again after trimming a clip it already used).

**Takes recorded on the device.** A device can send takes of the phrase recorded on purpose (kind
`take`, with `word`, `run`, `index`, `frame`): sync puts them straight into that word's recordings, no
listening needed; mark some held out there to test on them. Devices need a firmware that sends
them; see *Receiving clips from TECHO5 devices* in the README.

To update: the Update button on the Hardware page, or by hand:
`uv tool install --python 3.12 --force ./wakesmith-0.4.12-py3-none-any.whl`, then `wakesmith setup`.

Earlier: a training recipe that catches about twice as many real calls, recordings at any sample rate
and the "misses hide" note in
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
