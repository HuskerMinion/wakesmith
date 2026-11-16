## What's new in 0.4.10

**Devices can send their clips to wakesmith.** Clips your devices save when they fire (or nearly fire)
are the best training data there is, but until now they had to be copied off by hand or by a script.
wakesmith now has an inbox for them: on the Hardware page, under *Device upload*, press *Create token*,
then paste the URL and token into the device. On TECHO5 devices that is the setup page's *Send wake
clips to* → *wakesmith*, then *Test*. Each clip arrives as `<device>/<kind>/<name>.wav` with its `.json`,
the same layout as a device's own clip folder, so *Sync automatically* and *Device clips* pick them up
with nothing else to change. Sending a clip twice is harmless. The inbox is off until you make a token;
wakesmith keeps only a fingerprint of it, and *Revoke* turns it off again. Any other device can send to
it too (`POST /api/clips`, described in the README). Devices need to reach wakesmith, so start it with
`--host 0.0.0.0` on your own network; never expose it to the internet.

To update: the Update button on the Hardware page, or by hand:
`uv tool install --python 3.12 --force ./wakesmith-0.4.10-py3-none-any.whl`, then `wakesmith setup`.

Earlier: candidates on every GPU at once and a kept candidate that catches calls in
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
