## What's new in 0.4.11

**A training recipe that catches about twice as many real calls.** microWakeWord models now train
with the changes that held up best when wakesmith's recipe, Wake Word Studio's and Tater's trainer
settings were compared one piece at a time on one household's real calls: masking (SpecAugment), a
little more EQ, pitch and color noise, a gain sweep inside some clips, real read speech (Google
FLEURS, CC BY 4.0) as background as well as MUSAN, and 0.10-0.21 s of quiet after the phrase, so
the model fires 20-50 ms sooner. Real clips train as their own set with their quiet cut out, and a
real false trigger costs 3x a missed call. With the same few false wakes allowed (5 on 447 clips),
the middle of 8 candidates caught 45-56 of 97 real calls across four runs, against 24 before. Each
piece is a `project.yaml` setting (see *Training recipe* in the README). A wake word made with an
earlier wakesmith keeps its saved real-clip copies and weight and gets the rest on its next training
run. FLEURS is a new 1.4 GB download: `wakesmith setup` fetches it, or the first training run does.

**Recordings at any sample rate.** Folders of your own clips and device clip imports no longer stop
at 44.1 kHz or stereo WAV files: wakesmith converts them to 16 kHz mono as it reads them (4-768 kHz,
up to about 75 minutes of mono or 38 of stereo at 44.1 kHz, longer at lower rates). A clip that
can't be read is left out and counted instead of stopping the import. The device upload
(`POST /api/clips`) still takes 16 kHz mono, as devices are told.

**Misses hide.** A false wake is loud; a missed call is silent. When most of a test's calls are
clips your devices saved (which only holds calls a model fired or nearly fired on), Results now says
the misses are likely undercounted, and points to *Record a room* for a fair count.

To update: the Update button on the Hardware page, or by hand:
`uv tool install --python 3.12 --force ./wakesmith-0.4.11-py3-none-any.whl`, then `wakesmith setup`.

Earlier: the device inbox in
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
