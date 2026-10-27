## What's new in 0.3.0

**License.** From 0.3.0 wakesmith is under the [PolyForm Noncommercial License 1.0.0](LICENSE): use it,
change it and share it for any noncommercial purpose. 0.2.0 and earlier stay Apache-2.0. Code taken from
other projects keeps its own license ([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).

**Fast on a GPU, Intel Arc included.** Synthetic speech, microWakeWord training (a PyTorch port of its
trainer), openWakeWord features and openWakeWord training all run on the GPU: a full-size run takes about
15 minutes where it took about an hour. See [Speed](README.md#speed) in the README.

**Easier to follow.**
- The wake words page is a table: one row per wake word, its versions folded underneath, and a line
  saying what to do next.
- A welcome screen on first run, and the training data downloads from the Data & licenses page.
- Every round you train is a version (v1, v2, ...) with a short note; **New version** makes the next one in
  one step from the version before and the device clips that came since.
- After a test, the version on your devices is scored on the same recordings, and Results says in one
  sentence whether the new one is better. A slider turns each cutoff into calls missed and false alarms a
  day.
- Real voices shows everything a version learns from and is tested on, and plays any clip. Takes recorded
  in the browser always add to the folders you chose.

**Better models.**
- Best of N: a Standard run trains several microWakeWord candidates and keeps the one that does best on
  half of your test recordings; its results come from the other half.
- Optionally, one combined model that averages the candidates.
- Device clips: bring in what your devices saved, transcribed by Whisper; clear false alarms label
  themselves and you listen to the rest (one at a time, with the keyboard). A daily sync can do it for you.
- Record a room: one recording for every wake word, split into takes automatically, with the room's own
  echoes for training.

**Also:** a job queue with finished notices, archiving of idle projects to save disk, an optional local AI
assistant (explains failed runs, advises on a cutoff), and Common Voice takes for common words.

**Install:** download the `.whl` below, then `uv tool install ./wakesmith-0.3.0-py3-none-any.whl`,
`wakesmith setup` and `wakesmith serve` (see the README's Quick start).
