## What's new in 0.4.0

**German wake words (experimental).** A wake word now has a language. German is off by default: turn it
on with **Offer German for new wake words** on the Data & licenses page (or
`WAKESMITH_EXPERIMENTAL_LANGUAGES=1`), then pick **Deutsch (experimental)** on the New wake word page or use
`wakesmith new "hallo computer" --language de`. A German guide is in [docs/deutsch.md](docs/deutsch.md).

- Piper's eight German voices (Thorsten in three models, Kerstin, Eva K, Karlsson, Ramona, Pavoque),
  about 0.5 GB, downloaded the first time a German word makes its clips. Each clip gets its own speed,
  noise and pitch shift.
- The voices run batched on the GPU: a Standard run's 55,000 clips take about 5 minutes on an RTX 5060 Ti.
  Without a GPU they run on the CPU in parallel processes.
- German sound-alikes (umlauts included, checked with German espeak-ng), Whisper in German for device
  clips and recordings, and `trained_languages: ["de"]` in the exported models.
- In our test, a model trained on the synthetic German voices alone caught a German voice it had never
  heard 88% of the time (microWakeWord, cutoff 0.5) with 3 false triggers in 14 hours of German speech at
  cutoff 0.9. Real recordings still help most, and more so than in English.
- Thorsten and Kerstin are CC0; Pavoque is CC BY-NC-SA, and Eva K, Karlsson and Ramona (M-AILABS) are
  treated as personal use only. The export's license note lists every voice a model was trained on.

**Fixes.**
- `wakesmith setup` without uv no longer passes pip a uv-only option when it installs PyTorch.
- Downloads no longer hang on networks where IPv6 has no route out: they fall back to IPv4, and every
  request has a timeout.

**Install:** download the `.whl` below, then `uv tool install ./wakesmith-0.4.0-py3-none-any.whl`,
`wakesmith setup` and `wakesmith serve` (see the README's Quick start). To update from 0.3.0, add `--force`
to `uv tool install`, then run `wakesmith setup` again.
