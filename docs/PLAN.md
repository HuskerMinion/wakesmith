# wakesmith app plan (LiveKit-free)

Status: agreed 2026-10-20 (decisions at the bottom). Same repo and name. The current LiveKit-based recipe gets tagged `v0.1`
and stays reachable. `main` becomes the app.

## Goal

A local app with a web UI that takes someone from "I want the wake word *X*" to an ESPHome / Home
Assistant model file, on whatever hardware they have:

1. Detect the hardware and configure itself (Intel Arc, NVIDIA, CPU; cores, RAM, disk).
2. Phrase → preview TTS samples → suggested sound-alikes to reject.
3. Optional real recordings (upload or folder), held out by day, with the label review page.
4. Train with live progress and time estimates.
5. Score held-out clips, pick a cutoff, download the models: microWakeWord `.tflite` + ESPHome
   manifest, and an openWakeWord-compatible `.tflite`.

## What exists already (so we build the gap, not a clone)

- **TaterTotterson/microWakeWord-Trainer-Nvidia-Docker:** web UI, microWakeWord, actively
  maintained. NVIDIA only, and it has **no license**, so none of its code can be reused.
- **adman234/openwakeword-trainer:** MIT, Docker web UI, openWakeWord-first. NVIDIA, with a CPU
  fallback.
- **None of the existing trainers support Intel Arc.**

wakesmith's gap:
- Intel Arc and automatic hardware detection.
- The real-recordings workflow (review page, held-out-by-day scoring) that got your model from
  57/94 to 81/94 caught.
- Honest licensing.

## Stack (all verified 2026-10-20 unless marked)

| Job | Use | License | Verified here |
|---|---|---|---|
| Synthetic speech | Piper VITS generator models (`en_US-libritts_r-medium.pt`, also de/fr/nl) via **our own small TTS module**, derived from rhasspy/piper-sample-generator | MIT code; voices trained on CC BY 4.0 data | Runs on Arc: 2,000 clips in 16 s vs 64 s on CPU (~4×) |
| Augmentation, features, training, export | microWakeWord (mixednet → streaming int8 `.tflite`) | Apache-2.0 | Your v1 model: 81/94 caught, 8/69 false |
| Training compute | TensorFlow, CPU (Arc has no TF ≥2.18 plugin); NVIDIA GPU possible | Apache-2.0 | CPU: ~18 steps/s, 20k steps ≈ 20 min |
| openWakeWord model | openWakeWord's frozen frontend (`melspectrogram.onnx` + `embedding_model.onnx`, run with onnxruntime) → small DNN head **trained in Keras** → native TFLite export, `[1,16,96]` → `[1,1]` | Apache-2.0 code; frontend files' license ambiguous (see risks) | Your DNN v2 (via LiveKit): 68/94 caught, 2/69 false at 0.9 |
| One environment | torch XPU 2.14 + TF 2.21 + microWakeWord + numpy 2 coexist | — | Yes |

**Why our own TTS module instead of the piper-sample-generator package:**
- **No GPU flag.** It hard-codes `.cuda()` / MPS, so Arc needs a shim.
- **Broken PyPI package.** v3.2.0 on PyPI is missing its own `piper_train` module.
- **GPL dependency.** It depends on `piper-tts` 1.3.0, which is **GPL-3.0**, and that would pull
  GPL into an Apache-2.0 app.
- **Small job.** The generator code we need (VITS model + speaker mixing + batching) is MIT and a
  few hundred lines. We vendor it with attribution, add a `device` argument, and phonemize by
  calling the `espeak-ng` program, as the v0.1 recipe already did.

## Training data and licensing (shown in the UI)

| Data | License | Use |
|---|---|---|
| microWakeWord negative sets (speech, dinner_party, no_speech) | **CC BY-NC 4.0** | default "personal" profile (best quality) |
| MUSAN noise | CC BY 4.0 | both profiles |
| MIT room impulse responses | **no stated license** | personal profile; find a CC replacement for "open" |
| AudioSet / FMA (upstream notebook's noise) | unclear / partly NC | not used |
| openWakeWord ACAV100M negative features | **CC BY-NC-SA 4.0** | openWakeWord head, personal profile |
| Mozilla Common Voice (real volunteer voices, transcribed) | CC0 | both (planned, see Later) |
| Your recordings | yours | both |

- **Two data profiles:**
  - **Personal** (default): best models, non-commercial use, and the UI says so.
  - **Open** (later): CC BY/CC0 sources only (MUSAN, LibriSpeech, a CC-licensed RIR set), for
    people who need a commercially clean model.
- **App code** stays Apache-2.0.

## Hardware detection and auto-configuration

1. **Before installing anything:** read `lspci` vendor IDs (Intel `8086`, NVIDIA `10de`, AMD
   `1002`; skip integrated GPUs) plus cores, RAM and free disk.
2. **Install the matching PyTorch build** (XPU / CUDA / CPU). TensorFlow uses CUDA on NVIDIA if it
   works, otherwise CPU.
3. **Verify after install:** torch reports device names, memory and counts.
4. **Derive settings:**
   - TTS batch size from GPU memory.
   - One generation process per GPU (positives and sound-alikes split across cards).
   - Augmentation and feature workers = cores, capped by RAM (~3.5 GB per TTS process).
   - Warn when there's less than ~40 GB of disk (the LiveKit-free pipeline drops the 17 GB of
     ACAV100M features).
5. **`wakesmith doctor`** prints all of this, and the web UI shows the same thing on its first
   screen.

## Architecture

- **One Python package**, `wakesmith` (src layout), installed with `uv tool install`.
  `wakesmith setup` picks and installs the right torch build.
- **CLI:** `wakesmith doctor | setup | new | train | test | serve | export`. Everything the UI does,
  the CLI does.
- **Projects:** one folder per wake word, holding `project.yaml`, clips, features, models, review
  verdicts and logs. Stages are resumable: generate → real → augment → features → train → test →
  export.
- **Jobs:** one background worker process per training run, writing progress events. The web
  server streams them to the browser (server-sent events). Training survives closing the browser.
- **Web server:** FastAPI plus plain HTML/JS, with no build step.
  - Pages: Doctor, Project (phrase, TTS preview, sound-alikes), Recordings, Train, Review (the
    existing page, generalized), Results (held-out table and cutoff for each model type), Export.
  - Binds to localhost. A LAN mode needs a generated access token, since the review page plays
    private recordings.
- **Sound-alike suggestions:** phoneme-neighbor phrases from CMUdict plus the bare name without
  "hey". Shown as editable chips, with TTS preview for each.
- **Export:**
  - microWakeWord: `.tflite` + ESPHome manifest (cutoff, sliding window 5, arena size), plus a
    copy-paste ESPHome YAML snippet.
  - openWakeWord: `.tflite` + manifest, for Wyoming openWakeWord / Home Assistant.

## Milestones

1. **Core pipeline (CLI only, no UI).**
   - TTS module with device support; sound-alike generator; multicore augmentation and features
     (reused from v0.1's `mww_features.py`); training; held-out test; export.
   - openWakeWord head: features from the frozen frontend (batched, OpenVINO on Intel GPUs), a
     Keras DNN, and a native `.tflite` export.
   - **Accept when:** a "Hey Alfred" retrain on your data matches or beats both deployed models on
     the same held-out clips:
     - microWakeWord: ≥81/94 caught at ≤8/69 false triggers (v1).
     - openWakeWord: ≥68/94 at ≤2/69 (DNN v2).
     That's the proof that dropping LiveKit costs nothing.
   - **Status (2026-10-20):** passed on the reference (TFLite Micro) frontend, but **not** on the
     devices' frontend. The deployed devices run zserge/microwakeword, whose float frontend differs
     from the microfrontend microWakeWord trains on. Scored the way those devices hear the clips
     (`wakesmith test --frontend zserge`), on the full held-out recordings: v1 catches 86/94 at
     4/69 false (0.80), the new model 74/94 at 4/69. v1 stays live. Open question: match the
     frontends on the device side, or train on device features (the downloaded negative sets are
     microfrontend features only, so that needs raw-audio negatives).
2. **Hardware doctor + `setup`.**
   - **Accept when:** auto-configured correctly on this machine (2 × Arc), and in a CPU-only mode
     (`--cpu`).
   - NVIDIA is implemented but marked untested until someone with a card tries it.
   - **Status (2026-10-20): passed.** `doctor` finds both Arc cards (TTS on xpu:0 and xpu:1,
     batch 100, 12 workers). `--cpu` (or `WAKESMITH_CPU=1`) on doctor, train and serve gives
     TTS on cpu at batch 16 and generated a small project on the CPU. Under 40 GB free disk,
     doctor, train and the UI warn.
3. **Web UI + job runner.**
   - **Accept when:** a fresh phrase goes from phrase entry to downloaded model in the browser,
     with progress, on this machine.
   - **Status (2026-10-20): passed.** Driven in headless Chromium, twice, with fresh phrases
     ("hey cobalt", "okay lantern"), Quick preset: phrase, Hear it, sound-alikes, create and
     train, live progress for every stage, export, downloads, in about 8 minutes with no page
     errors. Fixed on the way: training progress (the step lines microWakeWord logs are about
     50 KB apart), one zip per model (both models are named <project>.tflite), favicon 404.
4. **Release v0.2.**
   - README rewrite with screenshots, licensing page, and the banner.
   - Move the v0.1 recipe docs to a `v0.1` tag and link to it.
5. **Later (optional):**
   - "Open" data profile.
   - Docker images (NVIDIA, Intel).
   - Non-English voices (de/fr/nl generators exist).
   - Recording clips straight from the browser.
   - **Real voices from Common Voice** (being tried on the Computer model, 2026-10-21): for a phrase
     made of common words, find the takes whose transcript contains it (English: "computer" 1,314
     takes from ~918 speakers, "help" 2,053; "cancel" only 55, two-word wake phrases mostly none),
     cut the word out by forced alignment on the known transcript (torchaudio MMS_FA, no new
     dependency), and offer it in the data step: "Common Voice: N takes, M speakers" with *use as
     test* (an honest recall test on real strangers) and, separately, *also train on them*. Only
     offered when the count is worth it. Open: the takes are spread over every audio shard (47 GB
     for English validated), so the app needs a one-time byte-offset index of the tars to fetch
     only the takes it needs with HTTP range requests; and the source (Mozilla took Common Voice
     off Hugging Face in Oct 2025; it is now on the Mozilla Data Collective, which needs an account
     and accepted terms; the trial uses the fsicoli/common_voice_17_0 mirror). Add only if the
     Computer trial shows it helps.

## Risks and unknowns

- **NVIDIA:** torch CUDA and TensorFlow CUDA in one environment can clash on CUDA library
  versions. Fallback: TF on CPU.
- **XPU TTS failures:** the "index out of bounds" batch failures seen with LiveKit's Piper
  generator may also hit this generator. Mitigation: retry the failed batch at a smaller size,
  and report the count.
- **openWakeWord frontend license:** openWakeWord's README calls the embedding model Apache-2.0
  (from Google's speech_embedding), but also says "all included pre-trained models" are
  CC BY-NC-SA 4.0. It's unclear which applies to the two frontend `.onnx` files. Download them at
  runtime from openWakeWord's release and don't bundle them.
- **Licenses still unclear:** MIT RIRs, and piper-sample-generator's bundled impulse WAVs.
  Neither is redistributed: both are downloaded at runtime.
- **Synthetic-only models are weak.** Without real recordings, quality is limited (your old
  synthetic-only device model: 57/94). The UI should push people to the recordings workflow and
  state this plainly.
- **Disk and time:** a full run is ~25k clips per class. Time estimates come from the doctor's
  measured rates.

## Decisions (2026-10-20)

1. **License:** Apache-2.0, with our own TTS module built from piper-sample-generator's MIT code.
   No `piper-tts` (GPL) dependency.
2. **Default data profile:** Personal (non-commercial, best quality), labeled clearly in the UI.
   The "open" profile comes later.
3. **openWakeWord output:** in the first release, alongside microWakeWord.
4. **Install:** native first (`uv tool install wakesmith` + `wakesmith setup`). Docker later.
