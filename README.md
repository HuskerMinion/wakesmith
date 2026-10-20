# wakesmith

![wakesmith: forge your own wake word](docs/banner.png)

Train your own wake word ("hey jarvis", "computer", anything) for **ESPHome / Home Assistant
Voice devices (microWakeWord)** and **openWakeWord**, from synthetic speech plus recordings from
your own home. It detects your hardware and sets itself up for Intel Arc, NVIDIA or a plain CPU.
Everything runs locally, from a web UI or the command line.

What makes it different:

- **Your real recordings count.** Mix in clips from your house, fix wrong labels on a review page,
  and get scored on clips held out by day: the honest number for how the model does at home.
- **Intel Arc support.** As far as we know, the only trainer that uses Intel GPUs. It also runs
  on NVIDIA and CPU.
- **No cloud, no Colab, no notebooks.** One install, one command, one page in your browser.

> Looking for the earlier LiveKit-based recipe? It's tagged
> [`v0.1`](https://github.com/HuskerMinion/wakesmith/tree/v0.1).

## Quick start

```bash
sudo apt install espeak-ng git          # Debian/Ubuntu; espeak-ng phonemizes the TTS text
uv tool install git+https://github.com/HuskerMinion/wakesmith
wakesmith setup                         # right PyTorch for your GPU; downloads ~24 GB of data
wakesmith serve                         # then open http://localhost:8765/
```

In the browser:

1. **New wake word:** type the phrase, press *Hear it* to check how the synthetic voices say it,
   then *Suggest sound-alikes*. That gives a list of near-misses the model should reject; edit it
   freely.
2. Optionally point it at **folders of your own recordings** (see below).
3. **Train.** Progress is shown live. A standard run takes about an hour on a 12-core CPU with
   an Arc GPU, mostly CPU-bound training.
4. **Results:** pick a cutoff from the table of held-out results, then **Export**. You get a
   `.tflite` + manifest for ESPHome (plus a YAML snippet) and an openWakeWord `.tflite`.

Everything the UI does is also on the command line:

```bash
wakesmith doctor                        # hardware, settings, downloaded data and licenses
wakesmith new "hey jarvis" --positives ~/rec/wake --negatives ~/rec/false
wakesmith train hey_jarvis              # resumable: generate real mww-features mww-train
                                        #            oww-features oww-train test export
wakesmith test hey_jarvis               # held-out results table
wakesmith export hey_jarvis --cutoff 0.9
wakesmith review hey_jarvis             # list recordings the model disagrees with
wakesmith train hey_jarvis --from real  # retrain after recordings or labels changed
```

Projects live in `~/.local/share/wakesmith/projects/<name>/`, one folder each, with all settings
in `project.yaml`.

## Your recordings

Synthetic speech gets you a working model. Real audio from the rooms your devices sit in is what
makes it good. Two folders, read-only to wakesmith:

- **Positives:** people saying the phrase to a device.
- **Negatives:** anything that is not the phrase. The best negatives are what fooled an earlier
  model: have your current wake word engine save a few seconds of audio whenever it scores above a
  low threshold, then label those clips.

```
my_positives/
  labels.csv     optional: a "file" column (paths relative to this folder); other columns show on
                 the review page
  HELDOUT.txt    optional: one path per line. Never trained on: these are what you're scored on
  kitchen/clip-0001.wav ...   16 kHz mono WAV, any length, any subfolders
```

**Hold out whole days, not random clips.** Clips from the same day share voices, noise and even
the same utterance caught twice; a random split leaks that into training and flatters the score.

**Review labels.** After a first training run, *Build review list* finds the recordings the model
disagrees with, which are usually labelling mistakes. Mark what was really said, then *Retrain
from recordings*. Held-out clips are never moved, so scores stay comparable between runs.

## Results from one household

The same 163 held-out real clips (94 times someone said "Hey Alfred", 69 real false triggers of
an earlier model, recorded on two days that were never trained on), scored the way the devices run
the models:

| microWakeWord | False triggers (of 69) | Caught (of 94) |
|---|---|---|
| Earlier LiveKit-based recipe (v0.1) | 8 | 81 (cutoff 0.97) |
| **wakesmith** | 8 | **84** (cutoff 0.85) |
| **wakesmith** | 4 | **79** (cutoff 0.93) |

| openWakeWord | False triggers (of 69) | Caught (of 94) |
|---|---|---|
| Earlier LiveKit-based model, in an openWakeWord runtime | 2 | 36 (threshold 0.9) |
| Earlier LiveKit-based model, in an openWakeWord runtime | 7 | 68 (threshold 0.5) |
| **wakesmith** | 2 | **76** (threshold 0.5) |

The earlier openWakeWord model had been trained on audio at a different scale than the openWakeWord
runtime feeds it (±1.0 floats instead of 16-bit sample values), so on a real runtime it lost about
half its recall at strict thresholds. wakesmith extracts features exactly as the runtime does.

One home, one phrase, a small test set: read these as a direction, not a benchmark.

## Hardware

`wakesmith setup` looks at your PCI devices before installing anything, picks the matching PyTorch
build, then checks the GPU really works and falls back to CPU if not. `wakesmith doctor` shows
what it found and the settings that follow: TTS batch size from GPU memory, one generation process
per GPU, and worker counts from free RAM.

| | Synthetic speech | Training | Status |
|---|---|---|---|
| Intel Arc (XPU) | GPU, ~50-130 clips/s per card | CPU | tested: 2 × Arc Pro B70 |
| NVIDIA (CUDA) | GPU | CPU (TensorFlow GPU if it works) | not tested yet: reports welcome |
| CPU only | ~3-30 clips/s | CPU | works, slow to generate |

Both trainers run on the CPU: microWakeWord needs TensorFlow 2.18+, which has no Intel GPU
support, and the openWakeWord head is small. Disk: about 43 GB for the shared training data
(unpacked) plus about 23 GB per project, mostly microWakeWord features. RAM: tested on a 29 GB
machine; worker counts follow free RAM, so close other big programs (e.g. a local LLM) during
training. More cores make feature building and training faster.

**Intel GPU note:** a crashed GPU job can leave the `xe` driver stuck, after which anything that
opens the GPU hangs until a reboot. wakesmith guards against the crash we found (an invalid index
in the TTS model's duration predictor), runs previews on the CPU, and times out GPU detection
instead of hanging. If `wakesmith doctor` says the GPU driver isn't responding, reboot.

## Licensing

wakesmith is Apache-2.0. Its TTS code is derived from
[piper-sample-generator](https://github.com/rhasspy/piper-sample-generator) (MIT); phonemes come
from the `espeak-ng` program, run as a separate process.

Training data is downloaded at setup, never bundled; `wakesmith doctor` lists each item and its
license. The default **personal** data profile uses datasets with **non-commercial** terms
(microWakeWord's negative sets: CC BY-NC 4.0; openWakeWord's ACAV100M features: CC BY-NC-SA 4.0),
so models trained with it are for personal use. Exports include a note saying so.

## Credits

[microWakeWord](https://github.com/kahrendt/microWakeWord) (Kevin Ahrendt),
[openWakeWord](https://github.com/dscripka/openWakeWord) (David Scripka),
[Piper](https://github.com/rhasspy/piper) and piper-sample-generator (Michael Hansen),
[CMUdict](https://github.com/cmusphinx/cmudict), MUSAN, and the MIT room impulse response survey.
