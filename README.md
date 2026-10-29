# wakesmith

![wakesmith: forge your own wake word](docs/images/banner.jpg)

Train your own wake word ("hey jarvis", "computer", anything) for **ESPHome / Home Assistant Voice
devices (microWakeWord)** and **openWakeWord**, from synthetic speech plus real voices: your own
takes, recorded right in the browser, clips your devices saved, and volunteers from Mozilla Common
Voice. It detects your hardware and sets itself up for Intel Arc, NVIDIA, AMD (experimental) or a
plain CPU, and runs every heavy stage on the GPU when there is one. Everything runs locally, from a
web page or the command line.

wakesmith is free for noncommercial use under the [PolyForm Noncommercial License](#licensing): use
it, change it and share it, as long as no one makes money from it.

What makes it different:

- **Real voices, made easy.** Record takes in the browser (near, far, quiet, TV on) with a level
  meter and speaker names, point it at folders of clips your devices saved, or add thousands of
  real strangers from Common Voice for common words. Some takes are always kept aside, so the test
  is honest.
- **Honest numbers.** Every model is scored on recordings it never trained on, the way the device
  runs it (ESPHome's audio front end, or openWakeWord's runtime), and every run is kept so you can
  compare them side by side before you pick a cutoff.
- **Fast on a GPU, including Intel Arc.** Synthetic speech, feature building and training for both
  model types run on the GPU: a full-size model in about 15 minutes on an RTX 5060 Ti, where the
  original tools took about an hour ([Speed](#speed)). As far as we
  know, it is the only trainer that uses Intel GPUs. It also runs on a CPU alone.
- **No cloud, no Colab, no notebooks.** One install, one command, one page in your browser.

> Looking for the earlier LiveKit-based recipe? It's tagged
> [`v0.1`](https://github.com/HuskerMinion/wakesmith/tree/v0.1).

## Quick start

```bash
sudo apt install espeak-ng git          # Debian/Ubuntu; espeak-ng phonemizes the TTS text
# macOS instead: brew install espeak-ng git
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv, if you don't have it; then open a new shell
uv tool install --python 3.12 ./wakesmith-<version>-py3-none-any.whl   # the wheel from the Releases page
wakesmith setup                         # right PyTorch for your GPU; downloads ~24 GB of data
wakesmith serve                         # then open http://localhost:8765/
```

wakesmith needs Python 3.10 to 3.12 (TensorFlow and microWakeWord's dependencies have no packages for
newer ones); `--python 3.12` makes uv fetch it if needed. On macOS, install the tools with
`brew install espeak-ng git` instead of apt. macOS hasn't been tested yet, and setup doesn't use Apple
GPUs: everything runs on the CPU (see [Status](#status)).

To update, run `uv tool install --python 3.12` on the newer wheel with `--force`, then `wakesmith setup` again: the
reinstall starts a fresh environment without PyTorch, and setup puts the right build back (the data
stays).

To use it from another computer or a phone on your LAN, run `wakesmith serve --host 0.0.0.0` and
open the link it prints (it carries an access token, which stays the same between restarts).
Recording in the browser needs the microphone, which browsers only allow on localhost or HTTPS: add
`--https` (a self-signed certificate; accept the browser's warning once). Changes are refused from
pages of other sites, so behind a reverse proxy keep the Host header as the browser sent it.

**Windows:** use WSL2 (Ubuntu) and run the same commands inside it. NVIDIA's Windows driver gives
WSL2 CUDA, and wakesmith finds the card through `nvidia-smi` there (tested on an RTX 5060 Ti).
The page then opens in your Windows browser at the printed link. In the NVIDIA Control Panel, set
Manage 3D Settings > **CUDA - Sysmem Fallback Policy** to **Prefer No Sysmem Fallback**: otherwise a job
that runs out of video memory spills into system memory, slows until Windows resets the GPU, and can
freeze the PC. If something else on Windows
already uses port 8765, add `--port 8780`.

You need about 16 GB of RAM (microWakeWord's training holds its negative sets in memory), 50 GB of
free disk plus about 20 GB per project while it trains, and Python 3.10-3.12.
Archiving a project you aren't training (`wakesmith archive PROJECT`, or Hardware > Storage) removes
what training makes again by itself, mainly its features, and keeps recordings, results and models;
the next Train rebuilds the features first. `wakesmith storage` lists what each project uses and could
free. Projects idle for 14 days are archived automatically (setting on the Storage panel, 0 = off),
except those with a model marked offered or live; synthetic clips are only removed on request
(`--clips`), and never while another project links to them.

One training run uses the whole machine, so runs go one at a time. While one is running, the Train
button of another wake word reads **Queue this run**: queued runs start one after another, in the order
shown in the Queue panel (Wake words and Training pages), also while `wakesmith serve` is stopped. When a
run finishes, fails or is stopped, the page shows a notice the next time you look at it; **Tell me when
runs finish** in the Queue panel adds a desktop notification.

## How it works

![Your wake words](docs/images/dashboard.png)

1. **What is it for?** Pick ESPHome voice satellite, Home Assistant (openWakeWord), an Echo-style
   or other device, or "just trying it out". That sets the model type, the export and the defaults.
2. **The phrase.** Type it, *Hear it* in a synthetic voice, and get a list of sound-alikes the
   model must learn to ignore ("commuter" for "computer"). The page warns about phrases that are
   too short, too long, or everyday words.

   ![New wake word](docs/images/new-wake-word.png)

3. **Real voices** (optional, and the biggest gain): record takes here, add folders, or Common Voice.
4. **Train.** Live progress for every stage, with the time left measured on your machine and live
   RAM and GPU memory. A Quick test takes about 8 minutes with a GPU and 20 on six CPU cores; a
   Standard model about 15 minutes with a GPU. A Standard run trains three microWakeWord candidates
   and keeps the one that does best on half of your held-out recordings (runs of one recipe differ a
   lot); its results come from the other half, which played no part in choosing, so the choice
   doesn't flatter them.
   Start another run while one is going and it waits in the queue; finished and failed runs show a
   notice, in the browser too if you allow it.

   ![Training progress](docs/images/training.png)
5. **Results and export.** Caught and false triggers at every cutoff, broken down by test set (your
   recordings, device clips, sound-alikes...) so a weak spot shows even when the totals look fine,
   every run side by side, then a `.tflite` with its manifest for ESPHome (plus a YAML snippet)
   and/or an openWakeWord `.tflite`. The Models list keeps every exported model with its checksum,
   cutoff and numbers; mark one offered or live as it moves to your devices.

   **Combined model.** A Standard run trains three microWakeWord candidates; wakesmith can also
   average them into one model (Advanced: "Combine the candidates", or `wakesmith combine PROJECT`).
   Runs from different seeds make different mistakes, so averaging can smooth them out; whether it
   helps depends on the round. On one household's held-out recordings, allowing the same 5 false
   triggers, one set of candidates combined caught 21 takes said nearby, 144 said far away and 186
   said over TV, against 18, 123 and 163 for the best single candidate; another round's combined
   model did no better than its best single one. So check it on Results before choosing it. It is
   tested next to the single model there, about 3x the size with 3x the tensor arena, and runs anywhere a
   single microWakeWord model runs (`wakesmith export PROJECT --combined`). A new Standard run keeps
   the previous round's candidates (their weights and models, a few hundred KB each) and combined
   model in `mww/*.previous`; `wakesmith combine PROJECT --previous` combines that round.

![Results: the cutoff in everyday terms, and every run side by side](docs/images/compare.png)

There is also a Help page in the app, and every step works from the command line:

```bash
wakesmith doctor                        # hardware, settings, downloaded data and licenses
wakesmith new "hey jarvis" --positives ~/rec/wake --negatives ~/rec/false
wakesmith train hey_jarvis              # resumable: generate real commonvoice conversational
                                        #   mww-features mww-train oww-features oww-train test export
wakesmith test hey_jarvis               # held-out results table
wakesmith export hey_jarvis --cutoff 0.9
wakesmith combine hey_jarvis            # one model averaging the best-of-N candidates
wakesmith combine hey_jarvis --previous # ... or the previous round's candidates
wakesmith review hey_jarvis             # recordings the model disagrees with (label mistakes)
wakesmith train hey_jarvis --from real  # retrain after recordings or labels changed
wakesmith commonvoice-count "computer"  # how many Common Voice takes say it
```

Projects live in `~/.local/share/wakesmith/projects/<name>/`, one folder each, with all settings in
`project.yaml`.

### AI assistant (optional)

Point wakesmith at a language model you run yourself (llama.cpp, Ollama, LM Studio, vLLM: anything
with an OpenAI-style `/v1/chat/completions`) on the Hardware page, and it helps in three places.
It is off until you turn it on; nothing is sent anywhere else, and audio never is.

- **Why it stopped** (Training): wakesmith's own checks always name the common failures (out of
  memory, disk full, a GPU runtime problem, a missing download); *Explain this failure* sends the
  failed stage's log lines to your model for a plain-language cause and what to try.
- **Advice** (Results): your results table goes to the model, which suggests a cutoff, whether to
  ship, and what to record next. It only advises, and shows the numbers it was given.
- **Everyday sentences** (experimental, off): the model writes ordinary sentences that use words of
  your phrase without saying it ("can you hand Jarvis the dishes" for "hey jarvis"); the synthetic
  voices speak them as not-the-phrase clips. Set the number of clips on the Training page.

## Where the models run

- **ESPHome voice satellites** (Home Assistant Voice PE and others): the microWakeWord `.tflite` with
  its `.json` manifest, plus a YAML snippet.
- **Home Assistant with openWakeWord** (the Wyoming add-on): the openWakeWord `.tflite`.
- **Kiosk Satellite / the Voice Satellite card:** the microWakeWord export (`.json` + `.tflite`) loads
  there as a custom wake word as it is (a wakesmith microWakeWord model has run on a tablet with
  Kiosk Satellite).
- **Other microWakeWord engines** (see [below](#results-from-one-household)): exported models record the
  audio front end they were trained on.

## Other languages (experimental)

Eine Anleitung auf Deutsch: [docs/deutsch.md](docs/deutsch.md).

German wake words are off by default. Turn them on with the switch on the **Data & licenses** page (or
`WAKESMITH_EXPERIMENTAL_LANGUAGES=1`, then `wakesmith new "hallo computer" --language de`); the New wake
word page then has a Language choice. A German word gets:

- **Piper's German voices**: Thorsten (three models, one with eight speaking styles), Kerstin, Eva K,
  Karlsson, Ramona and Pavoque, about 0.5 GB, downloaded the first time a German word makes its clips
  (`wakesmith setup` leaves them out). Piper ships them as ONNX; wakesmith rebuilds them as PyTorch
  models so they run batched on the GPU like the English ones (55,000 clips in about 5 minutes on an
  RTX 5060 Ti; without a GPU, onnxruntime in parallel processes). Each is one speaker, so every clip
  gets its own speed, noise and a pitch shift (+/-12%). In a test, a model trained on seven of them
  caught the eighth, a voice it had never heard, 88% of the time (microWakeWord, cutoff 0.5); English
  voices reading German made it worse on that voice, so they aren't used. Still, record real takes:
  they matter even more here than in English. Piper's 236-speaker German voice (de_DE-mls-medium) is
  left out: it says nothing intelligible.
- **Sound-alikes** from one-letter-sound changes (umlauts included), checked with German espeak-ng so
  none is pronounced like the wake word. There is no German pronouncing dictionary behind them yet.
- **Whisper in German** for device clips and recording sessions, and `trained_languages: ["de"]` in
  the exported model files.
- **Everyday sentences** (if turned on) asked for in German. Common Voice isn't offered: the set
  wakesmith downloads is the English one.

Thorsten and Kerstin are CC0; Pavoque is CC BY-NC-SA 4.0; Eva K, Karlsson and Ramona come from the
M-AILABS Speech Dataset, whose terms Piper points to without stating them, so wakesmith treats them as
personal use only. The negative examples (other speech, noise) are the same as for English.

## Real voices

Synthetic speech gets you a working model. Real voices from the rooms your devices sit in are what
make it good, and they are the only honest test.

**Record in the browser.** Real voices > Record here: a prompt for each take (close, across the room,
TV on, quietly...), a level meter, automatic stop when you finish, a name per speaker, and every
fourth take kept aside for testing (switch any take between train and test). The microphone is
recorded raw, without echo cancelling or automatic gain, closer to what a device hears.

![Recording takes in the browser](docs/images/record.png)

**Record a whole room.** With several wake words you don't have to open each one: *Record a room*
makes one recording for all of them. Name the room and who's speaking, tick the wake words, put the
phone next to the device and press Start. Then follow the script on the page (it can be printed):
ten seconds of quiet, three claps from the middle of the room, then each wake word three times from
each spot, saying the spot out loud first ("across the room"). The page keeps the screen on and
uploads the audio every ten seconds, so nothing is lost if it closes. After Stop, wakesmith finds the
takes, transcribes them with Whisper (on the GPU unless a training run is using it) and guesses what
each one is; you listen, fix the guesses and press *Send to the wake words*. Each take goes to its own
wake word with the room and spot as its condition, every fourth one kept aside for testing; room
sound and other talk go to every wake word as not-the-phrase clips. One wake word is never used as
a not-the-phrase clip for another. The claps become this house's own room echoes: turn on *Use this
house's room echoes* on a wake word's Training page (`microwakeword.house_rirs`) and training mixes
them in as often as the standard set. Sending twice adds nothing twice. From the command line:
`wakesmith room list|split|send`.

**Folders.** Clips your devices saved, or any 16 kHz mono WAV files:

```
my_positives/
  labels.csv     optional: a "file" column (paths relative to this folder); other columns show on
                 the review page
  HELDOUT.txt    optional: one path per line. Never trained on: these are what you're scored on
  kitchen/clip-0001.wav ...   any length, any subfolders
```

The best negatives are what fooled an earlier model: have your current wake word engine save a few
seconds of audio whenever it scores above a low threshold, then label those clips. **Hold out whole
days, not random clips:** clips from one day share voices, noise and even the same utterance caught
twice, so a random split flatters the score.

**Mozilla Common Voice** (CC0) for phrases made of common words. wakesmith finds the takes whose
transcript contains your phrase, cuts the phrase out with forced alignment, and splits them by
speaker: 40% of the speakers are a test set it never trains on, and you can add the rest to
training. `wakesmith commonvoice-download` fetches Common Voice 17.0 English (about 47 GB, once);
the first training run with it cuts the phrase out (about 10 minutes), later runs reuse the cuts.
Single common words have plenty ("computer": 1,216 takes from 859 speakers); two-word wake phrases
usually have none, and the page tells you so.

![Common Voice takes for a phrase](docs/images/common-voice.png)

**Device clips: the next round.** The best negatives are what fooled the current model, and devices
that save a clip when they fire collect them for you. Real voices > Device clips imports a folder of
such clips (a `.json` next to each with `at`, `device`, `model`, `score` is read), transcribes them with
Whisper in the background (install the `whisper` extra; faster-whisper, Home Assistant's engine, is used
when it's installed). Clips whose transcript has none of the phrase's words are labeled "not the phrase"
by themselves; the rest wait under *Listen*, because only a person can tell a call from the TV saying the
phrase. *Quick check* plays them one at a time (Y the phrase, N not the phrase, L leave out). Then choose
where the held-out days start, *Use the labeled clips*, and *Retrain from recordings*, or let *New
version* do all of that (below). From the command line:
`wakesmith clips import|transcribe|accept|status|apply PROJECT ...`.

To keep up without doing that by hand, fill in *Sync automatically* on the same tab: the folder new
clips arrive in, the start of the `model` name your devices write for this word (a shared folder holds
every word's clips), and optionally the time the current model went live. *Sync now* brings in only
the clips not seen before and transcribes them; *Every day at* runs the same for every wake word with a
folder set (a systemd user timer). The folder is only read. The Wake words card shows how many clips
wait for you. From the command line: `wakesmith clips sync PROJECT [--from DIR] [--model-prefix NAME]
[--since TIME] [--transcribe-device auto|cpu|xpu:N|cuda:N] [--save]`, `wakesmith clips sync-all`, and
`wakesmith clips schedule [--at 03:30] [--off] [--status]` (log: `clips-sync.log` next to the projects).
With `auto`, Whisper uses the GPU only while no other wakesmith job runs; a wake word whose own job
is running is skipped until the next run.

**Versions.** Every round of training is a version of its wake word (v1, v2, ...), with a short note on
what it tried; the wake words page shows one row per word and folds out its versions. *New version* (a
word's ⋯ menu, or `wakesmith version PROJECT`) makes the next round in one step: the recipe, recordings
and synthetic voices of the version before, its device clips plus any new ones, the newest day kept aside
for the test, then training. After its test, the model on your devices is scored on the same held-out
clips and Results says in one sentence whether the new version is better; a slider there turns each
cutoff into calls missed and false alarms a day. The line under each word on the wake words page says
what to do next.

**Review labels.** After a first run, *Build review list* finds the recordings the model disagrees
with, which are usually labeling mistakes. Mark what was really said, then *Retrain from
recordings*. Held-out clips are never moved, so scores stay comparable between runs.

## Results from one household

**A custom two-word wake word, microWakeWord.** The same 163 held-out real clips (94 times someone said it, 69 real
false triggers of an earlier model, recorded on two days that were never trained on), scored the
way ESPHome runs the models:

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
runtime feeds it, so on a real runtime it lost about half its recall at strict thresholds.
wakesmith extracts features exactly as the runtime does.

**"Computer", openWakeWord, with Common Voice.** Tested on takes from Common Voice speakers it never
trained on, at threshold 0.5:

| openWakeWord "Computer" | Caught |
|---|---|
| A published community model | 56% |
| wakesmith, synthetic voices only | 28% |
| **wakesmith, + 795 Common Voice takes from 506 other speakers** | **76%** |

On a real Echo-style device's own saved clips, the wakesmith model fired on 2 of 43 wakes where the
community model fired on 30 at the same threshold (21 at the 0.99 it runs at on that device).

**Other microWakeWord engines can disagree.** Some devices run
[zserge/microwakeword](https://github.com/zserge/microwakeword), a pure-Go engine whose audio front
end approximates ESPHome's. `wakesmith test --frontend zserge` scores on that front end, and
exported models record the front end they were trained on in their metadata
(`mww_feature_frontend`), so such engines can pick the matching one.

**microWakeWord against vsWakeWord.** vsWakeWord is Kiosk Satellite's own engine (a phoneme
recognizer plus a matcher); these rows use its bundled models. Both engines run as Kiosk Satellite
runs them (its own detector settings, default sensitivity), on 24 hours of audio that isn't the
wake word: audiobooks, meetings, house device clips, TV, MUSAN noise and MUSAN music (two thirds
with vocals).

| False triggers in 24 hours | vsWakeWord | microWakeWord |
|---|---|---|
| "Okay Nabu", each engine's official model | 1 | 0 |
| "Hey Jarvis", each engine's official model | 0 | 2 |
| A custom two-word wake word: vsWakeWord's bundled model, wakesmith's microWakeWord | 0 | 5 (all on house clips) |

| The custom wake word caught, of 94 test takes | vsWakeWord (bundled) | wakesmith microWakeWord |
|---|---|---|
| Quiet room | 45 | 87 |
| Over noise: 10 / 5 / 0 dB | 37 / 32 / 18 | 80 / 73 / 64 |
| Over music: 10 / 5 / 0 dB | 32 / 22 / 10 | 63 / 50 / 34 |

The bundled vsWakeWord model almost never fires by mistake, and misses about half the calls in a quiet
room and most of them with music on. The takes come from the same devices, which favors the
wakesmith model; the official-model rows don't have that bias.

**Why wakesmith doesn't make vsWakeWord models, for now.** wakesmith had an experimental vsWakeWord
trainer for a while (October 2026), and it was taken out for 0.3.0. vsWakeWord's real strength is
that it hardly ever fires by mistake: its bundled model for that wake word didn't fire once in the 24 hours above
or on 69 recordings that had fooled an earlier model, where wakesmith's microWakeWord fired 5 times
in the 24 hours. The trouble is how many real calls it misses:

- **On a real device.** On a Samsung tablet running Kiosk Satellite, with the same people calling
  from around the same room, the bundled vsWakeWord model answered about 6 calls in 20.
  wakesmith's microWakeWord model, loaded into the same app as a custom wake word, answered about 9
  in 10, including from across the room with the speaker facing away. That matches the held-out
  takes above (45 of 94 against 87).
- **wakesmith's own vsWakeWord models** loaded and ran in Kiosk Satellite's own runtime, and the best
  one beat the bundled model, but still fell short: 74 of 94 takes with 2 of the 69 false triggers,
  where microWakeWord caught 83 with 4 at a strict cutoff. Using vsWakeWord as a second opinion on
  microWakeWord's detections did no better than simply raising microWakeWord's cutoff.
- **Kiosk Satellite already accepts microWakeWord models**, so its users get the higher catch rate in
  the same app without a second engine.

For a wake word that has to work when someone calls from across the room, missing half the calls
costs more than an occasional false trigger. If someone builds a better phoneme recognizer for it,
which is where the missed calls came from, vsWakeWord would be worth another look. Requests with
recordings are welcome.

One home, a few phrases, small test sets: read these as a direction, not a benchmark.

## Speed

Every heavy stage runs on the GPU; the original tools run most of them on the CPU. Same recipe, same
data, models scoring alike on test recordings:

| Stage | Original tools | wakesmith with a GPU |
|---|---|---|
| microWakeWord training, 20,000 steps | 13.6 min (its own trainer) | **3.2 min** (Arc Pro B70) · 7.5 min on the CPU alone |
| microWakeWord validation, each time | about 11 s | **about 0.4 s** |
| openWakeWord features, Standard project | 17.9 min (CPU) | **2.8 min** (Arc Pro B70) |
| openWakeWord training | 9.5 min (Keras, CPU) | **1.8 min** |
| Synthetic speech | 3-30 clips/s (CPU) | **about 80 clips/s** (RTX 5060 Ti), 50-130 per Arc card |
| **A whole Standard run** (both model types) | **about an hour** | **15 minutes** (RTX 5060 Ti) |

Measured on a Ryzen 5 7600X with an Arc Pro B70, and on an i9-13900K with an RTX 5060 Ti; the whole-run
figures are both on the RTX 5060 Ti machine, before and after training moved to the GPU. The details
are under Hardware below.

## Hardware

`wakesmith setup` looks at your PCI devices (on Linux and WSL2) before installing anything, picks the
matching PyTorch build, then checks the GPU really works and falls back to the CPU if not. `wakesmith doctor` (and
the Hardware page) shows what it found and the settings that follow: TTS batch size from GPU memory,
one generation process per GPU, and worker counts from free RAM. `--cpu` on `doctor`, `train` and
`serve` ignores the GPU.

| | Synthetic speech | microWakeWord training | Runs on |
|---|---|---|---|
| Intel Arc (XPU) | GPU, ~50-130 clips/s per card | GPU, ~3 min for 20k steps | 2 × Arc Pro B70, Ubuntu 26.04 |
| NVIDIA (CUDA) | GPU, ~80 clips/s on an RTX 5060 Ti | GPU, ~3 min for 20k steps | RTX 5060 Ti + i9-13900K, Windows 11 via WSL2 |
| AMD (ROCm) | GPU: supported Radeon cards on Linux | GPU | experimental |
| CPU only | ~3-30 clips/s | CPU, ~7 min for 20k steps (12 threads) | Ubuntu 24.04, 6 cores, 16 GB |

microWakeWord training runs in wakesmith's PyTorch port of its trainer, on the first GPU with at least
6 GB, else the CPU; microWakeWord itself still exports the model, so devices get the same streaming int8
file. Same recipe, same data: on a Ryzen 5 7600X + Arc Pro B70, 20,000 steps took 13.6 minutes in
microWakeWord's own trainer, 3.2 minutes on the GPU and 7.5 on the CPU alone, and the models scored alike
on held-out real recordings (runs of one recipe differ as much among themselves). The port builds
batches in worker processes and validates in ~0.4 s on a GPU (microWakeWord: ~11 s on the CPU).
`WAKESMITH_MWW_TRAINER=tf` runs microWakeWord's own trainer instead.

With every stage on the GPU, a full-size (Standard) ESPHome + openWakeWord run on the RTX 5060 Ti
machine took 15 minutes: synthetic speech 5.7, microWakeWord features 0.6 and training 2.9,
openWakeWord features 2.6 and training 3.1. Before training moved to the GPU the same run took about an
hour.

openWakeWord features use the same GPU: the CPU cores augment the clips and the GPU runs openWakeWord's
melspectrogram and speech-embedding models (ported to PyTorch, same features within float32 rounding).
On a Ryzen 5 7600X + Arc Pro B70 a Standard project's features took 2.8 minutes instead of 17.9 on
the CPU alone. `WAKESMITH_OWW_FEATURES=cpu` keeps them on the CPU. Finished features are reused
while their clips, recordings and settings are unchanged (delete `oww/features/stamp.json` to rebuild).
The openWakeWord head trains on the same GPU too, in a PyTorch port of the Keras recipe (same model,
optimizer, schedule and batches, started from the Keras model's weights, copied back into it for
export): 1.8 minutes instead of 9.5 on the 7600X, models scoring alike on held-out recordings. Without
a GPU it trains in Keras as before; `WAKESMITH_OWW_TRAINER=tf` keeps it there.
TensorFlow, which microWakeWord uses for features and export, is kept off the GPU (`WAKESMITH_TF_GPU=0` lets it try). More cores make feature
building and CPU training faster; microWakeWord's own trainer is mostly
one busy thread, so wakesmith gives TensorFlow at most 8 (`WAKESMITH_TF_THREADS` changes it). Close
other big programs (a local LLM, say) while training: worker counts follow free RAM, and below
about 16 GB microWakeWord's training runs out of memory (doctor and the page warn).
If the PC resets or the GPU driver crashes during the speech step, `WAKESMITH_TTS_BATCH=32` (or lower)
makes smaller batches that draw less GPU power. On a 13th/14th-gen Intel Core i7/i9, update the BIOS
first: older microcode can hard-reset these CPUs under load (one did, until its 2023 BIOS was updated).
Under WSL2, give the Linux VM at least 32 GB in `%UserProfile%\.wslconfig` (`[wsl2]` then `memory=48GB`):
training holds ~17 GB and reads ~20 GB of features, which run best from memory.

**Intel GPU note:** use Intel's compute runtime 26.35 or newer (with IGC 2.41 or newer). Version
26.05, which some distributions still ship, runs a few of PyTorch's GPU operations wrong (`nonzero`,
`unique` and everything built on them); the bad results cut synthetic clips short and can fault the
GPU. `wakesmith doctor` tests for it and names the fix. A crashed GPU job can leave the `xe` driver
stuck, after which anything that opens the GPU hangs until a reboot: wakesmith runs previews on the
CPU and times out GPU detection instead of hanging, and if `wakesmith doctor` says the GPU driver
isn't responding, reboot. A job that outgrows the card's memory spills into system RAM, so keep
`WAKESMITH_TTS_BATCH` at or below the default.

## Status

wakesmith trains the microWakeWord and openWakeWord models running on the Echo-style devices of one
household. It runs on Linux (Ubuntu 26.04 with two Intel Arc cards; a CPU-only Ubuntu 24.04 virtual
machine) and on Windows 11 through WSL2 (with an NVIDIA card). macOS is untested: setup installs the
CPU build of PyTorch there (no Apple GPU), and reports from Mac users are welcome. English is the
language; German is experimental ([Other languages](#other-languages-experimental)).
Fixes and improvements are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md).

## Licensing

wakesmith is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE): you may use it,
change it and share it, changed or not, for any noncommercial purpose (personal, hobby, research,
nonprofit, education). Commercial use isn't allowed. If you share it, pass the license and its
copyright notice along. Version 0.2.0 and earlier were released under Apache-2.0 and keep that
license. Code derived from other projects keeps its own license (MIT: the TTS code from
[piper-sample-generator](https://github.com/rhasspy/piper-sample-generator) and a port of
zserge/microwakeword's frontend; Apache-2.0: the PyTorch ports of microWakeWord's and openWakeWord's
recipes); see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Phonemes come from the `espeak-ng`
program, run as a separate process. The page bundles the Montserrat, Inter
and JetBrains Mono fonts (SIL Open Font License 1.1).

Training data is downloaded at setup, never bundled; `wakesmith doctor` and the Data & licenses page
list each item and its license. The default **personal** data profile uses datasets with
**non-commercial** terms (microWakeWord's negative sets: CC BY-NC 4.0; openWakeWord's ACAV100M
features: CC BY-NC-SA 4.0), so models trained with it are for personal use. Common Voice is CC0; the
aligner that cuts phrases out of it (MMS, through torchaudio) has CC BY-NC 4.0 weights. Every export
includes a LICENSE-NOTE.txt listing what that model was trained on.

## Credits

[microWakeWord](https://github.com/kahrendt/microWakeWord) (Kevin Ahrendt),
[openWakeWord](https://github.com/dscripka/openWakeWord) (David Scripka),
[Piper](https://github.com/rhasspy/piper) and piper-sample-generator (Michael Hansen),
[Mozilla Common Voice](https://commonvoice.mozilla.org/) and its volunteers,
torchaudio's MMS forced aligner (Meta AI), [CMUdict](https://github.com/cmusphinx/cmudict), MUSAN,
and the MIT room impulse response survey.
