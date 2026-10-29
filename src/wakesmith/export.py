"""Device-ready files: <project>/export/microwakeword/ and <project>/export/openwakeword/."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import ensemble, models, mww
from .project import Project

WEBSITE = "https://github.com/HuskerMinion/wakesmith"

# TFLite metadata entry naming the feature frontend the model was trained on, for runtimes that
# offer more than one (zserge/microwakeword-based devices). It lives in the model file because
# ESPHome refuses manifests with keys it doesn't know, and Home Assistant only passes the manifest
# and model on. TFLite Micro ignores metadata.
FRONTEND_METADATA = "mww_feature_frontend"

ESPHOME_SNIPPET = """\
# ESPHome: copy {name}.tflite and {name}.json somewhere your ESPHome device config can reach
# (e.g. next to the YAML, or a web server), then:
micro_wake_word:
  models:
    - model: {name}.json
"""


# microWakeWord's usual arena size for one model of this architecture; not measured per model.
TENSOR_ARENA = ensemble.ARENA_PER_MODEL


def use_combined(p: Project, combined: bool | None = None) -> dict | None:
    """The combined model's info when the microWakeWord export is to be the combined model (ensemble.py):
    combined True asks for it (an error when there is none), False for the kept candidate, None
    follows the project's microwakeword.combine setting when a combined model exists."""
    info = ensemble.info(p)
    if combined and not info:
        raise ValueError(f"{p.name} has no combined model: run `wakesmith combine {p.name}` first")
    if combined is None:
        combined = p.microwakeword.combine
    return info if combined else None


def export(p: Project, combined: bool | None = None) -> list[Path]:
    """combined: export the combined model instead of the kept candidate (see use_combined)."""
    files: list[Path] = []
    both = use_combined(p, combined) if p.microwakeword.enabled else None
    src = ensemble.model_path(p) if both else mww.model_path(p)
    if p.microwakeword.enabled and src.exists():
        d = p.path("export", "microwakeword")
        d.mkdir(parents=True, exist_ok=True)
        tag_frontend(src, d / f"{p.name}.tflite", "tflm")
        manifest = {
            "type": "micro",
            "wake_word": p.display,
            "author": "wakesmith",
            "website": WEBSITE,
            "model": f"{p.name}.tflite",
            "trained_languages": [p.language],
            "version": 2,
            "micro": {
                "probability_cutoff": p.microwakeword.cutoff,
                "feature_step_size": 10,
                "sliding_window_size": 5,
                # a combined model holds every candidate's state: the sum of theirs
                "tensor_arena_size": both["tensor_arena_size"] if both else TENSOR_ARENA,
                "minimum_esphome_version": "2024.7.0",
            },
        }
        (d / f"{p.name}.json").write_text(json.dumps(manifest, indent=2))
        (d / "esphome.yaml").write_text(ESPHOME_SNIPPET.format(name=p.name))
        files += [d / f"{p.name}.tflite", d / f"{p.name}.json", d / "esphome.yaml"]
    oww_model = p.path("oww", "model", "model.tflite")
    if p.openwakeword.enabled and oww_model.exists():
        d = p.path("export", "openwakeword")
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(oww_model, d / f"{p.name}.tflite")
        manifest = {
            "wake_word": p.display,
            "model": f"{p.name}.tflite",
            "trained_languages": [p.language],
            "threshold": p.openwakeword.threshold,
        }
        (d / f"{p.name}.json").write_text(json.dumps(manifest, indent=2))
        files += [d / f"{p.name}.tflite", d / f"{p.name}.json"]
    if files:
        p.path("export", "LICENSE-NOTE.txt").write_text(license_note(p))
        files.append(p.path("export", "LICENSE-NOTE.txt"))
        models.record_export(p, combined=both)
    return files


def tag_frontend(src: Path, dst: Path, frontend: str) -> None:
    """Copy a .tflite model, recording `frontend` in its metadata (replacing an earlier tag)."""
    from tensorflow.lite.python import schema_py_generated as schema
    from tensorflow.lite.tools import flatbuffer_utils

    model = flatbuffer_utils.read_model(str(src))
    buf = schema.BufferT()
    buf.data = list(frontend.encode())
    model.buffers.append(buf)
    entry = schema.MetadataT()
    entry.name, entry.buffer = FRONTEND_METADATA, len(model.buffers) - 1
    keep = [
        m
        for m in model.metadata or []
        if m.name not in (FRONTEND_METADATA, FRONTEND_METADATA.encode())
    ]
    model.metadata = keep + [entry]
    flatbuffer_utils.write_model(model, str(dst))


def model_frontend(path: Path) -> str | None:
    """The frontend tag of a .tflite model, or None if it has none."""
    from tensorflow.lite.tools import flatbuffer_utils

    model = flatbuffer_utils.read_model(str(path))
    for m in model.metadata or []:
        if m.name in (FRONTEND_METADATA, FRONTEND_METADATA.encode()):
            return bytes(model.buffers[m.buffer].data).decode()
    return None


def license_note(p: Project) -> str:
    """What this project's models were trained on, with each source's license."""
    from . import assets

    a = assets.ASSETS
    line = lambda name, what: f"- {what}: {a[name].license}"
    out = [f"{p.display}: training data and licenses", ""]
    out.append("Used by every model here:")
    for m in p.tts_models:
        voice = Path(a[assets.TTS_MODELS[m]].path).name.split(".")[0].replace("en-us", "en_US")
        out.append(line(assets.TTS_MODELS[m], f"synthetic speech, Piper voice {voice}"))
    nc = any(a[assets.TTS_MODELS[m]].license.startswith("CC BY-NC") for m in p.tts_models)
    out.append(line("musan-noise", "background noise (MUSAN)"))
    out.append(line("mit-rirs", "room reverb (MIT impulse responses)"))
    if p.microwakeword.enabled:
        out += [
            "",
            "microWakeWord model:",
            line(
                "mww-speech",
                "negative examples (microWakeWord speech, dinner party, no-speech sets)",
            ),
        ]
        nc = True
    if p.openwakeword.enabled:
        out += [
            "",
            "openWakeWord model:",
            line("oww-acav-features", "negative examples (ACAV100M features)"),
            line("oww-validation-features", "false-trigger rate estimate"),
            line("oww-melspectrogram", "frontend"),
        ]
        nc = True
    real = []
    if p.real.positives or p.real.negatives:
        real.append("- your own recordings (not distributed with the model)")
    if p.uses_commonvoice:
        use = "training and testing" if p.commonvoice.train else "testing only"
        real.append(
            f"- Mozilla Common Voice 17.0 English, {use}: CC0; cut with the MMS aligner "
            "(torchaudio, weights CC BY-NC 4.0)"
        )
        nc = True
    if real:
        out += ["", "Real voices:"] + real
    if p.language == "en":
        cmu = (
            "Sound-alike phrases were chosen with the CMU Pronouncing Dictionary "
            f"({a['cmudict'].license}); it is not part of the model."
        )
        out += ["", cmu]
    out += [
        "",
        "Non-commercial: some of this data allows non-commercial use only (CC BY-NC / "
        "BY-NC-SA), so use these models for personal, non-commercial purposes."
        if nc
        else "Commercial use: none of the data above carries a non-commercial term.",
    ]
    return "\n".join(out) + "\n"
