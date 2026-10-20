"""Device-ready files: <project>/export/microwakeword/ and <project>/export/openwakeword/."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import mww
from .project import Project

ESPHOME_SNIPPET = """\
# ESPHome: copy {name}.tflite and {name}.json somewhere your ESPHome device config can reach
# (e.g. next to the YAML, or a web server), then:
micro_wake_word:
  models:
    - model: {name}.json
"""


def export(p: Project) -> list[Path]:
    files: list[Path] = []
    if p.microwakeword.enabled and mww.model_path(p).exists():
        d = p.path("export", "microwakeword")
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(mww.model_path(p), d / f"{p.name}.tflite")
        manifest = {
            "type": "micro",
            "wake_word": p.display,
            "author": "wakesmith",
            "website": "https://github.com/HuskerMinion/wakesmith",
            "model": f"{p.name}.tflite",
            "trained_languages": ["en"],
            "version": 2,
            "micro": {
                "probability_cutoff": p.microwakeword.cutoff,
                "feature_step_size": 10,
                "sliding_window_size": 5,
                # microWakeWord's usual arena size for this architecture; not measured per model.
                "tensor_arena_size": 30000,
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
            "trained_languages": ["en"],
            "threshold": p.openwakeword.threshold,
        }
        (d / f"{p.name}.json").write_text(json.dumps(manifest, indent=2))
        files += [d / f"{p.name}.tflite", d / f"{p.name}.json"]
    if p.data_profile == "personal" and files:
        (p.path("export", "LICENSE-NOTE.txt")).write_text(
            "These models were trained with datasets that allow non-commercial use only\n"
            "(microWakeWord negative sets: CC BY-NC 4.0; openWakeWord ACAV100M features:\n"
            "CC BY-NC-SA 4.0). Use them for personal, non-commercial purposes.\n"
        )
        files.append(p.path("export", "LICENSE-NOTE.txt"))
    return files
