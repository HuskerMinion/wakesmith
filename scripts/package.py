"""Copy finished models into WORK/deploy/ with the manifests Home Assistant / ESPHome expect.
  mww CUTOFF  microWakeWord: <model>.tflite (streaming, int8 in / uint8 out) + <model>.json
              (type micro, feature_step_size 10, sliding_window_size 5) -> WORK/deploy/<model>-mww/
  oww         openWakeWord: the DNN head's <model>.tflite ([1,16,96] float32 -> [1,1]) + .json
              -> WORK/deploy/<model>-oww/
Pick CUTOFF from test_real.py's table on your held-out clips (for comparison, one household's best
was 0.97 with ~0.85 also usable). tensor_arena_size is microWakeWord's usual 30000, not measured for
your model: if the device logs an arena error, raise it.
Usage: package.py mww 0.97 | package.py oww"""
import json
import shutil
import sys

import recipe

kind = sys.argv[1]
author = recipe.S.get("AUTHOR", "wakesmith")
if kind == "mww":
    cutoff = float(sys.argv[2])
    src = recipe.WORK / f"mww/trained/{recipe.MODEL}/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"
    dst = recipe.WORK / f"deploy/{recipe.MODEL}-mww"
    manifest = {
        "type": "micro",
        "wake_word": recipe.LABEL,
        "author": author,
        "website": "https://github.com/kahrendt/microWakeWord",
        "model": f"{recipe.MODEL}.tflite",
        "trained_languages": ["en"],
        "version": 2,
        "micro": {
            "probability_cutoff": cutoff,
            "feature_step_size": 10,
            "sliding_window_size": 5,
            "tensor_arena_size": 30000,
            "minimum_esphome_version": "2024.7.0",
        },
    }
elif kind == "oww":
    src = recipe.OUTPUT.parent / f"{recipe.MODEL}_dnn/{recipe.MODEL}_dnn.tflite"
    dst = recipe.WORK / f"deploy/{recipe.MODEL}-oww"
    manifest = {"wake_word": recipe.LABEL, "model": f"{recipe.MODEL}.tflite", "trained_languages": ["en"]}
else:
    sys.exit(__doc__)
dst.mkdir(parents=True, exist_ok=True)
shutil.copy(src, dst / f"{recipe.MODEL}.tflite")
json.dump(manifest, open(dst / f"{recipe.MODEL}.json", "w"), indent=2)
print(f"{dst}: {recipe.MODEL}.tflite ({(dst / f'{recipe.MODEL}.tflite').stat().st_size // 1024} KB) + {recipe.MODEL}.json")
