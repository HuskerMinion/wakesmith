"""Build one streaming microWakeWord model that averages several trained candidates (run by ensemble.py
in its own process, so TensorFlow never loads into the server or the pipeline).

Each candidate's best weights are loaded into microWakeWord's mixednet with the flags wakesmith trains
with (mww.MODEL_ARGS), turned into its streaming internal-state form, and all of them are called on one
shared input; the output is the mean of their probabilities. Export and int8 quantization are
microWakeWord's own, as for a single model, so the file runs anywhere a single model runs (ops: the
usual ones plus ADD and MUL).

usage (cwd = the project's mww folder, so training.yaml's relative feature paths work):
  python -m wakesmith.ensemble_build OUT_DIR WEIGHTS.h5 WEIGHTS.h5 [...]
writes OUT_DIR/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite
"""

from __future__ import annotations

import argparse
import os
import sys

from .mww import MODEL_ARGS

TFLITE = os.path.join("tflite_stream_state_internal_quant", "stream_state_internal_quant.tflite")
CALIBRATION = 500  # spectrograms for the int8 calibration (microWakeWord's export: 500)


def _flags(training_config: str):
    from microwakeword import mixednet

    p = argparse.ArgumentParser()
    p.add_argument("--training_config")
    sub = p.add_subparsers(dest="model_name")
    mixednet.model_parameters(sub.add_parser("mixednet"))
    return p.parse_args([f"--training_config={training_config}", *MODEL_ARGS])


def _convert(config, handler, saved: str, dest: str, samples: int) -> None:
    """microWakeWord's utils.convert_saved_model_to_tflite(quantize=True), with the number of
    calibration spectrograms as a parameter (microWakeWord takes 500)."""
    import numpy as np
    import tensorflow as tf

    def representative():
        specs, _, _ = handler.get_data(
            "training", samples, features_length=config["spectrogram_length"]
        )
        specs[0][0, 0] = 0.0  # one pixel at the frontend's min and one at its max, as upstream
        specs[0][0, 1] = 26.0
        stride = config["stride"]
        for s in specs:
            for i in range(0, s.shape[0] - stride, stride):
                yield [s[i : i + stride, :].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(saved)
    converter.optimizations = {tf.lite.Optimize.DEFAULT}
    converter._experimental_variable_quantization = True  # streaming state in int8, as upstream
    converter.target_spec.supported_ops = {tf.lite.OpsSet.TFLITE_BUILTINS_INT8}
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.uint8
    converter.representative_dataset = tf.lite.RepresentativeDataset(representative)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    data = converter.convert()
    with open(dest, "wb") as f:
        f.write(data)


def build(
    out: str,
    weights: list[str],
    training_config: str = "training.yaml",
    samples: int = CALIBRATION,
    seed: int = 0,
) -> str:
    import random

    import numpy as np
    import tensorflow as tf
    from microwakeword import data as input_data
    from microwakeword import mixednet, utils
    from microwakeword.layers import modes
    from microwakeword.model_train_eval import load_config

    # the calibration spectrograms are drawn at random: a fixed seed makes a build repeatable
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    flags = _flags(training_config)
    config = load_config(flags, mixednet)
    config["train_dir"] = out
    mode = modes.Modes.STREAM_INTERNAL_STATE_INFERENCE
    streams = []
    for i, w in enumerate(weights):
        m = mixednet.model(flags, shape=config["training_input_shape"], batch_size=1)
        m.load_weights(w)
        s = utils.to_streaming_inference(m, config, mode)
        s._name = f"candidate_{i + 1}"
        streams.append(s)
    x = tf.keras.layers.Input(
        shape=modes.get_input_data_shape(config, mode),
        batch_size=1,
        dtype=tf.float32,
        name="input_audio",
    )
    outs = [s(x) for s in streams]
    y = tf.keras.layers.Average(name="mean_probability")(outs) if len(outs) > 1 else outs[0]
    combined = tf.keras.Model(x, y, name="combined")

    saved = os.path.join(out, "stream_state_internal")
    os.makedirs(saved, exist_ok=True)
    archive = tf.keras.export.ExportArchive()
    archive.track(combined)
    archive.add_endpoint(
        name="serve",
        fn=combined.call,
        input_signature=[tf.TensorSpec(shape=combined.input.shape, dtype=tf.float32)],
    )
    archive.write_out(saved)
    dest = os.path.join(out, TFLITE)
    _convert(config, input_data.FeatureHandler(config), saved, dest, samples)
    return dest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Average microWakeWord candidates into one model.")
    ap.add_argument("out")
    ap.add_argument("weights", nargs="+")
    ap.add_argument("--training_config", default="training.yaml")
    ap.add_argument("--samples", type=int, default=CALIBRATION, help="calibration spectrograms")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    print("wrote", build(a.out, a.weights, a.training_config, a.samples, a.seed), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
