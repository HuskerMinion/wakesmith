#!/usr/bin/env bash
# The whole recipe, as resumable stages:
#   generate  Piper TTS clips (positives, sound-alikes, background), one process per GEN_GPUS card
#   real      prepare the real recordings and inject them into the generated splits (skipped without any)
#   augment   noise / reverb / pitch rounds, on every CPU core
#   extract   openWakeWord frontend features (OpenVINO on an Intel GPU, or batched CPU)
#   train, export, eval   the main livekit-wakeword model (configs/*.yaml: model_type)
#   dnn       DNN-head twin on the same features -> openWakeWord .tflite (OWW_TFLITE=1)
#   mww       microWakeWord model from the same clips (MWW=1; scripts/mww-train.sh)
#   test      score the held-out real recordings with every model trained
# Usage: train.sh [first_stage [last_stage]]
#   train.sh                full run (generation is resumable: finished splits are skipped)
#   train.sh real           real recordings or their labels changed: reuse the synthetic clips
#   train.sh test test      only re-score
# Models from a previous run are copied to WORK/models/<timestamp>/ before training over them.
# TTS generation needs ~3.5 GB of RAM per process: don't run training next to it.
set -euo pipefail
source "$(dirname "$0")/common.sh"
STAGES=(generate real augment extract train export eval dnn mww test)
first="${1:-generate}" last="${2:-test}"
[[ " ${STAGES[*]} " == *" $first "* && " ${STAGES[*]} " == *" $last "* ]] || { echo "stages: ${STAGES[*]}" >&2; exit 1; }
model=$(cfg model_name) out="$(cfg output_dir)/$model"
dnn_cfg="$WORK/${model}_dnn.yaml" dnn_out="$(cfg output_dir)/${model}_dnn"
mww_tflite="$WORK/mww/trained/$model/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"
lkw_py() { $PY -c "from livekit.wakeword import load_config, $1; $1(load_config('$WAKE_CONFIG'))"; }
has_real() { [ -n "${REAL_POS_DIR:-}${REAL_NEG_DIR:-}" ]; }
# Background noise and RIRs missing = livekit-wakeword silently skips them: check first.
check_aug_data() {
  $PY -c "import sys; from pathlib import Path; from livekit.wakeword import load_config
a = load_config(sys.argv[1]).augmentation
for p in a.background_paths + a.rir_paths:
    if not any(Path(p).glob('**/*.wav')):
        sys.exit(f'no .wav files under {p} (augmentation.background_paths / rir_paths in {sys.argv[1]}): run ./setup.sh, or point them at data_dir')" "$WAKE_CONFIG"
}
mkdir -p "$WORK"
log="$WORK/train.log"

run_stage() {
  case "$1" in
  generate)
    check_aug_data
    read -ra gpus <<< "${GEN_GPUS:-$GPU}"
    if [ "$ACCEL" != cpu ] && [ ${#gpus[@]} -ge 2 ]; then
      # One process per card. Not a multi-GPU job: each process generates a different part.
      LKW_GEN_PARTS=positive on_gpu "${gpus[0]}" $LKW generate "$WAKE_CONFIG" & pos=$!
      LKW_GEN_PARTS=negative on_gpu "${gpus[1]}" $LKW generate "$WAKE_CONFIG" & neg=$!
      wait $pos; wait $neg
      LKW_GEN_PARTS=background $LKW generate "$WAKE_CONFIG"
    else
      on_gpu "${gpus[0]}" $LKW generate "$WAKE_CONFIG"
    fi ;;
  real)
    has_real || { echo "no REAL_POS_DIR / REAL_NEG_DIR: synthetic data only"; return; }
    $PY scripts/prep_real.py
    $PY scripts/inject_real.py ;;
  augment)
    check_aug_data
    lkw_py run_augment ;;
  extract)
    if [ "$EXTRACT" = openvino ]; then
      LKW_OV_DEVICE="$OPENVINO_DEVICE" on_gpu "$GPU" .venv-ov/bin/python -c \
        "from livekit.wakeword import load_config, run_extraction; run_extraction(load_config('$WAKE_CONFIG'))"
    else
      LKW_OV_DEVICE=off lkw_py run_extraction
    fi ;;
  train)
    if [ -f "$out/$model.onnx" ]; then
      bak="$WORK/models/$(date +%Y%m%d-%H%M%S)"; mkdir -p "$bak"
      cp "$out/$model".{onnx,pt} "$out"/*_eval.json "$bak/" 2>/dev/null || true
      cp "$dnn_out/${model}_dnn".{onnx,pt,tflite} "$dnn_out"/*_eval.json "$bak/" 2>/dev/null || true
      echo "previous models backed up to $bak"
    fi
    on_gpu "$GPU" $LKW train "$WAKE_CONFIG" ;;
  export)
    $LKW export "$WAKE_CONFIG" ;;
  eval)
    on_gpu "$GPU" $LKW eval "$WAKE_CONFIG" ;;
  dnn)
    [ "$OWW_TFLITE" = 1 ] || return 0
    # Same config with a DNN head: the only livekit-wakeword head that converts to .tflite.
    $PY -c "import sys, yaml; c = yaml.safe_load(open(sys.argv[1])); c['model_name'] += '_dnn'
c.setdefault('model', {})['model_type'] = 'dnn'; yaml.safe_dump(c, open(sys.argv[2], 'w'), sort_keys=False)" \
      "$WAKE_CONFIG" "$dnn_cfg"
    mkdir -p "$dnn_out"
    for f in "$out"/*_features_*.npy; do ln -sfn "$(realpath "$f")" "$dnn_out/"; done
    on_gpu "$GPU" $LKW train "$dnn_cfg"
    .venv-tflite/bin/livekit-wakeword export "$dnn_cfg" --format tflite
    on_gpu "$GPU" $LKW eval "$dnn_cfg" ;;
  mww)
    [ "$MWW" = 1 ] || return 0
    scripts/mww-train.sh ;;
  test)
    has_real || { echo "no real recordings: nothing held out to test"; return; }
    [ -d "$WORK/real/test" ] || $PY scripts/prep_real.py
    for m in "$out/$model.onnx" "$dnn_out/${model}_dnn.onnx"; do
      [ -f "$m" ] && $PY scripts/test_real.py "$m"
    done
    [ -f "$mww_tflite" ] && .venv-mww/bin/python scripts/test_real.py "$mww_tflite"
    true ;;
  esac
}

{
  on=0
  for s in "${STAGES[@]}"; do
    [ "$s" = "$first" ] && on=1
    if [ $on = 1 ]; then
      echo "=== stage: $s  ($(date +%T)) ==="
      run_stage "$s"
    fi
    [ "$s" = "$last" ] && break
  done
  echo "=== done ($(date +%T)) ==="
} 2>&1 | tee -a "$log"
