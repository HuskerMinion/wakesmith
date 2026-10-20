#!/usr/bin/env bash
# Train a microWakeWord model (ESPHome / Home Assistant Voice devices) from the clips train.sh made.
# CPU only: Intel's TensorFlow GPU plugin stops at TF 2.15/2.16 and microWakeWord needs TF >= 2.18.
# ~18 steps/s on 12 cores, so the default 20k steps take ~20 min. Feature building uses every core.
# Model args are the upstream notebook's mixednet (10 ms features, stride 3).
# Output: WORK/mww/trained/<model>/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite
# Usage: mww-train.sh [--no-features]   (MWW_CONFIG=my.yaml: your own training config, @MODEL@ is filled in)
set -euo pipefail
source "$(dirname "$0")/common.sh"
model=$(cfg model_name)
mww="$WORK/mww"
mkdir -p "$mww"
[ -d "$mww/negative_datasets/speech" ] || scripts/mww-download-negatives.sh
if [ "${1:-}" != --no-features ]; then
  echo "== features"
  .venv-mww/bin/python scripts/mww_features.py
fi
config="$mww/$model.yaml"
sed "s/@MODEL@/$model/" "${MWW_CONFIG:-configs/mww.template.yaml}" > "$config"
# no real negatives -> no real_negative feature set to sample from
[ -d "$mww/features/real_negative" ] || .venv-mww/bin/python -c "import sys, yaml; c = yaml.safe_load(open(sys.argv[1]))
c['features'] = [f for f in c['features'] if f['features_dir'] != 'features/real_negative']
yaml.safe_dump(c, open(sys.argv[1], 'w'), sort_keys=False)" "$config"
cd "$mww"
# --restore_checkpoint resumes an interrupted run; a finished one would just be reloaded, so move it
# aside unless RESUME=1.
if [ -d "trained/$model" ] && [ "${RESUME:-0}" != 1 ]; then
  mv "trained/$model" "trained/$model.$(date +%Y%m%d-%H%M%S)"
fi
echo "== train ($config)"
"$ROOT/.venv-mww/bin/python" -m microwakeword.model_train_eval \
  --training_config="$model.yaml" \
  --train 1 --restore_checkpoint 1 \
  --test_tf_nonstreaming 0 --test_tflite_nonstreaming 0 --test_tflite_nonstreaming_quantized 0 \
  --test_tflite_streaming 0 --test_tflite_streaming_quantized 1 \
  --use_weights "best_weights" \
  mixednet \
  --pointwise_filters "64,64,64,64" \
  --repeat_in_block "1, 1, 1, 1" \
  --mixconv_kernel_sizes '[5], [7,11], [9,15], [23]' \
  --residual_connection "0,0,0,0" \
  --first_conv_filters 32 \
  --first_conv_kernel_size 5 \
  --stride 3
