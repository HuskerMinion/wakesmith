#!/usr/bin/env bash
# One-time setup: Python venvs for your ACCEL, the recipe's patches, and the training data downloads.
# Needs: uv, git, curl, GNU patch, espeak-ng (Piper TTS phonemizer).
#   .venv       torch (XPU / CUDA / CPU build) + livekit-wakeword[train,eval,export]==0.3.0
#   .venv-ov    EXTRACT=openvino: CPU torch + onnxruntime-openvino in place of onnxruntime
#   .venv-tflite OWW_TFLITE=1: CPU torch + TensorFlow, for the openWakeWord .tflite export
#   .venv-mww   MWW=1: microWakeWord (pinned commit) + TensorFlow, CPU training
# Every venv but .venv uses --link-mode=copy so its files never share a hard link with uv's cache.
# Downloads land in $WORK: ~18 GB for livekit-wakeword, ~5.7 GB zipped (25 GB unpacked) for
# microWakeWord's negative sets. A full run with both model types peaks around 80 GB of disk.
# Usage: setup.sh [--no-data]
set -euo pipefail
source "$(dirname "$0")/scripts/common.sh"
for tool in uv git curl patch espeak-ng; do
  command -v "$tool" >/dev/null || { echo "missing: $tool" >&2; exit 1; }
done

LKW_VER=0.3.0
MWW_REPO=https://github.com/kahrendt/microWakeWord.git
MWW_COMMIT=4665173cd35f1cff9a61e06fc427f124766c488e
CPU_TORCH=https://download.pytorch.org/whl/cpu

case "$ACCEL" in
  xpu) TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/xpu} ;;
  cuda) TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128} ;;
  cpu) TORCH_INDEX=${TORCH_INDEX:-$CPU_TORCH} ;;
  *) echo "ACCEL must be xpu, cuda or cpu" >&2; exit 1 ;;
esac

vpip() { local v="$1"; shift; VIRTUAL_ENV="$ROOT/$v" uv pip install -q "$@"; }

if [ ! -x .venv/bin/livekit-wakeword ]; then
  echo "== .venv ($ACCEL torch from $TORCH_INDEX)"
  uv venv -q -p 3.12 .venv
  vpip .venv torch torchaudio --index-url "$TORCH_INDEX"
  vpip .venv "livekit-wakeword[train,eval,export]==$LKW_VER"
fi

if [ "$EXTRACT" = openvino ] && [ ! -x .venv-ov/bin/python ]; then
  echo "== .venv-ov (onnxruntime-openvino)"
  uv venv -q -p 3.12 .venv-ov
  vpip .venv-ov --link-mode=copy "livekit-wakeword[train]==$LKW_VER" torch torchaudio \
    --extra-index-url "$CPU_TORCH" --index-strategy unsafe-best-match
  VIRTUAL_ENV="$ROOT/.venv-ov" uv pip uninstall -q onnxruntime
  vpip .venv-ov --link-mode=copy --reinstall onnxruntime-openvino
fi

if [ "$OWW_TFLITE" = 1 ] && [ ! -x .venv-tflite/bin/python ]; then
  echo "== .venv-tflite (TensorFlow, for the openWakeWord .tflite export)"
  uv venv -q -p 3.12 .venv-tflite
  vpip .venv-tflite --link-mode=copy "livekit-wakeword[train,export,tflite]==$LKW_VER" torch \
    --extra-index-url "$CPU_TORCH" --index-strategy unsafe-best-match
fi

if [ "$MWW" = 1 ]; then
  src="$WORK/microWakeWord"
  if [ ! -d "$src/.git" ]; then
    echo "== microWakeWord @ ${MWW_COMMIT:0:7}"
    git clone -q "$MWW_REPO" "$src"
    git -C "$src" checkout -q "$MWW_COMMIT"
  fi
  if [ ! -x .venv-mww/bin/python ]; then
    echo "== .venv-mww (microWakeWord + TensorFlow, CPU)"
    uv venv -q -p 3.12 .venv-mww
    # tensorboard: the trainer imports it, upstream's requirements omit it
    vpip .venv-mww --link-mode=copy -e "$src" tensorboard
  fi
fi

echo "== patches"
./repatch.sh
.venv/bin/python -c "import torch; from livekit.wakeword.utils import get_device; print('   training device:', get_device(), '- torch', torch.__version__)"

[ "${1:-}" = --no-data ] && exit 0
echo "== livekit-wakeword data (Piper TTS model, ACAV100M negative features, MUSAN noise, MIT RIRs)"
$LKW setup --config "$WAKE_CONFIG"
[ "$MWW" = 1 ] && scripts/mww-download-negatives.sh
echo "setup done"
