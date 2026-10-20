#!/usr/bin/env bash
# Apply the recipe's patches to livekit-wakeword 0.3.0 in each venv (setup.sh runs this; run it again
# after any reinstall or upgrade).
#   01-xpu-device                 get_device() picks an Intel GPU (XPU); stock only knows CUDA/MPS/CPU
#   02-tts-duration-nan           clamp NaN/inf phoneme durations in Piper TTS
#   03-augment-all-cores          augmentation on every CPU core (LKW_AUG_WORKERS=1 for serial)
#   04-extraction-cli-crash       0.3.0 bug: `augment`/`run` CLI crash at feature extraction (fixed upstream after 0.3.0, #92)
#   05-extraction-batched-openvino  batched feature extraction; on an Intel GPU via OpenVINO when installed
#   06-generate-two-gpus-phoneme-cache  LKW_GEN_PARTS splits TTS generation across processes; espeak-ng phonemes cached
# Idempotent: already-applied patches are skipped. Exits non-zero if a patch no longer fits
# (upstream changed that code), which means the patch needs porting by hand.
# Patches are applied with GNU patch, which writes new files. uv installs packages as hard links into
# its cache, so editing a site-packages file in place would also change the cache and every other venv
# linked to it. Never hand-edit those files: change the .patch and re-run this.
# Usage: repatch.sh [venv ...]   (default: every .venv* next to this script that has livekit-wakeword)
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
venvs=("$@"); [ ${#venvs[@]} -eq 0 ] && venvs=("$here"/.venv*)
status=0
for venv in "${venvs[@]}"; do
  [ -x "$venv/bin/python" ] || continue
  site=$("$venv/bin/python" -c "import livekit.wakeword, pathlib; print(pathlib.Path(livekit.wakeword.__file__).parents[2])" 2>/dev/null) \
    || continue
  ver=$("$venv/bin/python" -c "import importlib.metadata as m; print(m.version('livekit-wakeword'))")
  echo "== $venv  (livekit-wakeword $ver)"
  for p in "$here"/patches/*.patch; do
    name=$(basename "$p" .patch)
    if patch -d "$site" -p1 -R --dry-run -s -f < "$p" >/dev/null 2>&1; then
      echo "   $name: already applied"
    elif patch -d "$site" -p1 --dry-run -s -f < "$p" >/dev/null 2>&1; then
      patch -d "$site" -p1 -s -f -b -z .orig < "$p" && echo "   $name: APPLIED"
    else
      echo "   $name: FAILED - doesn't fit the installed code. After updating this repo, an older"
      echo "      version of the patch may be installed: rebuild with  rm -rf .venv* && ./setup.sh --no-data"
      echo "      Otherwise livekit-wakeword itself changed, and $p needs porting by hand."; status=1
    fi
  done
done
exit $status
