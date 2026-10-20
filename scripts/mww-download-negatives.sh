#!/usr/bin/env bash
# Download microWakeWord's precomputed negative feature sets (huggingface.co/datasets/kahrendt/microwakeword)
# into WORK/mww/negative_datasets. ~5.7 GB of zips, ~25 GB unpacked. They are NOT part of this repo:
# check that dataset's card for the licenses of the audio they were built from (some sources are
# non-commercial) before using a model trained on them.
set -euo pipefail
source "$(dirname "$0")/common.sh"
dst="$WORK/mww/negative_datasets"
mkdir -p "$dst"
for set in speech dinner_party no_speech dinner_party_eval; do
  [ -d "$dst/$set" ] && { echo "$set: already there"; continue; }
  echo "$set: downloading"
  curl -fL --progress-bar -o "$dst/$set.zip" "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/$set.zip"
  python3 -I -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$dst/$set.zip" "$dst"
  rm "$dst/$set.zip"
done
du -sh "$dst"/*
