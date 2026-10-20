# Sourced by the recipe's shell scripts: repo root, settings from recipe.env, helpers.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [ -f recipe.env ]; then
  # Environment variables already set win over recipe.env.
  while IFS='=' read -r key val; do
    [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || continue
    [ -n "${!key+x}" ] && continue
    val="${val%%#*}"; val="${val%"${val##*[![:space:]]}"}"; val="${val#\"}"; val="${val%\"}"
    export "$key=$val"
  done < recipe.env
else
  echo "no recipe.env - copy recipe.env.example to recipe.env and edit it" >&2; exit 1
fi
: "${WAKE_CONFIG:?set WAKE_CONFIG in recipe.env}" "${ACCEL:=cpu}" "${GPU:=0}" "${WORK:=work}"
: "${EXTRACT:=cpu}" "${OPENVINO_DEVICE:=GPU}" "${OWW_TFLITE:=1}" "${MWW:=1}"
export WAKE_CONFIG ACCEL GPU WORK EXTRACT OPENVINO_DEVICE OWW_TFLITE MWW

PY=.venv/bin/python
LKW=.venv/bin/livekit-wakeword

# Read a key from the livekit-wakeword YAML: cfg model_name
cfg() {
  $PY -c "import sys, yaml; d = {'data_dir': './data', 'output_dir': './output'}
print(yaml.safe_load(open(sys.argv[1])).get(sys.argv[2], d.get(sys.argv[2])))" "$WAKE_CONFIG" "$1"
}

# Environment that pins a process to one GPU: on_gpu 1 cmd args...
on_gpu() {
  local id="$1"; shift
  case "$ACCEL" in
    xpu) ZE_AFFINITY_MASK="$id" "$@" ;;
    cuda) CUDA_VISIBLE_DEVICES="$id" "$@" ;;
    *) "$@" ;;
  esac
}
