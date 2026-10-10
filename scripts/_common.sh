# Shared by every numbered script. Not meant to be run on its own.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/config.env"
PY="$ROOT/.venv/bin/python"
export PYTHONUNBUFFERED=1

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "HATA / ERROR: $*" >&2; exit 1; }

# run CMD...: runs it, or only prints it with DRY=1
run() { if [ "${DRY:-0}" = "1" ]; then echo "DRY: $*"; else "$@"; fi; }

require_venv() { [ -x "$PY" ] || die "önce / first run: ./scripts/0_setup.sh"; }

# cuda when asked for and present, else cpu (a Mac without NVIDIA still runs the model)
run_device() {
  if [ "$DEVICE" = "cuda" ] && ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo cpu
  else echo "$DEVICE"; fi
}

model_ckpt() {
  [ -f "$MODEL" ] || die "model yok / missing: $MODEL"
  if head -c 40 "$MODEL" | grep -q "git-lfs"; then
    die "model indirilmemiş (Git LFS) / not downloaded: git lfs install && git lfs pull"
  fi
  echo "$MODEL"
}
