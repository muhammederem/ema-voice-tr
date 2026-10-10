#!/usr/bin/env bash
# 0) Kurulum / setup: Python ortamı, bağımlılıklar, modeller. ~10 dk.  Python env, dependencies, models. ~10 min.
source "$(dirname "$0")/_common.sh"
log "EMA Voice kurulumu başlıyor / setting up"

case "$(uname -s)" in
  Linux|Darwin) ;;
  *) die "desteklenmeyen işletim sistemi / unsupported OS: $(uname -s)" ;;
esac

TORCH_INDEX=""
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/GPU: /'
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  major=${driver%%.*}
  if [ "$major" -lt 525 ]; then
    die "NVIDIA sürücüsü çok eski / driver too old ($driver): sudo ubuntu-drivers install && sudo reboot"
  elif [ "$major" -lt 580 ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu126"  # the default build needs driver 580+ (CUDA 13)
  fi
else
  log "NVIDIA GPU yok, işlemci kullanılacak / no NVIDIA GPU, using the CPU"
fi

free_gb=$(df -Pk "$ROOT" | awk 'NR==2 {print int($4/1024/1024)}')
[ "$free_gb" -ge 10 ] || die "en az 10 GB boş disk gerekli / needs 10 GB free disk (${free_gb} GB)"

model_ckpt >/dev/null

if ! command -v uv >/dev/null; then
  log "uv kuruluyor / installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

log "Python ortamı / environment (.venv)"
[ -d .venv ] || uv venv -p 3.12 .venv
if [ -n "$TORCH_INDEX" ]; then
  uv pip install -p .venv "torch==2.14.1" "torchaudio==2.11.0" --index-url "$TORCH_INDEX"
fi
uv pip install -p .venv -e .
# macOS can mark .pth files hidden, and Python then skips them: the package would not import
[ "$(uname -s)" = "Darwin" ] && chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null || true

log "Modeller indiriliyor / downloading models (DACVAE, MMS aligner, Whisper)"
"$PY" - <<'PY'
import torchaudio
from huggingface_hub import hf_hub_download, snapshot_download
hf_hub_download("facebook/dacvae-watermarked", "weights.pth")
torchaudio.pipelines.MMS_FA.get_model(with_star=False)
snapshot_download("openai/whisper-large-v3-turbo", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.tiktoken"])
PY

"$PY" -c "import ema_voice.app" 2>/dev/null || die "paket yüklenemedi / package does not import"
log "Bitti / done.  Arayüz / interface:  ./scripts/6_app.sh    Komut satırı / CLI:  ./scripts/5_say.sh \"Merhaba.\" d1-00002"
