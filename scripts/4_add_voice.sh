#!/usr/bin/env bash
# 4) Yeni ses ekle / add a voice: ./scripts/4_add_voice.sh <wav | klasör/folder> <isim/name> [--refine | --deep]
source "$(dirname "$0")/_common.sh"
if [ $# -lt 2 ]; then
  cat <<TXT
Kullanım / usage:  ./scripts/4_add_voice.sh <wav dosyası veya klasör | wav file or folder> <isim | name> [--refine | --deep]

  anında / instant (varsayılan / default)   5-10 s tek kayıt yeter / one 5-10 s recording; saniyeler / seconds
  --refine                                   aynı kayıt / same recording; ~2 dk/min; daha benzer / closer
  --deep                                     3-5 dk/min kayıt / of speech; ~20-30 dk/min (GPU); en benzer / closest

Sessiz bir oda, mikrofon ağzına ~20 cm. prompts/kayit_metinleri.txt cümlelerini oku.
Quiet room, microphone ~20 cm away. Read the sentences in prompts/kayit_metinleri.txt.
01.wav'ın yanına 01.txt (okunan cümle) koyabilirsin; yoksa Whisper yazıya döker.
You can put 01.txt (what you read) next to 01.wav; otherwise Whisper transcribes it.
TXT
  exit 1
fi
require_venv
ckpt=$(model_ckpt)
run "$PY" -m ema_voice.voices --ckpt "$ckpt" --source "$1" --name "$2" --out "$VOICES" \
  --device "$(run_device)" ${3:+"$3"}
log "Ses eklendi / voice added: $2.  ./scripts/5_say.sh \"Merhaba, bu benim sesim.\" $2"
