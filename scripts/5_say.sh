#!/usr/bin/env bash
# 5) Metinden ses / text to speech: ./scripts/5_say.sh "metin" <ses/voice> [çıktı/out.wav]    (./scripts/5_say.sh --list)
source "$(dirname "$0")/_common.sh"
require_venv
ckpt=$(model_ckpt)
if [ "${1:-}" = "--list" ] || [ $# -lt 2 ]; then
  run "$PY" -m ema_voice.api --ckpt "$ckpt" --voices-dir "$VOICES" --list
  echo 'Kullanım / usage: ./scripts/5_say.sh "metin / text" <ses / voice> [çıktı.wav / out.wav]'
  exit 0
fi
run "$PY" -m ema_voice.api --ckpt "$ckpt" --voices-dir "$VOICES" --text "$1" --voice "$2" --out "${3:-cikti.wav}"
