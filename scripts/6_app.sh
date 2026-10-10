#!/usr/bin/env bash
# 6) Arayüz / web interface: ./scripts/6_app.sh  → http://127.0.0.1:7860 (kapatmak / stop: Ctrl-C)
source "$(dirname "$0")/_common.sh"
require_venv
ckpt=$(model_ckpt)
run "$PY" -m ema_voice.app --ckpt "$ckpt" --voices-dir "$VOICES" --device "$(run_device)" --port "${PORT:-7860}"
