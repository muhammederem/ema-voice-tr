"""A local web page for EMA Voice: type text and hear it, or add your own voice from a recording.

    python -m ema_voice.app --ckpt models/ema-voice.pt --voices-dir voices
"""
import argparse
import re
import shutil
import tempfile
from pathlib import Path

import torch

from ema_voice.api import RATE, EMAVoice
from ema_voice.voices import add_voice

NAME = re.compile(r"^[\w.-]{1,40}$")
MODES = {"Anında / Instant (10 s)": "instant", "Refine (10 s, ~2 dk/min)": "refine",
         "Deep (3-5 dk/min, ~20-30 dk/min)": "deep"}


def voice_choices(tts):
    """Your added voices first, then the built-in ones."""
    added = [v for v in tts.voices() if v not in tts.table]
    return added + list(tts.table)


def speak(tts, text, voice, speed=1.0, seed=None):
    """(rate, audio) and a status line, for the Speak tab."""
    if not text or not text.strip():
        raise ValueError("Metin boş / Text is empty")
    if not voice:
        raise ValueError("Bir ses seç / Pick a voice")
    seed = None if seed is None or seed == "" else int(seed)
    speech = tts.say(text, voice, speed=float(speed), seed=seed)
    return (RATE, speech.audio), f"{speech.duration:.1f} s · seed {speech.seed}"


def enroll(ckpt, voices_dir, device, name, files, mic, mode):
    """Copies the recordings into one folder and adds them as voice `name`; returns the saved path."""
    name = (name or "").strip()
    if not NAME.match(name):
        raise ValueError("İsim: harf, rakam, '-', '_' (en çok 40) / Name: letters, digits, '-', '_' (max 40)")
    paths = [Path(f) for f in (files or [])] + ([Path(mic)] if mic else [])
    if not paths:
        raise ValueError("Kayıt yükle ya da mikrofonla kaydet / Upload or record audio")
    with tempfile.TemporaryDirectory() as tmp:
        for i, p in enumerate(paths):
            dest = Path(tmp) / p.name  # names kept, so 01.wav still pairs with its 01.txt transcript
            if dest.exists():
                dest = Path(tmp) / f"{p.stem}_{i}{p.suffix}"
            shutil.copy(p, dest)
        return add_voice(ckpt, tmp, name, voices_dir, mode, device)


def build(ckpt, voices_dir, device):
    import gradio as gr

    tts = EMAVoice.load(ckpt, device=device, voices_dir=voices_dir)

    def on_speak(text, voice, speed, seed):
        try:
            return speak(tts, text, voice, speed, seed)
        except ValueError as e:
            raise gr.Error(str(e)) from e

    def on_add(name, files, mic, mode):
        try:
            path = enroll(ckpt, voices_dir, str(tts.device), name, files, mic, MODES[mode])
        except ValueError as e:
            raise gr.Error(str(e)) from e
        return gr.Dropdown(choices=voice_choices(tts), value=path.stem), f"Eklendi / Added: {path.stem}"

    with gr.Blocks(title="EMA Voice") as page:
        gr.Markdown("# EMA Voice\nTürkçe metinden ses · Turkish text to speech")
        with gr.Tab("Konuş / Speak"):
            text = gr.Textbox(label="Metin / Text", lines=4, value="Merhaba, bu ses yapay zekâ ile üretilmiştir.")
            choices = voice_choices(tts)
            with gr.Row():
                voice = gr.Dropdown(choices, value=choices[0] if choices else None, label="Ses / Voice")
                speed = gr.Slider(0.7, 1.4, value=1.0, step=0.05, label="Hız / Speed")
                seed = gr.Number(value=None, precision=0, label="Seed (boş = rastgele / empty = random)")
            go = gr.Button("Oku / Speak", variant="primary")
            audio = gr.Audio(label="Çıktı / Output", type="numpy")
            info = gr.Markdown()
            go.click(on_speak, [text, voice, speed, seed], [audio, info])
        with gr.Tab("Ses ekle / Add voice"):
            gr.Markdown(
                "Yalnızca kendi sesini ya da izni olan birinin sesini ekle. "
                "Only add your own voice or one you have permission to use.\n\n"
                "Sessiz bir oda, mikrofon ağzına ~20 cm. `prompts/kayit_metinleri.txt` cümlelerini oku. "
                "Quiet room, mic ~20 cm away; read the sentences in `prompts/kayit_metinleri.txt`.")
            name = gr.Textbox(label="İsim / Name", placeholder="erem")
            with gr.Row():
                files = gr.File(label="Kayıtlar / Recordings (wav, flac, mp3; .txt optional)",
                                file_count="multiple", type="filepath")
                mic = gr.Audio(label="ya da kaydet / or record", sources=["microphone"], type="filepath")
            mode = gr.Radio(list(MODES), value=next(iter(MODES)), label="Seviye / Level")
            add = gr.Button("Ekle / Add", variant="primary")
            status = gr.Markdown()
            add.click(on_add, [name, files, mic, mode], [voice, status])
    return page


def main():
    ap = argparse.ArgumentParser(description="Local web page for EMA Voice")
    ap.add_argument("--ckpt", default="models/ema-voice.pt")
    ap.add_argument("--voices-dir", default="voices")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    Path(args.voices_dir).mkdir(parents=True, exist_ok=True)
    build(args.ckpt, args.voices_dir, args.device).queue().launch(server_name="127.0.0.1", server_port=args.port,
                                                                   inbrowser=True)


if __name__ == "__main__":
    main()
