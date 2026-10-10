# EMA Voice

**English** · [Türkçe](README.tr.md)

Turkish text-to-speech with **2,736 built-in voices** and **voice cloning from a 10-second recording**. 48 kHz output,
runs on an NVIDIA GPU or a laptop CPU. Built on [EMA Lightning](https://huggingface.co/canberkkkkkk/ema-lightning).

## Results

Measured on held-out sentences (8 sampling steps, CFG 2.0).

| Metric | Value |
|---|---|
| Intelligibility, WER (Whisper large-v3 turbo, 200 sentences) | **6.9%**, vs 7.7% for the real recordings after the same codec |
| Speaker similarity, generated ↔ real (WavLM) | **0.944**, vs 0.955 for two real clips of the same speaker (**99%**) |
| Speed, RTX 3070 | 0.06 RTF, **16× real time** |
| Speed, Apple M5 CPU | **3.7× real time** |

**Cloning voices the model never heard** (16 voices, similarity to a real clip of the same speaker):

| Level | Recording | Time to add (RTX 3070) | Similarity |
|---|---|---|---|
| instant | 10 s | seconds | 0.915 |
| `--refine` | 10 s | ~2 min | 0.931 |
| `--deep` | 3-5 min | ~20-30 min | 0.933 (2 voices so far) |

WER is limited by the audio codec: the real recordings score 7.7% once they pass through it, and the model does
slightly better. The test voices are synthetic like the training voices; real microphone recordings may score lower.
Listen for yourself in [`samples/`](samples).

## Install

Linux or macOS. An NVIDIA GPU is optional (CPU works; `--deep` is slow without a GPU). About 10 GB of disk.

```bash
git lfs install                      # the model file (26 MB) is stored with Git LFS
git clone https://github.com/muhammederem/ema-voice-tr.git && cd ema-voice-tr
./scripts/0_setup.sh                 # Python env + model downloads, ~10 min
```

Without an NVIDIA GPU everything runs on the CPU automatically.

## Use

### Web interface

```bash
./scripts/6_app.sh                   # opens http://127.0.0.1:7860
```

**Speak** tab: type text, pick a voice, press the button. **Add voice** tab: upload or record yourself, name the
voice, pick a level. Everything runs on your machine.

### Command line

```bash
./scripts/5_say.sh "Merhaba, bu benim sesim." d1-00002 hello.wav   # built-in voice → hello.wav
./scripts/5_say.sh --list                                          # list voices

./scripts/4_add_voice.sh me.wav erem                 # instant, from 10 s
./scripts/4_add_voice.sh me.wav erem --refine        # same recording, ~2 min, closer
./scripts/4_add_voice.sh recordings/ erem --deep     # 3-5 min of speech, ~20-30 min, closest
./scripts/5_say.sh "Merhaba, bu benim sesim." erem
```

### Python

```python
from ema_voice.api import EMAVoice

tts = EMAVoice.load("models/ema-voice.pt", voices_dir="voices")
speech = tts.say("Siparişiniz yarın kargoya verilecek.", voice="erem", path="order.wav")
# optional: speed=1.1, seed=7 (same seed, same output), steps=8
```

### Recording tips

- A quiet room, the microphone (or phone) about 20 cm from your mouth.
- Read the sentences in [`prompts/kayit_metinleri.txt`](prompts/kayit_metinleri.txt) (60 sentences ≈ 5 min).
- One file per sentence (`01.wav`, `02.wav` ...) is best. You can put what you read next to it as `01.txt`;
  otherwise Whisper transcribes it for `--refine` and `--deep`.
- Added voices are saved as `voices/<name>.pt`. The model file is never changed.

10 seconds gives a voice that sounds like you, not an exact copy. More audio and `--deep` bring it closer.

## How it works

A 5.6M-parameter flow-matching model from EMA Lightning generates audio latents for the
[DACVAE](https://github.com/facebookresearch/dacvae) codec (48 kHz). Each voice is a 64-number vector. A reference
encoder computes that vector from a short recording; `--refine` tunes the vector on your recording, and `--deep`
also learns small LoRA adapters that are applied only when that voice speaks.

## License, credits and responsible use

- Code: Apache 2.0. Base model: [EMA Lightning](https://huggingface.co/canberkkkkkk/ema-lightning) (Apache 2.0).
  Text normalization: [normalizer-tr](https://github.com/erdemtuna/normalizer-tr).
- Voices: **Alania Turkish Synthetic Speech by PatientDesk AI**
  (https://huggingface.co/datasets/cloud0day3/alania-synthetic-speech-tr), CC BY 4.0.
- Codec: [DACVAE](https://github.com/facebookresearch/dacvae) (code Apache 2.0, weights SAM License). All generated
  audio carries an AudioSeal watermark.
- **Only add your own voice or a voice you have explicit permission to use.** Tell listeners the audio is
  AI-generated. Do not use it to impersonate real people, to deceive or defraud anyone, or to put words in someone's
  mouth.
