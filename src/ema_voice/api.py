"""Text in, 48 kHz speech out, in any table voice or a voice added with `ema_voice.voices`."""
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from ema_lightning.api import Speech
from ema_lightning.chunker import chunk
from ema_lightning.frontend import Frontend

from ema_voice.codec import Codec, denormalize
from ema_voice.lora import applied
from ema_voice.model import load
from ema_voice.sample import synthesize_latents

RATE = 48000


class EMAVoice:
    def __init__(self, model, meta, codec, device, voices_dir=None):
        self.model, self.meta, self.codec, self.device = model, meta, codec, device
        self.voices_dir = Path(voices_dir) if voices_dir else None
        self.table = {v: i for i, v in enumerate(meta["voices"])}
        self.frontend = Frontend(model.vocab)

    @classmethod
    def load(cls, ckpt, device="auto", voices_dir=None):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model, meta = load(ckpt, device)
        return cls(model, meta, Codec(device), torch.device(device), voices_dir)

    def voices(self):
        added = sorted(p.stem for p in self.voices_dir.glob("*.pt")) if self.voices_dir else []
        return list(self.table) + [v for v in added if v not in self.table]

    def _voice(self, voice):
        """(vector [1, 64], LoRA weights or None) for a table voice or an added one."""
        if voice in self.table:
            return self.model.voice(torch.tensor([self.table[voice]], device=self.device)), None
        path = self.voices_dir / f"{voice}.pt" if self.voices_dir else None
        if path is None or not path.exists():
            raise ValueError(f"unknown voice {voice!r}; see voices()")
        saved = torch.load(path, map_location=self.device, weights_only=False)
        return saved["vector"][None].to(self.device), saved["lora"]

    @torch.no_grad()
    def say(self, text, voice, speed=1.0, seed=None, steps=8, cfg=None, path=None):
        v, lora = self._voice(voice)
        seed = random.randrange(2**31) if seed is None else int(seed)
        cfg = self.meta.get("cfg_weight", 2.0) if cfg is None else cfg
        pieces = chunk(self.frontend(text), speed)
        parts = []
        if pieces:
            with applied(self.model, lora):
                latents = synthesize_latents(self.model, [p for p, _ in pieces], v.expand(len(pieces), -1),
                                             speed=speed, steps=steps, cfg=cfg, seed=seed)
            for z, (_, pause) in zip(latents, pieces, strict=True):
                parts.append(self.codec.decode(denormalize(z, self.meta["stats"])[None])[0].numpy())
                parts.append(np.zeros(int(pause * RATE), np.float32))
        audio = np.clip(np.concatenate(parts), -1, 1) if parts else np.zeros(0, np.float32)
        if path is not None:
            sf.write(path, audio, RATE)
        return Speech(audio=audio, sample_rate=RATE, duration=len(audio) / RATE, seed=seed)


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Text to speech with EMA Voice")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--voices-dir", default=None)
    ap.add_argument("--voice")
    ap.add_argument("--text")
    ap.add_argument("--out", default="cikti.wav")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--list", action="store_true", help="list the voices and exit")
    args = ap.parse_args()
    tts = EMAVoice.load(args.ckpt, voices_dir=args.voices_dir)
    if args.list or not (args.voice and args.text):
        added = [v for v in tts.voices() if v not in tts.table]
        print(f"{len(tts.table)} hazır ses (ör. {', '.join(list(tts.table)[:5])} ...)")
        print("eklenen sesler:", ", ".join(added) if added else "yok")
        return
    speech = tts.say(args.text, args.voice, speed=args.speed, seed=args.seed, path=args.out)
    print(f"{args.out}: {speech.duration:.1f} s, ses {args.voice}, seed {speech.seed}")


if __name__ == "__main__":
    main()
