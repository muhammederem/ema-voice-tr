"""Add a voice to the model from someone's recordings, without changing the model file.

instant  the reference encoder's vector, averaged over 10 s windows; no training, no transcript needed.
refine   that vector, then tuned on the recordings themselves (needs what was said: a .txt or Whisper).
deep     the vector plus small adapters on the DiT, tuned together; best with a few minutes of speech.

The result is `<out_dir>/<name>.pt`, which `EMAVoice.load(..., voices_dir=out_dir)` picks up by name.
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from ema_lightning.frontend import Frontend

from ema_voice import lora
from ema_voice.align import Aligner
from ema_voice.batch import collate
from ema_voice.codec import Codec, normalize
from ema_voice.model import load
from ema_voice.timeline import encode_text, prepare_text

RATE = 48000
FPS = 25
MAX_CLIP_SECONDS = 20.0  # longer recordings are cut at Whisper's segment boundaries
MIN_SECONDS = 5.0
WINDOW = 10 * FPS
STEPS = {"refine": 500, "deep": 3000}
LR_VECTOR, LR_LORA = 5e-3, 1e-3
BATCH_FRAMES = 4000
WHISPER = "openai/whisper-large-v3-turbo"


def load_audio(path):
    """Mono 48 kHz at about -20 dBFS, peaks under -1 dBFS (how the training audio was levelled)."""
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(1)
    if sr != RATE:
        wav = AF.resample(torch.from_numpy(wav), sr, RATE).numpy()
    rms = np.sqrt(np.mean(wav**2)) + 1e-9
    wav = wav * (10 ** (-20 / 20) / rms)
    peak = np.abs(wav).max() + 1e-9
    return (wav * min(1.0, 10 ** (-1 / 20) / peak)).astype(np.float32)


def read_recordings(source):
    """[(wav, transcript or None)] from one audio file or a folder of them, with same-name .txt files."""
    source = Path(source)
    files = [source] if source.is_file() else sorted(p for p in source.iterdir()
                                                     if p.suffix.lower() in (".wav", ".flac", ".mp3", ".m4a", ".ogg"))
    out = []
    for f in files:
        txt = f.with_suffix(".txt")
        wav = load_audio(f)
        text = txt.read_text().strip() if txt.exists() else None
        out.append((wav, text if len(wav) <= MAX_CLIP_SECONDS * RATE else None))
    return out


def transcribe(clips, device):
    """Fill missing transcripts with Whisper, cutting long recordings at its segment boundaries (3-15 s)."""
    if all(t for _, t in clips):
        return clips
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model=WHISPER, device=device,
                   torch_dtype=torch.float16 if str(device).startswith("cuda") else torch.float32)
    out = []
    for wav, text in clips:
        if text:
            out.append((wav, text))
            continue
        audio16 = AF.resample(torch.from_numpy(wav), RATE, 16000).numpy()
        res = asr({"raw": audio16, "sampling_rate": 16000}, return_timestamps=True,
                  generate_kwargs={"language": "turkish", "task": "transcribe"})
        seg_start, seg_text = None, []
        for c in res["chunks"]:
            start, end = c["timestamp"]
            end = end if end is not None else len(wav) / RATE
            seg_start = start if seg_start is None else seg_start
            seg_text.append(c["text"].strip())
            if end - seg_start >= 3.0 or c is res["chunks"][-1] or end - seg_start >= MAX_CLIP_SECONDS - 5:
                out.append((wav[int(seg_start * RATE):int(end * RATE)], " ".join(seg_text)))
                seg_start, seg_text = None, []
    return out



def item(z, text, letter_frames, word_frames, stoi):
    ids, cw, wstart = encode_text(text, stoi)
    return {"ids": ids, "cw": cw, "wstart": wstart, "letter_frames": torch.as_tensor(letter_frames).float(),
            "word_frames": torch.as_tensor(word_frames).long(), "z1": z, "voice_id": 0, "ref": z[:WINDOW]}


@torch.no_grad()
def instant_vector(model, latents):
    """Mean of the reference encoder over 10 s windows of all the speech (shorter tail windows over 2 s count)."""
    z = torch.cat(latents)
    windows = [z[i:i + WINDOW] for i in range(0, len(z), WINDOW) if len(z[i:i + WINDOW]) >= 2 * FPS] or [z]
    vecs = [model.ref_encoder(w[None], torch.ones(1, len(w), dtype=torch.bool, device=w.device)) for w in windows]
    return torch.cat(vecs).mean(0)


def tune(model, items, v0, mode, steps, device):
    """Optimise the voice vector (and adapters for deep); returns (vector, adapters or None, eval losses)."""
    v = torch.nn.Parameter(v0.clone())
    lengths = [it["z1"].shape[0] for it in items]

    def batch(g):
        order = torch.randperm(len(items), generator=g).tolist()
        chosen, total = [], 0
        for i in order:
            if chosen and total + lengths[i] > BATCH_FRAMES:
                break
            chosen.append(i)
            total += lengths[i]
        return collate([items[i] for i in chosen]).to(device)

    def evaluate():
        """Loss on a fixed batch with fixed noise, so before and after compare like for like."""
        torch.manual_seed(0)
        b = batch(torch.Generator().manual_seed(0))
        with torch.no_grad():
            return float(model.loss_for(b, v.expand(b.z1.shape[0], -1), p_drop=0.0)["total"])

    def run(params):
        opt = torch.optim.Adam(params)
        g = torch.Generator().manual_seed(1)
        for _ in range(steps):
            b = batch(g)
            loss = model.loss_for(b, v.expand(b.z1.shape[0], -1), p_drop=0.1)["total"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.requires_grad_(False)
    before = evaluate()
    if mode == "deep":
        with lora.wrapped(model) as adapters:
            run([{"params": [v], "lr": LR_VECTOR},
                 {"params": [p for a in adapters.values() for p in (a.A, a.B)], "lr": LR_LORA}])
            after = evaluate()
            exported = lora.export(adapters)
        return v.detach(), exported, (before, after)
    run([{"params": [v], "lr": LR_VECTOR}])
    return v.detach(), None, (before, evaluate())


def add_voice(ckpt, source, name, out_dir, mode="instant", device="cpu", steps=None,
              min_seconds=MIN_SECONDS):
    if mode not in ("instant", "refine", "deep"):
        raise ValueError(f"mode must be instant, refine or deep, not {mode!r}")
    device = torch.device(device)
    model, meta = load(ckpt, device)
    stats = meta["stats"]
    clips = read_recordings(source)
    seconds = sum(len(w) for w, _ in clips) / RATE
    if seconds < min_seconds:
        raise ValueError(f"en az {min_seconds:g} s kayıt gerekli; bu kayıtlar {seconds:.1f} s")
    codec = Codec(device)
    if mode != "instant":
        clips = transcribe(clips, device)
    zs = codec.encode_list([w for w, _ in clips])
    latents = [normalize(torch.from_numpy(z.astype(np.float32)), stats).to(device) for z in zs]
    texts = None
    if mode != "instant":
        aligner, texts, kept = Aligner(device), [], []
        frontend = Frontend(model.vocab)
        for (wav, text), z in zip(clips, latents, strict=True):
            text = prepare_text(frontend(text), model.vocab)  # digits and abbreviations read out first
            a = aligner(wav, text) if text else None
            if a is not None and a.total_frames == z.shape[0]:
                texts.append((text, a.letter_frames, a.word_frames))
                kept.append(z)
        if not kept:
            raise ValueError("kayıtlar metinleriyle hizalanamadı; .txt dosyalarını kontrol et")
        latents = kept

    v = instant_vector(model, latents)
    adapters, eval_loss, seconds = None, None, sum(len(z) for z in latents) / FPS
    if mode != "instant":
        items = [item(z, t, lf, wf, model.stoi) for z, (t, lf, wf) in zip(latents, texts, strict=True)]
        v, adapters, eval_loss = tune(model, items, v, mode, steps or STEPS[mode], device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.pt"
    torch.save({"vector": v.cpu(), "lora": adapters, "mode": mode, "seconds": seconds, "eval_loss": eval_loss},
               path)
    return path


def main():
    ap = argparse.ArgumentParser(description="Add a voice from recordings (a file or a folder)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="voices")
    level = ap.add_mutually_exclusive_group()
    level.add_argument("--refine", action="store_const", dest="mode", const="refine")
    level.add_argument("--deep", action="store_const", dest="mode", const="deep")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    path = add_voice(args.ckpt, args.source, args.name, args.out, args.mode or "instant", args.device)
    print(path)


if __name__ == "__main__":
    main()
