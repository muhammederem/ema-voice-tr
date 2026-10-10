import hashlib

import numpy as np
import pytest
import soundfile as sf
import torch

from ema_voice.api import EMAVoice
from ema_voice.voices import add_voice


def digest(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_add_voice_instant_no_training(voice_ckpt, recordings, tmp_path):
    before = digest(voice_ckpt)
    path = add_voice(voice_ckpt, recordings, "ben", tmp_path / "voices", mode="instant", device="cpu")
    saved = torch.load(path, weights_only=False)
    assert digest(voice_ckpt) == before
    assert saved["vector"].shape == (64,) and saved["lora"] is None and saved["mode"] == "instant"
    tts = EMAVoice.load(voice_ckpt, device="cpu", voices_dir=tmp_path / "voices")
    assert "ben" in tts.voices()
    assert tts.say("Merhaba.", voice="ben", steps=1).duration > 0.3


def test_add_voice_refine_loss_decreases(voice_ckpt, recordings, tmp_path):
    path = add_voice(voice_ckpt, recordings, "ben", tmp_path, mode="refine", device="cpu", steps=40)
    saved = torch.load(path, weights_only=False)
    before, after = saved["eval_loss"]
    assert after < before


def test_add_voice_deep_lora_isolated(voice_ckpt, recordings, tmp_path):
    add_voice(voice_ckpt, recordings, "ben", tmp_path, mode="deep", device="cpu", steps=20)
    tts = EMAVoice.load(voice_ckpt, device="cpu", voices_dir=tmp_path)
    weights = {k: v.clone() for k, v in tts.model.state_dict().items()}
    assert tts.say("Bugün hava güzel.", voice="ben", seed=3, steps=2).duration > 0.3
    # the adapters were merged only for that call (audio itself varies a little: DACVAE adds a watermark)
    assert all(torch.equal(weights[k], v) for k, v in tts.model.state_dict().items())
    lora = torch.load(tmp_path / "ben.pt", weights_only=False)["lora"]
    assert 150_000 < sum(t.numel() for ab in lora.values() for t in ab.values() if torch.is_tensor(t)) < 250_000


def test_add_voice_from_single_wav_with_txt(voice_ckpt, recordings, tmp_path):
    path = add_voice(voice_ckpt, recordings / "01.wav", "tek", tmp_path, mode="instant", device="cpu",
                     min_seconds=1.0)
    assert path.exists()


def test_add_voice_too_short_raises(voice_ckpt, ema_clips, tmp_path):
    wav, text = ema_clips[3]
    sf.write(tmp_path / "kisa.wav", wav[:48000 * 3], 48000)
    (tmp_path / "kisa.txt").write_text(text)
    with pytest.raises(ValueError, match="5 s"):
        add_voice(voice_ckpt, tmp_path / "kisa.wav", "kisa", tmp_path / "v", device="cpu")


def test_long_recording_with_txt_is_segmented_not_one_clip(tmp_path, ema_clips):
    from ema_voice.voices import MAX_CLIP_SECONDS, read_recordings

    long = np.concatenate([w for w, _ in ema_clips] * 3)
    sf.write(tmp_path / "uzun.wav", long, 48000)
    (tmp_path / "uzun.txt").write_text("çok uzun bir metin")
    ((wav, text),) = read_recordings(tmp_path / "uzun.wav")
    assert len(wav) > MAX_CLIP_SECONDS * 48000 and text is None  # Whisper will segment it
