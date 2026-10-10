import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download

TEXTS = [
    "merhaba, size nasıl yardımcı olabilirim?",
    "siparişiniz yarın sabah kargoya verilecek.",
    "sabahın erken saatlerinde liman henüz uyanmamıştı.",
    "lütfen kart numaranızı tekrar söyler misiniz?",
]


@pytest.fixture(scope="session")
def ema_ckpt():
    return hf_hub_download("canberkkkkkk/ema-lightning", "ema.pt")


@pytest.fixture(scope="session")
def ema():
    from ema_lightning import EMA

    torch.manual_seed(0)
    return EMA(device="cpu")


@pytest.fixture(scope="session")
def ema_clips(ema):
    """Four 48 kHz clips made by EMA Lightning, with the exact text they say."""
    return [(np.asarray(ema.say(t, seed=i).audio, dtype=np.float32), t) for i, t in enumerate(TEXTS)]


def fake_item(text, stoi, voice_id=0, seed=0):
    """A training item with a plausible timeline and random latents."""
    from ema_voice.timeline import encode_text

    g = torch.Generator().manual_seed(seed)
    ids, cw, wstart = encode_text(text, stoi)
    word_frames = torch.randint(3, 9, (int(cw[-1]) + 1,), generator=g)
    letter_frames = torch.zeros(len(ids))
    for w, n in enumerate(word_frames):
        sel = cw == w
        letter_frames[sel] = n.float() / sel.sum()
    T = int(word_frames.sum())
    return dict(ids=ids, cw=cw, wstart=wstart, letter_frames=letter_frames, word_frames=word_frames,
                z1=torch.randn(T, 128, generator=g), voice_id=voice_id,
                ref=torch.randn(int(torch.randint(50, 120, (1,), generator=g)), 128, generator=g))




@pytest.fixture(scope="session")
def voice_ckpt(ema_ckpt, tmp_path_factory):
    """An untrained EMA Voice checkpoint with two built-in voices, in the released file format."""
    from ema_voice.model import from_ema, save

    torch.manual_seed(0)
    model = from_ema(ema_ckpt, n_voices=2)
    with torch.no_grad():
        model.voice_mod.weight.normal_(std=0.5)
        model.out_proj.weight.normal_(std=0.05)
    path = tmp_path_factory.mktemp("ckpt") / "model.pt"
    save(model, path, {"voices": ["v0", "v1"], "stats": {"mean": torch.zeros(128), "std": torch.ones(128)},
                       "cfg_weight": 2.0})
    return path


@pytest.fixture
def recordings(tmp_path, ema_clips):
    """A folder of someone's recordings with transcripts, as a user would bring them."""
    import soundfile as sf

    d = tmp_path / "rec"
    d.mkdir()
    for i, (wav, text) in enumerate(ema_clips):
        sf.write(d / f"{i:02d}.wav", wav, 48000)
        (d / f"{i:02d}.txt").write_text(text)
    return d
