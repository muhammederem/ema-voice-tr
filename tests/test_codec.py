import numpy as np
import pytest
import torch

from ema_voice.codec import Codec, denormalize, normalize


@pytest.fixture(scope="module")
def codec():
    return Codec("cpu")


def test_encode_shapes(codec):
    z = codec.encode(torch.randn(2, int(48000 * 2.5)) * 0.1)
    assert z.shape == (2, 63, 128)


def test_encode_list_matches_single(codec, ema_clips):
    wavs = [w for w, _ in ema_clips[:2]]
    batch = codec.encode_list(wavs)
    for wav, z in zip(wavs, batch):
        single = codec.encode(torch.as_tensor(wav)[None])[0].numpy()
        assert z.dtype == np.float16 and z.shape == single.shape
        # only the last few frames see past the clip's end, where batch padding differs from DACVAE's own
        assert np.abs(z[:-4].astype(np.float32) - single[:-4]).max() < 0.05


def test_roundtrip_quality(codec, ema_clips):
    wav = torch.as_tensor(ema_clips[0][0])[None]
    out = codec.decode(codec.encode(wav))
    assert out.shape[-1] == int(np.ceil(wav.shape[-1] / 1920)) * 1920
    mel = lambda x: torch.log(torch.stft(x, 2048, 512, window=torch.hann_window(2048), return_complex=True).abs() + 1e-5)
    n = wav.shape[-1]
    assert (mel(out[:, :n]) - mel(wav)).abs().mean() < 1.0


def test_normalize_inverse():
    stats = {"mean": torch.randn(128), "std": torch.rand(128) + 0.5}
    z = torch.randn(3, 10, 128)
    assert torch.allclose(denormalize(normalize(z, stats), stats), z, atol=1e-5)


def test_encode_list_halves_batch_on_oom(codec, ema_clips, monkeypatch):
    real = codec.encode

    def picky(wav):
        if wav.shape[0] > 1:
            raise torch.OutOfMemoryError("fake")
        return real(wav)

    monkeypatch.setattr(codec, "encode", picky)
    out = codec.encode_list([w for w, _ in ema_clips])
    assert [z.shape[0] for z in out] == [int(np.ceil(len(w) / 1920)) for w, _ in ema_clips]
