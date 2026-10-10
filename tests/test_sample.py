import pytest
import torch

from ema_voice.model import from_ema
from ema_voice.sample import synthesize_latents

TEXTS = ["merhaba, size nasıl yardımcı olabilirim?", "evet."]


@pytest.fixture(scope="module")
def model(ema_ckpt):
    torch.manual_seed(0)
    m = from_ema(ema_ckpt, n_voices=3).eval()
    with torch.no_grad():  # make the voice and the new layers matter, as after training
        m.voice_mod.weight.normal_(std=0.5)
        m.out_proj.weight.normal_(std=0.05)
    return m


def voices(model, ids):
    return model.voice(torch.tensor(ids))


def test_deterministic_with_seed(model):
    a = synthesize_latents(model, TEXTS, voices(model, [0, 1]), seed=7, steps=4)
    b = synthesize_latents(model, TEXTS, voices(model, [0, 1]), seed=7, steps=4)
    c = synthesize_latents(model, TEXTS, voices(model, [0, 1]), seed=8, steps=4)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not torch.allclose(a[0], c[0])


def test_speed_scales_length(model):
    (slow,) = synthesize_latents(model, TEXTS[:1], voices(model, [0]), speed=1.0, steps=2)
    (fast,) = synthesize_latents(model, TEXTS[:1], voices(model, [0]), speed=2.0, steps=2)
    assert fast.shape[0] == pytest.approx(slow.shape[0] / 2, rel=0.15)


def test_cfg_changes_output(model):
    (a,) = synthesize_latents(model, TEXTS[:1], voices(model, [0]), cfg=1.0, steps=2)
    (b,) = synthesize_latents(model, TEXTS[:1], voices(model, [0]), cfg=3.0, steps=2)
    assert a.shape == b.shape and not torch.allclose(a, b)


def test_batch_equals_single(model):
    both = synthesize_latents(model, TEXTS, voices(model, [0, 2]), seed=3, steps=4)
    one = synthesize_latents(model, TEXTS[1:], voices(model, [2]), seed=3, first_index=1, steps=4)
    assert both[1].shape == one[0].shape
    assert torch.allclose(both[1], one[0], atol=1e-4)
