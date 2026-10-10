import torch
from conftest import fake_item
from ema_lightning.model import load_acoustic

from ema_voice.batch import collate
from ema_voice.model import VoiceAcoustic, from_ema, load, save

TEXTS = ["merhaba, dünya.", "bugün hava çok güzel.", "randevunuz yarın saat onda."]


def batch_for(model, texts=TEXTS, voices=(0, 1, 2)):
    return collate([fake_item(t, model.stoi, v, seed=i) for i, (t, v) in enumerate(zip(texts, voices))])


def test_from_ema_loads_shared_weights(ema_ckpt):
    model = from_ema(ema_ckpt, n_voices=5)
    ema = torch.load(ema_ckpt, map_location="cpu", weights_only=False)["ema"]
    own = model.state_dict()
    skipped = {"in_proj.weight", "in_proj.bias", "out_proj.weight", "out_proj.bias"}
    for k, v in ema.items():
        if k not in skipped:
            assert torch.equal(own[k], v), k
    assert own["in_proj.weight"].shape == (224, 128) and own["out_proj.weight"].shape == (128, 224)


def test_voice_zero_init_is_ema_text_stage(ema_ckpt):
    model = from_ema(ema_ckpt, n_voices=5).eval()
    ema = load_acoustic(ema_ckpt, "cpu")
    b = batch_for(model)
    _, want = ema.text_stage(b.ids, b.mask)
    for vid in (0, 3):
        v = model.voice(torch.full((3,), vid))
        _, dur = model.text_stage(b.ids, b.mask, v)
        assert torch.allclose(dur, want, atol=1e-5)


def test_voice_changes_output_after_update(ema_ckpt):
    torch.manual_seed(0)
    model = from_ema(ema_ckpt, n_voices=5)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    model.loss(batch_for(model))["total"].backward()
    opt.step()
    b = batch_for(model)
    h, _ = model.text_stage(b.ids, b.mask, model.voice(b.voice_ids))
    cond = model.conditions(h, b.dur_gt, b.mask, b.cw, b.wstart, b.fw, b.fp)
    t = torch.full((3,), 0.5)
    v0 = model.velocity(b.z1, cond, t, model.voice(torch.zeros(3, dtype=torch.long)), b.fmask)
    v1 = model.velocity(b.z1, cond, t, model.voice(torch.ones(3, dtype=torch.long)), b.fmask)
    assert not torch.allclose(v0, v1)


def test_loss_finite_and_backward(ema_ckpt):
    model = from_ema(ema_ckpt, n_voices=5)
    out = model.loss(batch_for(model))
    assert set(out) == {"total", "fm", "dur", "ref"}
    out["total"].backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name


def test_cond_dropout_zeroes_cond(ema_ckpt):
    model = from_ema(ema_ckpt, n_voices=5)
    a = collate([fake_item("merhaba dünya.", model.stoi, 0, seed=1)])
    b = collate([fake_item("güle güle sana.", model.stoi, 0, seed=1)])
    b.z1, b.fw, b.fp, b.fmask = a.z1, a.fw, a.fp, a.fmask
    torch.manual_seed(0)
    fa = model.loss(a, p_drop=1.0, p_ref=0.0)["fm"]
    torch.manual_seed(0)
    fb = model.loss(b, p_drop=1.0, p_ref=0.0)["fm"]
    assert torch.allclose(fa, fb)


def test_ref_encoder_masking(ema_ckpt):
    model = from_ema(ema_ckpt, n_voices=5).eval()
    ref = torch.randn(1, 80, 128)
    mask = torch.ones(1, 80, dtype=torch.bool)
    padded = torch.cat([ref, torch.randn(1, 40, 128) * 9], 1)
    pmask = torch.cat([mask, torch.zeros(1, 40, dtype=torch.bool)], 1)
    assert torch.allclose(model.ref_encoder(ref, mask), model.ref_encoder(padded, pmask), atol=1e-5)


def test_ref_aux_loss_grad_only_to_encoder(ema_ckpt):
    model = from_ema(ema_ckpt, n_voices=5)
    _, aux = model.voice_vectors(batch_for(model), p_ref=0.0)
    aux.backward()
    assert model.voice.weight.grad is None or model.voice.weight.grad.abs().sum() == 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.ref_encoder.parameters())


def test_save_load_roundtrip(ema_ckpt, tmp_path):
    model = from_ema(ema_ckpt, n_voices=5)
    meta = {"voices": list("abcde"), "stats": {"mean": torch.zeros(128), "std": torch.ones(128)}}
    save(model, tmp_path / "m.pt", meta)
    again, meta2 = load(tmp_path / "m.pt", "cpu")
    assert isinstance(again, VoiceAcoustic) and meta2["voices"] == meta["voices"]
    for k, v in model.state_dict().items():
        assert torch.equal(again.state_dict()[k], v)
