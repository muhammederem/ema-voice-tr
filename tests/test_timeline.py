import torch
from ema_lightning.engine import Engine
from ema_lightning.model import load_acoustic

from ema_voice.timeline import encode_text, frames_from_counts, letter_positions, prepare_text

TEXTS = ["merhaba, dünya.", ",önce noktalama.", "iki  boşluk var.", "a", "tek kelimelik?"]



def test_encode_text_matches_engine(ema_ckpt):
    model = load_acoustic(ema_ckpt, "cpu")
    eng = Engine.__new__(Engine)
    eng.model = model
    for text in TEXTS:
        piece = eng.piece(text, 0.0, 0)
        ids, cw, wstart = encode_text(text, model.stoi)
        assert torch.equal(ids, piece.ids) and torch.equal(cw, piece.cw) and torch.equal(wstart, piece.wstart)


def test_frames_from_counts_matches_engine():
    n = torch.tensor([3, 1, 5])
    fw, fp = frames_from_counts(n)
    frames = int(n.sum())
    want_fw = torch.repeat_interleave(torch.arange(n.numel()), n)[:frames]
    want_fp = ((torch.arange(frames) - (n.cumsum(0) - n)[want_fw]).double() / n[want_fw].double()).float()
    assert torch.equal(fw, want_fw) and torch.equal(fp, want_fp)


def test_letter_positions_matches_sound_stage(ema_ckpt):
    model = load_acoustic(ema_ckpt, "cpu")
    ids, cw, wstart = encode_text("merhaba, güzel dünya.", model.stoi)
    dur = torch.rand(1, ids.numel()) * 5
    mask = torch.ones(1, ids.numel(), dtype=torch.bool)
    c = dur.clamp(min=1e-4) * mask
    done = c.cumsum(-1)
    before = done - c
    word = cw[None].clamp(min=0)
    total = torch.zeros_like(c).scatter_add_(1, word, c).gather(1, word)
    want = ((done - before.gather(1, wstart[None]) - 0.5 * c) / total.clamp(min=1e-8)).clamp(0.0, 1.0) * mask
    assert torch.allclose(letter_positions(dur, mask, cw[None], wstart[None]), want)


def test_prepare_text_ends_with_terminal_punct(ema_ckpt):
    vocab = load_acoustic(ema_ckpt, "cpu").vocab
    assert prepare_text("Merhaba, nasılsınız", vocab) == "merhaba, nasılsınız."
    assert prepare_text("İSTANBUL ve IŞIK!", vocab) == "istanbul ve ışık!"
