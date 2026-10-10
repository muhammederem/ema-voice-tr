import numpy as np
import torch
import pytest

from ema_voice.align import HOP, Aligner, romanize


@pytest.fixture(scope="module")
def aligner():
    return Aligner("cpu")


def test_romanize_preserves_length():
    text = "çığöşü ışık, güzel!"
    out = romanize(text)
    assert len(out) == len(text)
    assert out == "cigosu isik, guzel!"


def test_alignment_sums(aligner, ema_clips):
    for wav, text in ema_clips:
        a = aligner(wav, text)
        assert a is not None, text
        total = int(np.ceil(len(wav) / HOP))
        assert a.total_frames == total
        assert a.word_frames.sum() == total
        assert (a.word_frames >= 1).all()
        assert (a.letter_frames >= 0).all()
        assert len(a.letter_frames) == len(text)
        assert a.letter_frames.sum() == pytest.approx(total, abs=1e-3)


def test_letters_sum_to_their_word(aligner, ema_clips):
    from ema_voice.timeline import encode_text

    wav, text = ema_clips[1]
    a = aligner(wav, text)
    _, cw, _ = encode_text(text, {})
    per_word = np.zeros(len(a.word_frames))
    np.add.at(per_word, cw.numpy(), a.letter_frames)
    assert np.allclose(per_word, a.word_frames, atol=1e-3)


def test_wrong_text_rejected(aligner, ema_clips):
    wav, _ = ema_clips[0]
    assert aligner(wav, ema_clips[2][1]) is None


def test_batch_matches_single(aligner, ema_clips):
    wavs, texts = [w for w, _ in ema_clips], [t for _, t in ema_clips]
    batch = aligner.batch(wavs, texts)
    for (wav, text), b in zip(ema_clips, batch):
        s = aligner(wav, text)
        assert np.array_equal(s.word_frames, b.word_frames)


def test_hyphen_and_apostrophe_are_not_aligned_as_letters(aligner, ema_clips):
    wav, text = ema_clips[1]  # "siparişiniz yarın sabah kargoya verilecek."
    a = aligner(wav, text.replace("yarın sabah", "yarın-sabah").replace("kargoya", "kargo'ya"))
    assert a is not None and a.word_frames.sum() == a.total_frames


def test_batch_splits_by_seconds_and_survives_oom(aligner, ema_clips, monkeypatch):
    wavs, texts = [w for w, _ in ema_clips], [t for _, t in ema_clips]
    want = [a.word_frames for a in aligner.batch(wavs, texts)]
    real = aligner._emissions
    sizes = []

    def picky(audio):
        sizes.append(len(audio))
        if len(audio) > 1:
            raise torch.OutOfMemoryError("fake")
        return real(audio)

    monkeypatch.setattr(aligner, "_emissions", picky)
    got = aligner.batch(wavs, texts, max_seconds=5.0)
    assert all(np.array_equal(a.word_frames, b) for a, b in zip(got, want))
    assert max(sizes) <= 2


def test_words_without_letters_keep_the_clip(aligner, ema_clips):
    wav, text = ema_clips[2]
    for variant in ["- " + text, text.replace(" henüz ", " - henüz ")]:
        a = aligner(wav, variant)
        assert a is not None and (a.word_frames >= 1).all() and a.word_frames.sum() == a.total_frames


def test_alignment_independent_of_batch(aligner, ema_clips):
    wav, text = ema_clips[0]
    loud = (ema_clips[3][0] * 8).clip(-1, 1)
    alone = aligner(wav, text)
    together = aligner.batch([wav, loud], [text, ema_clips[3][1]])[0]
    assert np.array_equal(alone.word_frames, together.word_frames)
