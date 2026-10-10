"""Text and frame timelines, built by exactly the rules EMA Lightning uses at inference.

A word is a run of letters plus whatever spaces and punctuation follow it, so a word's frames include the pause
after it. Training builds the timeline from aligned durations; inference from predicted ones.
"""
from itertools import pairwise

import torch
from ema_lightning.chunker import finish
from ema_lightning.frontend import Frontend


def encode_text(text, stoi):
    """Letter ids, the word of each letter and the first letter of that word (as `Engine.piece`)."""
    ids = [stoi.get(ch, 1) for ch in text]
    starts = [i for i, ch in enumerate(text) if ch != " " and (i == 0 or text[i - 1] == " ")] or [0]
    bounds = [0] + starts[1:] + [len(text)]
    cw, wstart = [], []
    for w, (a, b) in enumerate(pairwise(bounds)):
        cw += [w] * (b - a)
        wstart += [a] * (b - a)
    return torch.tensor(ids), torch.tensor(cw), torch.tensor(wstart)


def prepare_text(text_spoken, vocab):
    """Already-spoken text into the model's alphabet, ending on a sentence end as EMA's chunker does."""
    frontend = Frontend.__new__(Frontend)  # skip the normalizer: the text is already in spoken form
    frontend.vocab = frozenset(vocab)
    text = frontend.alphabet(text_spoken)
    return finish(text) if text else ""


def frames_from_counts(n):
    """Frames per word into the word of each frame and its position inside the word (as `Engine.plan`)."""
    frames = int(n.sum())
    fw = torch.repeat_interleave(torch.arange(n.numel()), n)[:frames]
    fp = ((torch.arange(frames) - (n.cumsum(0) - n)[fw]).double() / n[fw].double()).float()
    return fw, fp


def letter_positions(dur, mask, cw, wstart):
    """Each letter's centre as a fraction of its word's duration (as `Acoustic.sound_stage`)."""
    c = dur.clamp(min=1e-4) * mask
    done = c.cumsum(-1)
    before = done - c
    word = cw.clamp(min=0)
    total = torch.zeros_like(c).scatter_add_(1, word, c).gather(1, word)
    return ((done - before.gather(1, wstart) - 0.5 * c) / total.clamp(min=1e-8)).clamp(0.0, 1.0) * mask
