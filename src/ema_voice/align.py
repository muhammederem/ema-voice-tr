"""Letter durations at 25 Hz from forced alignment with torchaudio's MMS_FA.

Turkish letters are romanized one to one, so every letter of the text keeps its own aligned span. A word owns the
frames from its first letter to the next word's first letter: its letters, then the pause after it, which its
trailing spaces and punctuation share. The audio before the first letter goes to the first word.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF

from ema_voice.timeline import encode_text

RATE = 48000
HOP = 1920  # 25 Hz
ALIGN_RATE = 16000
MAX_BATCH_SECONDS = 120.0
MIN_SCORE = -1.5  # mean log-probability of the aligned letters; a wrong transcript scores far below
ROMAN = str.maketrans("çğıöşü", "cgiosu")


def romanize(text):
    return text.translate(ROMAN)


@dataclass
class Alignment:
    letter_frames: np.ndarray  # float32, one per character of the text, in 25 Hz frames
    word_frames: np.ndarray  # int64, one per word; sums to total_frames
    total_frames: int
    score: float


class Aligner:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        bundle = torchaudio.pipelines.MMS_FA
        self.model = bundle.get_model(with_star=False).to(self.device).eval()
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = self.model.to(self.dtype)
        self.dict = bundle.get_dict(star=None)
        self.stride = bundle.sample_rate // 50  # 20 ms emission frames

    def __call__(self, wav, text):
        return self.batch([wav], [text])[0]

    def batch(self, wavs, texts, max_seconds=MAX_BATCH_SECONDS):
        """Alignments for many clips, in sub-batches of at most max_seconds of audio; halves a sub-batch that
        runs out of GPU memory."""
        order = sorted(range(len(wavs)), key=lambda i: len(wavs[i]))
        out, group, total = [None] * len(wavs), [], 0.0
        groups = []
        for i in order:
            sec = len(wavs[i]) / RATE
            if group and total + sec > max_seconds:
                groups.append(group)
                group, total = [], 0.0
            group.append(i)
            total += sec
        if group:
            groups.append(group)
        while groups:
            g = groups.pop()
            audio = [AF.resample(torch.as_tensor(wavs[i], dtype=torch.float32), RATE, ALIGN_RATE) for i in g]
            try:
                emissions = self._emissions(audio)
            except torch.OutOfMemoryError:
                if len(g) == 1:
                    raise
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                groups += [g[:len(g) // 2], g[len(g) // 2:]]
                continue
            for i, e in zip(g, emissions, strict=True):
                out[i] = self._align(e, wavs[i], texts[i])
        return out

    @torch.no_grad()
    def _emissions(self, audio):
        """Log-probabilities per 20 ms frame for each 16 kHz clip."""
        lengths = torch.tensor([a.numel() for a in audio])
        # normalise each clip on its own: torchaudio's wrapper would normalise the whole padded batch together
        padded = torch.zeros(len(audio), int(lengths.max()))
        for i, a in enumerate(audio):
            padded[i, :a.numel()] = (a - a.mean()) / (a.std() + 1e-5)
        emissions, out_lengths = self.model.model(padded.to(self.device, self.dtype), lengths.to(self.device))
        emissions = torch.log_softmax(emissions.float(), -1).cpu()
        return [emissions[i, :int(n)] for i, n in enumerate(out_lengths.cpu())]

    def _align(self, emission, wav, text):
        roman = romanize(text)
        letters = [i for i, ch in enumerate(roman) if ch.isalpha() and ch in self.dict]
        if not letters or len(letters) > emission.shape[0]:
            return None
        targets = torch.tensor([[self.dict[roman[i]] for i in letters]], dtype=torch.int32)
        try:
            path, scores = AF.forced_align(emission[None], targets, blank=0)
        except (RuntimeError, ValueError):  # e.g. more letters than frames
            return None
        spans = AF.merge_tokens(path[0], scores[0])
        if len(spans) != len(letters):
            return None
        score = float(torch.cat([scores[0, s.start:s.end] for s in spans]).mean())
        if not score >= MIN_SCORE:  # also rejects NaN
            return None
        sec = self.stride / ALIGN_RATE
        starts = {i: s.start * sec for i, s in zip(letters, spans, strict=True)}
        ends = {i: s.end * sec for i, s in zip(letters, spans, strict=True)}
        return timeline(text, starts, ends, int(np.ceil(len(wav) / HOP)), score)


def timeline(text, starts, ends, total, score):
    """Aligned letter spans (seconds, keyed by character index) into per-word and per-character frames."""
    _, cw, _ = encode_text(text, {})
    cw = cw.numpy()
    n_words = int(cw[-1]) + 1
    # a word starts at its first letter; a word without letters (a lone "-") right after the previous letter
    begin, last_end = [], 0.0
    for w in range(n_words):
        lets = [i for i in np.flatnonzero(cw == w) if i in starts]
        begin.append(starts[lets[0]] if lets else last_end)
        if lets:
            last_end = ends[lets[-1]]
    bounds = [0] + [round(t * RATE / HOP) for t in begin[1:]] + [total]
    for w in range(1, n_words + 1):  # every word keeps at least one frame
        bounds[w] = max(bounds[w], bounds[w - 1] + 1)
    if bounds[-1] > total:
        return None
    word_frames = np.diff(bounds).astype(np.int64)
    letter_frames = np.zeros(len(text), np.float32)
    fps = RATE / HOP
    for w in range(n_words):
        chars = np.flatnonzero(cw == w)
        lets = [i for i in chars if i in starts]
        rest = [i for i in chars if i not in starts]
        if not lets:
            letter_frames[chars] = word_frames[w] / len(chars)
            continue
        for a, b in zip(lets, lets[1:]):
            letter_frames[a] = (starts[b] - starts[a]) * fps
        letter_frames[lets[-1]] = (ends[lets[-1]] - starts[lets[-1]]) * fps
        letter_frames[lets[0]] += max(0.0, starts[lets[0]] * fps - bounds[w])  # lead-in, e.g. initial silence
        gap = word_frames[w] - letter_frames[chars].sum()
        if rest and gap > 0:
            letter_frames[rest] = gap / len(rest)
        letter_frames[chars] *= word_frames[w] / max(letter_frames[chars].sum(), 1e-6)
    return Alignment(letter_frames, word_frames, total, score)
