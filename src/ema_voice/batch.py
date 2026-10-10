"""A padded training batch, padded the way EMA Lightning's engine pads (-1 for words, 0 elsewhere)."""
from dataclasses import dataclass, fields

import torch

from ema_voice.timeline import frames_from_counts


@dataclass
class Batch:
    ids: torch.Tensor  # [B, L] letters
    mask: torch.Tensor  # [B, L]
    cw: torch.Tensor  # [B, L] word of each letter, -1 past the end
    wstart: torch.Tensor  # [B, L] first letter of that word
    dur_gt: torch.Tensor  # [B, L] aligned frames per letter (float)
    fw: torch.Tensor  # [B, T] word of each frame, -1 past the end
    fp: torch.Tensor  # [B, T] position of the frame inside its word
    fmask: torch.Tensor  # [B, T]
    z1: torch.Tensor  # [B, T, 128] normalised latents
    voice_ids: torch.Tensor  # [B]
    ref: torch.Tensor  # [B, R, 128] normalised latents of another clip of the same voice
    ref_mask: torch.Tensor  # [B, R]

    def to(self, device):
        return Batch(**{f.name: getattr(self, f.name).to(device, non_blocking=True) for f in fields(self)})

    @property
    def frames(self):
        return int(self.fmask.sum())


def _stack(rows, fill, dtype=None):
    size = max(r.shape[0] for r in rows)
    out = torch.full((len(rows), size, *rows[0].shape[1:]), fill, dtype=dtype or rows[0].dtype)
    for i, r in enumerate(rows):
        out[i, :r.shape[0]] = r
    return out


def collate(items):
    """items: dicts with ids, cw, wstart (LongTensor [L]), letter_frames (float [L]), word_frames (Long [W]),
    z1 (float [T, 128]), voice_id (int), ref (float [R, 128])."""
    timelines = [frames_from_counts(it["word_frames"]) for it in items]
    return Batch(
        ids=_stack([it["ids"] for it in items], 0),
        mask=_stack([torch.ones(len(it["ids"]), dtype=torch.bool) for it in items], False),
        cw=_stack([it["cw"] for it in items], -1),
        wstart=_stack([it["wstart"] for it in items], 0),
        dur_gt=_stack([it["letter_frames"].float() for it in items], 0.0),
        fw=_stack([fw for fw, _ in timelines], -1),
        fp=_stack([fp for _, fp in timelines], 0.0),
        fmask=_stack([torch.ones(it["z1"].shape[0], dtype=torch.bool) for it in items], False),
        z1=_stack([it["z1"].float() for it in items], 0.0),
        voice_ids=torch.tensor([it["voice_id"] for it in items]),
        ref=_stack([it["ref"].float() for it in items], 0.0),
        ref_mask=_stack([torch.ones(it["ref"].shape[0], dtype=torch.bool) for it in items], False),
    )
