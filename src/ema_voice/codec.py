"""DACVAE (facebook/dacvae-watermarked): 48 kHz audio to 128-dim latents at 25 Hz and back.

Encoding keeps the posterior mean, so the same audio always gives the same latents. The model trains on latents
normalised per dimension with statistics measured on the training data.
"""
import contextlib

import numpy as np
import torch
from dacvae import DACVAE

REPO = "facebook/dacvae-watermarked"
RATE = 48000
HOP = 1920
DIM = 128


class Codec:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.model = DACVAE.load(REPO).to(self.device).eval().requires_grad_(False)
        self.hop, self.sample_rate, self.dim = HOP, RATE, DIM

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    @torch.no_grad()
    def encode(self, wav):
        """[B, S] audio -> [B, ceil(S / 1920), 128] latent means."""
        with self._autocast():
            h = self.model.encoder(self.model._pad(wav.to(self.device)[:, None]))
            mean, _ = self.model.quantizer.in_proj(h).chunk(2, dim=1)
        return mean.float().transpose(1, 2).cpu()

    def encode_list(self, wavs, max_batch_seconds=60.0):
        """Variable-length clips -> one float16 [T_i, 128] array each, batched by similar length; a batch that
        runs out of GPU memory is split in half and retried."""
        order = sorted(range(len(wavs)), key=lambda i: len(wavs[i]))
        groups, cur = [], []
        for i in order:
            if cur and len(wavs[i]) * (len(cur) + 1) > max_batch_seconds * RATE:
                groups.append(cur)
                cur = []
            cur.append(i)
        if cur:
            groups.append(cur)
        out = [None] * len(wavs)
        while groups:
            g = groups.pop()
            padded = torch.zeros(len(g), max(len(wavs[i]) for i in g))
            for row, i in enumerate(g):
                padded[row, :len(wavs[i])] = torch.as_tensor(wavs[i])
            try:
                z = self.encode(padded)
            except torch.OutOfMemoryError:
                if len(g) == 1:
                    raise
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                groups += [g[:len(g) // 2], g[len(g) // 2:]]
                continue
            for row, i in enumerate(g):
                out[i] = z[row, :int(np.ceil(len(wavs[i]) / HOP))].numpy().astype(np.float16)
        return out

    @torch.no_grad()
    def decode(self, z):
        """[B, T, 128] latents -> [B, T * 1920] audio."""
        with self._autocast():
            audio = self.model.decode(z.to(self.device).transpose(1, 2))
        return audio.float()[:, 0].cpu()


def normalize(z, stats):
    return (z - stats["mean"].to(z.device)) / stats["std"].to(z.device)


def denormalize(z, stats):
    return z * stats["std"].to(z.device) + stats["mean"].to(z.device)
