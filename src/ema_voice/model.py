"""EMA Lightning's acoustic model with voices: 128-dim DACVAE latents and a 64-dim voice vector.

A voice vector comes from the voice table or from the reference encoder (a few seconds of someone's speech). It
reaches the DiT through the timestep modulation, so every block and the output layer see it, and the duration
predictor through its input, so speaking rate follows the voice. Both projections start at zero: a fresh model
behaves exactly like EMA Lightning.
"""
import copy
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from ema_lightning.model import Acoustic, ConvNeXtBlock, masked_group_norm, rope

from ema_voice.timeline import letter_positions

LATENT_DIM = 128
VOICE_DIM = 64
NEW_LAYERS = ("in_proj.", "out_proj.", "voice", "voice_mod.", "voice_dur.", "ref_encoder.")


class RefEncoder(nn.Module):
    """Latents of a reference clip -> one voice vector, by attentive mean and std pooling."""

    def __init__(self, in_dim=LATENT_DIM, d=192, out_dim=VOICE_DIM, n_blocks=4):
        super().__init__()
        self.proj = nn.Linear(in_dim, d)
        self.blocks = nn.ModuleList(ConvNeXtBlock(d) for _ in range(n_blocks))
        self.norm = nn.LayerNorm(d)
        self.attn = nn.Linear(d, 1)
        self.out = nn.Linear(2 * d, out_dim)

    def forward(self, ref, mask):
        x = self.proj(ref) * mask[..., None]
        for block in self.blocks:
            x = block(x, mask) * mask[..., None]
        x = self.norm(x)
        w = self.attn(x).squeeze(-1).masked_fill(~mask, -1e4).softmax(-1)[..., None]
        mean = (w * x).sum(1)
        std = ((w * x * x).sum(1) - mean * mean).clamp(min=1e-5).sqrt()
        return self.out(torch.cat([mean, std], -1))


class VoiceAcoustic(Acoustic):
    def __init__(self, cfg, vocab, n_voices, voice_dim=VOICE_DIM):
        cfg = copy.deepcopy(cfg)
        cfg["latent_dim"] = LATENT_DIM
        super().__init__(cfg, vocab, shared=True)
        self.cfg, self.n_voices = cfg, n_voices
        self.voice = nn.Embedding(n_voices, voice_dim)
        nn.init.normal_(self.voice.weight, std=0.02)
        self.voice_mod = nn.Linear(voice_dim, self.d)
        self.voice_dur = nn.Linear(voice_dim, self.d)
        for layer in (self.voice_mod, self.voice_dur):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.ref_encoder = RefEncoder(LATENT_DIM, 192, voice_dim)

    def _log_dur(self, h, mask):
        """The duration predictor before its expm1: frames per letter in log1p space."""
        net, m = self.chardur.net, mask[:, None].float()
        x = masked_group_norm(F.silu(net[0](h.transpose(1, 2) * m)), net[2], m) * m
        x = masked_group_norm(F.silu(net[4](x)), net[6], m)
        return self.chardur.out(x.transpose(1, 2)).squeeze(-1).masked_fill(~mask, 0.0)

    def text_stage(self, ids, mask, v):
        """Letters and voice vectors -> text features and per-letter durations in frames."""
        h = self.text(ids, mask)
        log_d = self._log_dur(h + self.voice_dur(v)[:, None], mask)
        return h, torch.expm1(log_d.clamp(max=6.0)).clamp(min=1e-3) * mask.float()

    def conditions(self, h, dur, mask, cw, wstart, fw, fp):
        cp = letter_positions(dur, mask, cw, wstart)
        return self.aligner(h, cw, cp, fw, fp, mask, cw.shape[1])

    def velocity(self, x, cond, t, v, fmask):
        cos, sin = rope(torch.arange(x.shape[1], device=x.device), self.dh)
        c = self.t_embed(t) + self.voice_mod(v)
        bc = self.ada_shared(c).view(-1, 6, self.d)
        x = self.in_proj(x) + cond
        for block in self.blocks:
            x = block(x, bc, cos, sin, fmask)
        s, g = self.ada_out(c).chunk(2, -1)
        return self.out_proj(self.norm_out(x) * (1 + g.unsqueeze(1)) + s.unsqueeze(1))

    def voice_vectors(self, batch, p_ref=0.5):
        """Each item's voice from the table or, with probability p_ref, from its reference clip."""
        table = self.voice(batch.voice_ids)
        ref = self.ref_encoder(batch.ref, batch.ref_mask)
        use_ref = torch.rand(table.shape[0], device=table.device) < p_ref
        v = torch.where(use_ref[:, None], ref, table)
        return v, F.mse_loss(ref, table.detach())

    def loss(self, batch, p_drop=0.1, p_ref=0.5):
        v, ref_loss = self.voice_vectors(batch, p_ref)
        out = self.loss_for(batch, v, p_drop)
        out["total"] = out["total"] + ref_loss
        out["ref"] = ref_loss
        return out

    def loss_for(self, batch, v, p_drop=0.1):
        """Flow-matching and duration losses for given voice vectors [B, 64]."""
        h = self.text(batch.ids, batch.mask)
        log_d = self._log_dur(h + self.voice_dur(v)[:, None], batch.mask)
        m = batch.mask.float()
        dur_loss = (((log_d - torch.log1p(batch.dur_gt)) ** 2) * m).sum() / m.sum()

        cond = self.conditions(h, batch.dur_gt, batch.mask, batch.cw, batch.wstart, batch.fw, batch.fp)
        B = cond.shape[0]
        keep = (torch.rand(B, device=cond.device) >= p_drop).float()
        cond = cond * keep[:, None, None]
        t = torch.sigmoid(torch.randn(B, device=cond.device))
        eps = torch.randn_like(batch.z1)
        zt = (1 - t[:, None, None]) * eps + t[:, None, None] * batch.z1
        pred = self.velocity(zt, cond, t, v, batch.fmask)
        fm_mask = batch.fmask.float()[..., None]
        fm = (((pred - (batch.z1 - eps)) ** 2) * fm_mask).sum() / (fm_mask.sum() * pred.shape[-1])
        return {"total": fm + dur_loss, "fm": fm, "dur": dur_loss}


def from_ema(ema_path, n_voices):
    """A VoiceAcoustic warm-started from EMA Lightning's ema.pt; latent in/out layers start fresh."""
    ck = torch.load(ema_path, map_location="cpu", weights_only=False)
    model = VoiceAcoustic(ck["cfg"], ck["vocab"], n_voices)
    own = model.state_dict()
    shared = {k: v for k, v in ck["ema"].items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(shared, strict=False)
    nn.init.normal_(model.out_proj.weight, std=1e-3)
    nn.init.zeros_(model.out_proj.bias)
    return model


def save(model, path, meta, extra=None):
    """Written to a temporary file and renamed, so a crash never leaves a half-written checkpoint."""
    path = Path(path)
    ck = {"cfg": model.cfg, "vocab": model.vocab, "n_voices": model.n_voices,
          "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "meta": meta}
    if extra:
        ck.update(extra)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(ck, tmp)
    os.replace(tmp, path)


def load(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = VoiceAcoustic(ck["cfg"], ck["vocab"], ck["n_voices"])
    model.load_state_dict(ck["state"])
    return model.to(device).eval(), ck["meta"]


def is_new_layer(name):
    return name.startswith(NEW_LAYERS)
