"""Text and voice vectors to latents: predicted durations, the frame timeline, then a midpoint ODE with CFG."""
import torch

from ema_voice.timeline import encode_text, frames_from_counts

MAX_WORD_FRAMES = 250
MAX_FRAMES = 3000


def _stack(rows, fill):
    out = torch.full((len(rows), max(r.shape[0] for r in rows), *rows[0].shape[1:]), fill, dtype=rows[0].dtype)
    for i, r in enumerate(rows):
        out[i, :r.shape[0]] = r
    return out


@torch.no_grad()
def synthesize_latents(model, texts, voices, speed=1.0, steps=8, cfg=2.0, seed=0, first_index=0):
    """Normalised latents [T_i, 128] per text. Item i's noise comes from seed and its index, so a text gets the
    same audio alone or in a batch (pass first_index for a batch that starts later in a list)."""
    device = voices.device
    enc = [encode_text(t, model.stoi) for t in texts]
    ids, cw, wstart = (_stack([e[k] for e in enc], fill).to(device) for k, fill in ((0, 0), (1, -1), (2, 0)))
    mask = ids != 0
    h, dur = model.text_stage(ids, mask, voices)
    dur = dur / speed
    counts = torch.zeros_like(dur).scatter_add_(1, cw.clamp(min=0), dur).round().clamp(1, MAX_WORD_FRAMES).long()
    timelines, noise = [], []
    for i, e in enumerate(enc):
        n = counts[i, :int(e[1][-1]) + 1].cpu()
        fw, fp = frames_from_counts(n)
        fw, fp = fw[:MAX_FRAMES], fp[:MAX_FRAMES]
        timelines.append((fw, fp))
        g = torch.Generator().manual_seed(seed * 1_000_003 + first_index + i)
        noise.append(torch.randn(fw.numel(), model.latent_dim, generator=g))
    fw = _stack([t[0] for t in timelines], -1).to(device)
    fp = _stack([t[1] for t in timelines], 0.0).to(device)
    fmask = fw >= 0
    cond = model.conditions(h, dur, mask, cw, wstart, fw, fp)
    x = _stack(noise, 0.0).to(device)

    def velocity(x, t):
        tt = torch.full((x.shape[0],), t, device=device)
        if cfg == 1.0:
            return model.velocity(x, cond, tt, voices, fmask)
        both = model.velocity(torch.cat([x, x]), torch.cat([cond, torch.zeros_like(cond)]), tt.repeat(2),
                              voices.repeat(2, 1), fmask.repeat(2, 1))
        c, u = both.chunk(2)
        return u + cfg * (c - u)

    dt = 1.0 / steps
    for k in range(steps):
        t = k * dt
        mid = x + 0.5 * dt * velocity(x, t)
        x = x + dt * velocity(mid, t + 0.5 * dt)
    return [x[i, :timelines[i][0].numel()].cpu() for i in range(len(texts))]
