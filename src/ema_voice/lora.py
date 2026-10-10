"""Low-rank adapters on the DiT's linear layers, for one added voice.

Training wraps the layers; inference adds B @ A into the weights for the duration of one call and then puts the
original tensors back, so other voices are untouched bit for bit.
"""
import contextlib

import torch
import torch.nn as nn

TARGETS = ("attn.qkv", "attn.proj", "ff.w1", "ff.w2", "ff.w3")
RANK, ALPHA = 8, 16


def target_layers(model):
    return {f"blocks.{i}.{t}": block.get_submodule(t) for i, block in enumerate(model.blocks) for t in TARGETS}


class LoRALinear(nn.Module):
    def __init__(self, base, rank=RANK, alpha=ALPHA):
        super().__init__()
        self.base, self.scale = base, alpha / rank
        self.A = nn.Parameter(torch.randn(rank, base.in_features, device=base.weight.device) / base.in_features**0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


@contextlib.contextmanager
def wrapped(model):
    """Wrap the target layers with fresh adapters; yields {name: LoRALinear}; unwraps on exit."""
    layers = target_layers(model)
    adapters = {}
    for name, layer in layers.items():
        parent, attr = name.rsplit(".", 1)
        adapters[name] = LoRALinear(layer)
        setattr(model.get_submodule(parent), attr, adapters[name])
    try:
        yield adapters
    finally:
        for name, layer in layers.items():
            parent, attr = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), attr, layer)


def export(adapters):
    return {name: {"A": a.A.detach().cpu(), "B": a.B.detach().cpu(), "scale": a.scale} for name, a in adapters.items()}


@contextlib.contextmanager
def applied(model, lora):
    """Merge saved adapters into the weights for this block only."""
    if not lora:
        yield
        return
    layers = target_layers(model)
    originals = {name: layers[name].weight.detach().clone() for name in lora}
    try:
        with torch.no_grad():
            for name, ab in lora.items():
                w = layers[name].weight
                w += (ab["B"].to(w) @ ab["A"].to(w)) * ab["scale"]
        yield
    finally:
        with torch.no_grad():
            for name, w in originals.items():
                layers[name].weight.copy_(w)
