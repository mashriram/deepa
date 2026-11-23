"""Small shared modules. Parameter names match the research code (hsnlm) so trained checkpoints load directly."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class MLP(nn.Module):
    """``gated=True``: SwiGLU (hidden = int(dim * mult * 2/3)); ``gated=False``: GELU (hidden = int(dim * mult)).
    ``hidden`` overrides the multiplier with an exact width."""

    def __init__(self, dim: int, mult: float, gated: bool, hidden: int = None):
        super().__init__()
        self.gated = gated
        hidden = hidden or int(dim * mult * (2 / 3 if gated else 1))
        self.up = nn.Linear(dim, hidden * (2 if gated else 1), bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.up(x)
        if self.gated:
            a, b = u.chunk(2, -1)
            return self.down(F.silu(a) * b)
        return self.down(F.gelu(u))
