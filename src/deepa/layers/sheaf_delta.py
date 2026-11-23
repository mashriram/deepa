"""Sheaf-Delta mixer: gated delta-rule matrix memory (global) + dyadic Householder sheaf transport (local).

    q, k, v      = W_qkv u            (per head, q and k L2-normalised)
    b_t          = beta_max * sigmoid(.)       in (0, beta_max)        step size; > 1 -> negative eigenvalue
    a_t          = decay_min + (1 - decay_min) * sigmoid(.)            decay
    o_delta      = gated delta rule(q, k, v, a, b), readout scale d_h^-1/2
    o_dyadic_t   = (1/sqrt(K)) * sum_{d in {1,2,4,8,16}} g_{t,d} * U_d v_{t-d}     U_d = product of M Householder
                   reflections (fixed per lag, acting on the full D-dim value), g = sigmoid(W_e u)
    y            = W_out[ (RMSNorm_head(o_delta) + o_dyadic) * SiLU(W_g u) ]

Decode state per sequence: S [H, d_h, d_h] + the last max_lag value vectors [max_lag, D]  (constant in T).
Parameter names (qkv, ba, g, out, edge, V, onorm) match the research code, so its checkpoints load unchanged.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DeepaConfig
from ..ops.delta_rule import gated_delta_rule, gated_delta_step
from .common import RMSNorm

MixerState = Tuple[torch.Tensor, torch.Tensor]  # (S [B,H,dk,dv] k-first, past values [B,max_lag,D] oldest -> newest)


def householder_coefs(a: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """Coefficients c of H_{M-1}..H_0 x = x - 2 sum_m c_m r_m, given a_m = <x, r_m> and Gram G[k,m,j] = <r_m, r_j>.
    a: [..., K, M]. Equivalent to applying the reflections r_0, r_1, ... in sequence without forming any matrix."""
    c = []
    for m in range(a.shape[-1]):
        cm = a[..., m]
        for j in range(m):
            cm = cm - 2 * c[j] * G[:, m, j]
        c.append(cm)
    return torch.stack(c, -1)


class SheafDeltaMixer(nn.Module):
    def __init__(self, cfg: DeepaConfig):
        super().__init__()
        D, H = cfg.dim, cfg.mixer_heads
        self.D, self.H, self.dh, self.M = D, H, D // H, cfg.householder
        self.offsets = cfg.lags
        self.K = len(self.offsets)
        self.P = cfg.max_lag
        self.beta_max, self.decay_min = cfg.beta_max, cfg.decay_min
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.ba = nn.Linear(D, 2 * H)                       # [:H] -> step size b, [H:] -> decay a
        nn.init.constant_(self.ba.bias[H:], cfg.decay_bias_init)
        self.g = nn.Linear(D, D, bias=False)                # output gate
        self.out = nn.Linear(D, D, bias=False)
        self.edge = nn.Linear(D, self.K * H)                # dyadic edge gates
        self.V = nn.Parameter(torch.randn(self.K, self.M, D))  # Householder reflectors per lag
        self.onorm = RMSNorm(self.dh, eps=cfg.norm_eps)
        self.impl = "auto"                                  # delta-rule backend: auto | fla | chunked | recurrent

    @classmethod
    def standalone(cls, dim: int, heads: int, **cfg_kwargs) -> "SheafDeltaMixer":
        """Use the mixer as a drop-in token mixer in any PyTorch model: ``SheafDeltaMixer.standalone(512, 8)(x)``.
        Input/output [B, T, dim]; causal; supports ``return_state=True`` and ``step()`` for O(1) decoding."""
        return cls(DeepaConfig(dim=dim, mixer_heads=heads, attn_heads=heads, n_layers=1, pattern="d", **cfg_kwargs))

    # ---------------------------------------------------------------- projections
    def _proj(self, x: torch.Tensor):
        sh = x.shape[:-1]
        q, k, v = self.qkv(x).view(*sh, 3, self.H, self.dh).unbind(-3)
        b, a = self.ba(x).view(*sh, 2, self.H).unbind(-2)
        b = self.beta_max * torch.sigmoid(b)
        a = self.decay_min + (1 - self.decay_min) * torch.sigmoid(a)
        return F.normalize(q, dim=-1), F.normalize(k, dim=-1), v, b, a

    def _reflectors(self):
        R = F.normalize(self.V, dim=-1)
        return R, torch.einsum("kmd,kjd->kmj", R, R)

    def _transport(self, u: torch.Tensor, k: int, R: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        """Apply lag-k transport U_k to u [..., D]."""
        a = torch.einsum("...d,md->...m", u, R[k])[..., None, :]
        c = householder_coefs(a, G[k:k + 1])[..., 0, :]
        return u - 2 * torch.einsum("...m,md->...d", c, R[k])

    def init_state(self, batch: int, device=None, dtype=torch.float32) -> MixerState:
        return (torch.zeros(batch, self.H, self.dh, self.dh, device=device, dtype=torch.float32),
                torch.zeros(batch, self.P, self.D, device=device, dtype=dtype))

    # ---------------------------------------------------------------- parallel (train / prefill)
    def forward(self, x: torch.Tensor, state: Optional[MixerState] = None, return_state: bool = False):
        B, T, D = x.shape
        q, k, v, b, a = self._proj(x)
        S0 = state[0] if state is not None else None
        o, S = gated_delta_rule(q, k, v, a, b, scale=self.dh ** -0.5, initial_state=S0,
                                output_final_state=return_state, impl=self.impl)
        y = self.onorm(o).reshape(B, T, D)
        # dyadic branch over [past values | current values]
        vf = v.reshape(B, T, D)
        past = state[1].to(vf.dtype) if state is not None else vf.new_zeros(B, self.P, D)
        full = torch.cat([past, vf], 1)
        g = torch.sigmoid(self.edge(x)).view(B, T, self.K, self.H)
        R, G = self._reflectors()
        acc = vf.new_zeros(B, T, self.H, self.dh)
        for i, d in enumerate(self.offsets):
            src = self._transport(full[:, self.P - d:self.P - d + T], i, R, G).view(B, T, self.H, self.dh)
            acc = acc + g[:, :, i, :, None] * src
        y = y + (acc / self.K ** 0.5).reshape(B, T, D)
        out = self.out(y * F.silu(self.g(x)))
        if return_state:
            return out, (S.float(), full[:, -self.P:])
        return out

    # ---------------------------------------------------------------- recurrent (decode)
    def step(self, x: torch.Tensor, state: Optional[MixerState]) -> Tuple[torch.Tensor, MixerState]:
        """x [B, D] -> (y [B, D], new state). O(1) in sequence length."""
        B = x.shape[0]
        if state is None:
            state = self.init_state(B, x.device, x.dtype)
        S, past = state
        q, k, v, b, a = self._proj(x)
        S, o = gated_delta_step(S, q.float(), k.float(), v.float(), a.float(), b.float(), self.dh ** -0.5)
        y = self.onorm(o.to(x.dtype)).reshape(B, self.D)
        g = torch.sigmoid(self.edge(x)).view(B, self.K, self.H)
        R, G = self._reflectors()
        acc = x.new_zeros(B, self.H, self.dh)
        for i, d in enumerate(self.offsets):
            acc = acc + g[:, i, :, None] * self._transport(past[:, self.P - d].to(x.dtype), i, R, G).view(B, self.H, self.dh)
        y = y + (acc / self.K ** 0.5).reshape(B, self.D)
        out = self.out(y * F.silu(self.g(x)))
        past = torch.cat([past[:, 1:], v.reshape(B, 1, self.D).to(past.dtype)], 1)
        return out, (S, past)


__all__ = ["SheafDeltaMixer", "householder_coefs"]
