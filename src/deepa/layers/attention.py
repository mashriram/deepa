"""Causal attention with ALiBi or RoPE positions and a simple KV cache.

ALiBi fast path ("folded" ALiBi): score = s*q.k - m_h*(i - j). The -m_h*i term is constant along a query row, so softmax
ignores it; the remaining m_h*j is written as a dot product by appending [256*m_h/s, m_h/s] to q and [j // 256, j % 256]
to k. The augmented head dim is zero-padded to a multiple of 8 and run with ``is_causal=True`` on the fused
FlashAttention / memory-efficient SDPA kernel: O(T) memory, no [H, T, T] bias tensor. Exact up to bf16 rounding of the
two query constants (matches the dense-bias path to ~4e-5 bits/byte in the paper's validation).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DeepaConfig

KV = Tuple[torch.Tensor, torch.Tensor]  # (k, v), each [B, H, T_past, d]
_SLOPES: Dict[Tuple[int, str], torch.Tensor] = {}


def alibi_slopes(H: int, device) -> torch.Tensor:
    key = (H, str(torch.device(device)))
    if key not in _SLOPES:
        _SLOPES[key] = torch.tensor([2.0 ** (-8.0 * (i + 1) / H) for i in range(H)], device=device, dtype=torch.float32)
    return _SLOPES[key]


def _pad_dim(dh: int) -> int:
    return -(-(dh + 2) // 8) * 8


def alibi_attention_folded(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q, k, v [B, H, T, d] with no past (T_q == T_k). Fused-kernel exact ALiBi."""
    B, H, T, dh = q.shape
    s, m, pad = dh ** -0.5, alibi_slopes(H, q.device), _pad_dim(dh)
    pos = torch.arange(T, device=q.device)
    kx = torch.stack([(pos // 256).float(), (pos % 256).float()], -1)[None, None].expand(B, H, T, 2).to(k.dtype)
    qx = torch.stack([256 * m / s, m / s], -1)[None, :, None, :].expand(B, H, T, 2).to(q.dtype)
    z = lambda t: t.new_zeros(*t.shape[:-1], pad - dh - 2)  # noqa: E731
    q2, k2 = torch.cat([q, qx, z(q)], -1), torch.cat([k, kx, z(k)], -1)
    v2 = torch.cat([v, v.new_zeros(B, H, T, pad - dh)], -1)
    return F.scaled_dot_product_attention(q2, k2, v2, is_causal=True, scale=s)[..., :dh]


def alibi_attention_dense(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_offset: int = 0, chunk: int = 1024) -> torch.Tensor:
    """Explicit-bias ALiBi; supports a KV cache (queries at positions q_offset .. q_offset+T_q-1, keys at 0 .. T_k-1).
    Query rows are processed in chunks so memory is O(chunk * T_k) instead of O(T_q * T_k)."""
    H, Tq, Tk = q.shape[1], q.shape[2], k.shape[2]
    m = alibi_slopes(H, q.device)
    kj = torch.arange(Tk, device=q.device)
    outs = []
    for i in range(0, Tq, chunk):
        qi = torch.arange(q_offset + i, q_offset + min(i + chunk, Tq), device=q.device)
        rel = (kj[None, :] - qi[:, None]).float()
        bias = (m[:, None, None] * rel[None]).masked_fill(rel[None] > 0, float("-inf"))
        outs.append(F.scaled_dot_product_attention(q[:, :, i:i + chunk], k, v, attn_mask=bias[None].to(q.dtype)))
    return torch.cat(outs, 2)


def rope(x: torch.Tensor, pos: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Half-split rotary embedding. x [B, H, T, d], pos [T]."""
    dh = x.shape[-1]
    inv = 1.0 / (base ** (torch.arange(0, dh, 2, device=x.device).float() / dh))
    ang = pos[:, None].float() * inv[None]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., :dh // 2].float(), x[..., dh // 2:].float()
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: DeepaConfig):
        super().__init__()
        self.h, self.kv, self.position, self.rope_base = cfg.attn_heads, cfg.kv_heads, cfg.position, cfg.rope_base
        self.dh = cfg.dim // cfg.attn_heads
        # q | k | v packed in one projection; with kv == h this is exactly the paper layout (3 * dim outputs)
        self.qkv = nn.Linear(cfg.dim, cfg.dim + 2 * self.kv * self.dh, bias=False)
        self.o = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.alibi_impl = "auto"  # auto | folded | dense

    def forward(self, x: torch.Tensor, cache: Optional[KV] = None, return_state: bool = False):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split([D, self.kv * self.dh, self.kv * self.dh], -1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.kv, self.dh).transpose(1, 2)
        v = v.view(B, T, self.kv, self.dh).transpose(1, 2)
        past = 0 if cache is None else cache[0].shape[2]
        if self.position == "rope":
            pos = torch.arange(past, past + T, device=x.device)
            q, k = rope(q, pos, self.rope_base), rope(k, pos, self.rope_base)
        if cache is not None:
            k, v = torch.cat([cache[0], k], 2), torch.cat([cache[1], v], 2)
        new_cache = (k, v)                                   # cache keeps only the kv heads (GQA memory saving)
        if self.kv != self.h:
            k, v = k.repeat_interleave(self.h // self.kv, 1), v.repeat_interleave(self.h // self.kv, 1)
        if self.position == "alibi":
            folded = self.alibi_impl == "folded" or (self.alibi_impl == "auto" and past == 0 and x.is_cuda
                                                      and q.dtype in (torch.float16, torch.bfloat16))
            y = alibi_attention_folded(q, k, v) if folded and past == 0 else alibi_attention_dense(q, k, v, past)
        else:
            if past == 0:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:  # bottom-right aligned causal mask for cached decoding / chunked prefill
                mask = torch.ones(T, k.shape[2], dtype=torch.bool, device=x.device).tril(past)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = self.o(y.transpose(1, 2).reshape(B, T, D))
        return (out, new_cache) if return_state else out


__all__ = ["Attention", "alibi_slopes", "alibi_attention_folded", "alibi_attention_dense", "rope"]
