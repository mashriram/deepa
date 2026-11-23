"""Gated delta rule (Gated DeltaNet recurrence) with optional negative eigenvalues.

Recurrence per head (state S in R^{d_k x d_v}, fla layout "k-first"):

    S_t = a_t * S_{t-1} (I - b_t k_t k_t^T) + b_t k_t v_t^T        (written here in k-first form)
    o_t = scale * S_t^T q_t

with a_t in (0, 1] (decay) and b_t in (0, 2) (step size; b_t > 1 gives eigenvalue 1 - b_t < 0 along k_t).

Three interchangeable implementations, all taking q, k, v of shape [B, T, H, d] and a, b of shape [B, T, H]:
  * ``gated_delta_rule_recurrent`` - token-by-token reference (exact, slow; any device)
  * ``gated_delta_rule_chunked``   - chunkwise WY form in pure PyTorch (exact up to fp rounding; any device)
  * fla's Triton ``chunk_gated_delta_rule`` - fast CUDA path (used automatically when available)
``gated_delta_rule`` dispatches. Note: b is passed in already squashed (b = beta_max * sigmoid(.)), so fla's
``allow_neg_eigval`` flag (which only affects its in-kernel sigmoid) is irrelevant and b in (0, 2) is used as is.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

Tensor = torch.Tensor

try:  # optional fast path
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_chunk
except Exception:  # pragma: no cover - depends on environment
    _fla_chunk = None


def fla_available() -> bool:
    return _fla_chunk is not None and torch.cuda.is_available()


def gated_delta_rule_recurrent(q: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor, scale: Optional[float] = None,
                               initial_state: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    """Reference implementation. Returns (o [B,T,H,dv], final state [B,H,dk,dv]). Computes in fp32."""
    dt = v.dtype
    q, k, v, a, b = (x.float() for x in (q, k, v, a, b))
    B, T, H, dk = k.shape
    scale = dk ** -0.5 if scale is None else scale
    S = q.new_zeros(B, H, dk, v.shape[-1]) if initial_state is None else initial_state.float().clone()
    out = []
    for t in range(T):
        S, o = gated_delta_step(S, q[:, t], k[:, t], v[:, t], a[:, t], b[:, t], scale)
        out.append(o)
    return torch.stack(out, 1).to(dt), S


def gated_delta_step(S: Tensor, q: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor, scale: float) -> Tuple[Tensor, Tensor]:
    """One decode step. S [B,H,dk,dv]; q,k [B,H,dk]; v [B,H,dv]; a,b [B,H]. Returns (S_new, o [B,H,dv])."""
    S = S * a[..., None, None]
    err = v - torch.einsum("bhkv,bhk->bhv", S, k)
    S = S + b[..., None, None] * k[..., :, None] * err[..., None, :]
    o = torch.einsum("bhkv,bhk->bhv", S, q) * scale
    return S, o


def gated_delta_rule_chunked(q: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor, scale: Optional[float] = None,
                             initial_state: Optional[Tensor] = None, chunk: int = 32) -> Tuple[Tensor, Tensor]:
    """Chunkwise-parallel exact form (pure PyTorch, fp32 internally).

    Inside a chunk let g_t = prod_{i<=t} a_i and S_t = g_t * R_t. Then R follows an *ungated* delta rule on
    v_t / g_t, whose chunk solution is W = (I + diag(b) L)^{-1} diag(b) (V/g - K R_0) with L = tril(K K^T, -1).
    """
    dt = v.dtype
    q, k, v, a, b = (x.float() for x in (q, k, v, a, b))
    B, T, H, dk = k.shape
    dv = v.shape[-1]
    scale = dk ** -0.5 if scale is None else scale
    pad = (-T) % chunk
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, 0, 0, pad)) for x in (q, k, v))
        b = F.pad(b, (0, 0, 0, pad))
        a = F.pad(a, (0, 0, 0, pad), value=1.0)
    n = (T + pad) // chunk
    to_c = lambda x: x.transpose(1, 2).reshape(B, H, n, chunk, *x.shape[3:])  # noqa: E731
    q, k, v = to_c(q), to_c(k), to_c(v)
    a, b = to_c(a), to_c(b)                                                   # [B,H,n,C]
    logg = torch.log(a).cumsum(-1)
    g = logg.exp()                                                            # inclusive cumulative decay
    eye = torch.eye(chunk, device=q.device)
    KK = torch.tril(k @ k.transpose(-1, -2), -1)
    Tinv = torch.linalg.solve_triangular(eye + b[..., None] * KK, eye.expand_as(KK), upper=False)
    TB = Tinv * b[..., None, :]                                               # (I + diag(b) L)^{-1} diag(b)
    QK = torch.tril(q @ k.transpose(-1, -2))
    R = q.new_zeros(B, H, dk, dv) if initial_state is None else initial_state.float().clone()
    outs = []
    for c in range(n):
        kc, qc, gc = k[:, :, c], q[:, :, c], g[:, :, c]
        W = TB[:, :, c] @ (v[:, :, c] / gc[..., None] - kc @ R)               # [B,H,C,dv]
        outs.append((qc @ R + QK[:, :, c] @ W) * gc[..., None] * scale)
        R = gc[..., -1, None, None] * (R + kc.transpose(-1, -2) @ W)
    o = torch.cat(outs, 2).reshape(B, H, n * chunk, dv).transpose(1, 2)[:, :T]
    return o.to(dt), R


def gated_delta_rule(q: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor, scale: Optional[float] = None,
                     initial_state: Optional[Tensor] = None, output_final_state: bool = False,
                     impl: str = "auto") -> Tuple[Tensor, Optional[Tensor]]:
    """Dispatch: impl in {"auto", "fla", "chunked", "recurrent"}. State layout is [B, H, dk, dv] everywhere."""
    if impl == "auto":
        impl = "fla" if (fla_available() and q.is_cuda) else "chunked"
    if impl == "fla":
        if _fla_chunk is None:
            raise RuntimeError("fla (flash-linear-attention) is not installed; pip install 'deepa-lm[fast]'")
        dt = v.dtype
        q, k, b = q.to(dt), k.to(dt), b.to(dt)
        g = torch.log(a.float()).to(dt)
        o, S = _fla_chunk(q, k, v, g=g, beta=b, scale=scale, initial_state=initial_state, output_final_state=output_final_state)
        return o, S
    fn = gated_delta_rule_chunked if impl == "chunked" else gated_delta_rule_recurrent
    o, S = fn(q, k, v, a, b, scale=scale, initial_state=initial_state)
    return o, (S if output_final_state else None)


__all__ = ["gated_delta_rule", "gated_delta_rule_chunked", "gated_delta_rule_recurrent", "gated_delta_step", "fla_available"]
