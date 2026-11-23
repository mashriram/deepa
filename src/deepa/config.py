"""Model configuration and the published presets.

A DEEPA model is a stack of blocks described by a layer ``pattern`` string:
``"d"`` = Sheaf-Delta block (gated delta-rule memory + dyadic Householder transport, constant decode state),
``"a"`` = attention block (ALiBi or RoPE, KV cache). The paper model is ``"dddaddda"`` (3:1).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


def spread_pattern(n_layers: int, ratio: int = 4) -> str:
    """One attention layer per ``ratio`` layers, attention last in each group ("dddaddda" for 8 layers, ratio 4)."""
    return "".join("a" if (i + 1) % ratio == 0 else "d" for i in range(n_layers))


@dataclass
class DeepaConfig:
    vocab_size: int = 256                 # byte-level by default
    dim: int = 384
    n_layers: int = 8
    pattern: str = "dddaddda"             # "d" = Sheaf-Delta, "a" = attention; len == n_layers
    attn_heads: int = 8                   # attention heads (head dim = dim / attn_heads)
    mixer_heads: int = 6                  # Sheaf-Delta heads (head dim = dim / mixer_heads)
    position: str = "alibi"               # attention positional scheme: "alibi" | "rope"
    rope_base: float = 10000.0
    householder: int = 2                  # reflections per dyadic transport map
    n_lags: int = 5                       # dyadic lags 1, 2, 4, ..., 2**(n_lags-1)
    beta_max: float = 2.0                 # b_t = beta_max * sigmoid(.); 2.0 -> eigenvalues in [-1, 1]
    decay_min: float = 0.8                # a_t = decay_min + (1 - decay_min) * sigmoid(.)
    decay_bias_init: float = 3.0          # initial a ~ 0.9905 (~73-token half-life)
    mlp_mult_mixer: float = 3.65          # SwiGLU width multiplier in Sheaf-Delta blocks
    mlp_mult_attn: float = 3.65           # GELU width multiplier in attention blocks (paper hybrid: 3.65, same as mixer;
                                          # the pure-attention baselines in the paper used 4.0 to match parameter count)
    norm_eps: float = 1e-6
    init_std: float = 0.02
    tie_embeddings: bool = True
    max_position_embeddings: int = 1024   # training context; NOT a hard limit for ALiBi / Sheaf-Delta
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.pattern) != self.n_layers:
            raise ValueError(f"pattern {self.pattern!r} has {len(self.pattern)} entries but n_layers={self.n_layers}")
        if set(self.pattern) - {"a", "d"}:
            raise ValueError("pattern may only contain 'a' (attention) and 'd' (Sheaf-Delta)")
        if self.position not in ("alibi", "rope"):
            raise ValueError("position must be 'alibi' or 'rope'")
        for h, name in ((self.attn_heads, "attn_heads"), (self.mixer_heads, "mixer_heads")):
            if self.dim % h:
                raise ValueError(f"dim={self.dim} must be divisible by {name}={h}")

    @property
    def lags(self):
        return [2 ** i for i in range(self.n_lags)]

    @property
    def max_lag(self) -> int:
        return self.lags[-1]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DeepaConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    # ------------------------------------------------------------------ presets
    @classmethod
    def deepa_alibi(cls, **kw) -> "DeepaConfig":
        """The paper model: ALiBi-DEEPA 3:1, h6 (14.44M params at dim 384 / 8 layers, byte vocab)."""
        return cls(**{"pattern": "dddaddda", "position": "alibi", **kw})

    @classmethod
    def deepa_rope(cls, **kw) -> "DeepaConfig":
        """Same hybrid with RoPE attention (better on 2D-raster data such as images; does not length-extrapolate)."""
        return cls(**{"pattern": "dddaddda", "position": "rope", **kw})

    @classmethod
    def sheafdelta(cls, **kw) -> "DeepaConfig":
        """Pure Sheaf-Delta: no attention at all -> constant-size decode state, no KV cache."""
        n = kw.pop("n_layers", 8)
        return cls(**{"n_layers": n, "pattern": "d" * n, **kw})

    @classmethod
    def from_size(cls, size: str, kind: str = "deepa_alibi", **kw) -> "DeepaConfig":
        """Sizes used in the paper's scaling study: '2m', '6m', '14m', '38m'."""
        dims = {"2m": (192, 4, 4, 3), "6m": (256, 8, 4, 4), "14m": (384, 8, 8, 6), "38m": (512, 12, 8, 8)}
        d, L, ha, hm = dims[size.lower()]
        base = dict(dim=d, n_layers=L, attn_heads=ha, mixer_heads=hm)
        if kind == "sheafdelta":
            return cls.sheafdelta(**{**base, **kw})
        return getattr(cls, kind)(**{**base, "pattern": spread_pattern(L), **kw})

    def approx_decode_state_bytes(self, seq_len: int, bytes_per_value: int = 2) -> int:
        """Per-sequence decode memory: constant Sheaf-Delta state + KV cache of the attention layers."""
        dh = self.dim // self.mixer_heads
        per_d = self.mixer_heads * dh * dh + self.max_lag * self.dim
        per_a = 2 * seq_len * self.dim
        return bytes_per_value * (self.pattern.count("d") * per_d + self.pattern.count("a") * per_a)

    def __repr__(self) -> str:  # compact
        return (f"DeepaConfig(pattern={self.pattern!r}, dim={self.dim}, position={self.position!r}, "
                f"attn_heads={self.attn_heads}, mixer_heads={self.mixer_heads}, vocab={self.vocab_size})")


__all__ = ["DeepaConfig", "spread_pattern"]
