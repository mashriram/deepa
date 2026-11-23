"""Model configuration: the paper presets plus a general builder for any size (125M ... 200B+).

A DEEPA model is a stack of blocks described by a layer ``pattern`` string:
``"d"`` = Sheaf-Delta block (gated delta-rule memory + dyadic Householder transport, constant decode state),
``"a"`` = attention block (ALiBi or RoPE, KV cache). The paper model is ``"dddaddda"`` (3:1).

Three ways to get a config, from most to least hands-on:

    DeepaConfig(dim=2048, n_layers=24, pattern=..., ...)      # every knob explicit (like any Transformer config)
    DeepaConfig.from_params("2B", vocab_size=32000)           # pick a size; shape is solved for you
    DeepaConfig.deepa_alibi()                                 # the exact 14.4M model from the paper
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Optional, Union

# Conventional sizes; DeepaConfig.from_params("7b") etc. Values are targets, the solver gets within a few percent.
NAMED_SIZES = {"125m": 125e6, "350m": 350e6, "760m": 760e6, "1b": 1.0e9, "1.3b": 1.3e9, "2b": 2.0e9, "3b": 3.0e9,
               "7b": 7.0e9, "13b": 13e9, "30b": 30e9, "70b": 70e9, "200b": 200e9}


def spread_pattern(n_layers: int, ratio: int = 4) -> str:
    """One attention layer per ``ratio`` layers, attention last in each group ("dddaddda" for 8 layers, ratio 4).
    ratio=0 -> no attention at all (pure Sheaf-Delta)."""
    if ratio <= 0:
        return "d" * n_layers
    return "".join("a" if (i + 1) % ratio == 0 else "d" for i in range(n_layers))


def pattern_from_layers(n_layers: int, attn_layers: Iterable[int]) -> str:
    """Explicit placement: attention at the given 0-based layer indices, Sheaf-Delta everywhere else."""
    s = set(attn_layers)
    return "".join("a" if i in s else "d" for i in range(n_layers))


def _parse_count(x: Union[str, int, float]) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    s = x.strip().lower()
    if s in NAMED_SIZES:
        return NAMED_SIZES[s]
    m = re.fullmatch(r"([\d.]+)\s*([kmbt]?)", s)
    if not m:
        raise ValueError(f"cannot parse parameter count {x!r}; use e.g. '2B', '350M', 7e9")
    return float(m.group(1)) * {"": 1, "k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}[m.group(2)]


@dataclass
class DeepaConfig:
    # ---- size / shape
    vocab_size: int = 256                 # 256 = byte-level; set to your tokenizer's vocab for BPE / SentencePiece
    dim: int = 384
    n_layers: int = 8
    pattern: str = "dddaddda"             # "d" = Sheaf-Delta, "a" = attention; len == n_layers
    attn_heads: int = 8                   # attention query heads (head dim = dim / attn_heads)
    n_kv_heads: Optional[int] = None      # grouped-query attention: K/V heads (None = attn_heads, i.e. plain MHA)
    mixer_heads: int = 6                  # Sheaf-Delta heads (head dim = dim / mixer_heads; state = heads * d_h^2)
    # ---- attention
    position: str = "alibi"               # "alibi" (length-extrapolates) | "rope"
    rope_base: float = 10000.0
    # ---- Sheaf-Delta mixer
    householder: int = 2                  # reflections per dyadic transport map
    n_lags: int = 5                       # dyadic lags 1, 2, 4, ..., 2**(n_lags-1)
    beta_max: float = 2.0                 # b_t = beta_max * sigmoid(.); 2.0 -> eigenvalues in [-1, 1]
    decay_min: float = 0.8                # a_t = decay_min + (1 - decay_min) * sigmoid(.)
    decay_bias_init: float = 3.0          # initial a ~ 0.9905 (~73-token half-life)
    # ---- MLPs
    mlp_mult_mixer: float = 3.65          # SwiGLU width multiplier in Sheaf-Delta blocks (hidden = dim * mult * 2/3)
    attn_mlp: str = "gelu"                # MLP type in attention blocks: "gelu" (paper) | "swiglu"
    mlp_mult_attn: float = 3.65           # width multiplier in attention blocks (paper hybrid: 3.65; the paper's pure
                                          # attention baselines used 4.0)
    mlp_hidden_mixer: Optional[int] = None  # exact hidden size override (e.g. 5632); takes precedence over the mult
    mlp_hidden_attn: Optional[int] = None
    # ---- misc
    norm_eps: float = 1e-6
    init_std: float = 0.02
    scaled_init: bool = False             # GPT-2 style: residual output projections init std / sqrt(2 * n_layers)
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
        if self.attn_mlp not in ("gelu", "swiglu"):
            raise ValueError("attn_mlp must be 'gelu' or 'swiglu'")
        for h, name in ((self.attn_heads, "attn_heads"), (self.mixer_heads, "mixer_heads")):
            if self.dim % h:
                raise ValueError(f"dim={self.dim} must be divisible by {name}={h}")
        if self.attn_heads % self.kv_heads:
            raise ValueError(f"attn_heads={self.attn_heads} must be divisible by n_kv_heads={self.kv_heads}")

    # ------------------------------------------------------------------ derived
    @property
    def kv_heads(self) -> int:
        return self.n_kv_heads or self.attn_heads

    @property
    def attn_head_dim(self) -> int:
        return self.dim // self.attn_heads

    @property
    def mixer_head_dim(self) -> int:
        return self.dim // self.mixer_heads

    @property
    def lags(self):
        return [2 ** i for i in range(self.n_lags)]

    @property
    def max_lag(self) -> int:
        return self.lags[-1]

    def mlp_hidden(self, kind: str) -> int:
        """Hidden width of the MLP in a 'd' or 'a' block (matches deepa.layers.common.MLP)."""
        if kind == "d":
            return self.mlp_hidden_mixer or int(self.dim * self.mlp_mult_mixer * 2 / 3)
        gated = self.attn_mlp == "swiglu"
        return self.mlp_hidden_attn or int(self.dim * self.mlp_mult_attn * (2 / 3 if gated else 1))

    # ------------------------------------------------------------------ sizing (no model needed)
    def num_parameters(self) -> int:
        """Exact parameter count of DeepaLM(self), computed analytically (works for 200B configs on a laptop)."""
        D, V = self.dim, self.vocab_size
        H, K, M = self.mixer_heads, self.n_lags, self.householder
        dh_a = self.attn_head_dim
        d_mixer = 3 * D * D + (D * 2 * H + 2 * H) + D * D + D * D + (D * K * H + K * H) + K * M * D + D // H
        hd = self.mlp_hidden("d")
        d_block = 2 * D + d_mixer + D * 2 * hd + hd * D                        # norms + mixer + SwiGLU
        a_attn = D * (D + 2 * self.kv_heads * dh_a) + D * D
        ha = self.mlp_hidden("a")
        a_mlp = (D * 2 * ha + ha * D) if self.attn_mlp == "swiglu" else (D * ha + ha * D)
        a_block = 2 * D + a_attn + a_mlp
        n = V * D + D + self.pattern.count("d") * d_block + self.pattern.count("a") * a_block
        return n + (0 if self.tie_embeddings else V * D)

    def decode_state_bytes(self, seq_len: int, bytes_per_value: int = 2) -> int:
        """Per-sequence inference memory: constant Sheaf-Delta state + KV cache of the attention layers.
        (Sheaf-Delta state S is kept in fp32 by the reference decoder; this counts bytes_per_value for everything.)"""
        dh = self.mixer_head_dim
        per_d = self.mixer_heads * dh * dh + self.max_lag * self.dim
        per_a = 2 * seq_len * self.kv_heads * self.attn_head_dim
        return bytes_per_value * (self.pattern.count("d") * per_d + self.pattern.count("a") * per_a)

    approx_decode_state_bytes = decode_state_bytes  # backwards-compatible name

    def estimate_memory(self, seq_len: int = 4096, batch: int = 1, param_bytes: int = 2) -> Dict[str, float]:
        """Rough memory budget in GB: weights, per-batch decode state, and mixed-precision AdamW training state
        (bf16 weights+grads, fp32 master+2 moments = 16 bytes/param; activations excluded)."""
        n, gb = self.num_parameters(), 1024 ** 3
        return {"params_billion": n / 1e9, "weights_gb": n * param_bytes / gb,
                "decode_state_gb": batch * self.decode_state_bytes(seq_len, param_bytes) / gb,
                "transformer_kv_cache_gb_same_shape": batch * 2 * self.n_layers * seq_len * self.kv_heads * self.attn_head_dim * param_bytes / gb,
                "train_state_gb_adamw": n * 16 / gb}

    # ------------------------------------------------------------------ serialisation
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DeepaConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    # ------------------------------------------------------------------ paper presets (exact, checkpoint-compatible)
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
        """Paper scaling-study sizes ('2m', '6m', '14m', '38m'); any other size string goes to from_params()."""
        dims = {"2m": (192, 4, 4, 3), "6m": (256, 8, 4, 4), "14m": (384, 8, 8, 6), "38m": (512, 12, 8, 8)}
        if size.lower() not in dims:
            return cls.from_params(size, attn_every=0 if kind == "sheafdelta" else 4,
                                   position="rope" if kind == "deepa_rope" else "alibi", **kw)
        d, L, ha, hm = dims[size.lower()]
        base = dict(dim=d, n_layers=L, attn_heads=ha, mixer_heads=hm)
        if kind == "sheafdelta":
            return cls.sheafdelta(**{**base, **kw})
        return getattr(cls, kind)(**{**base, "pattern": spread_pattern(L), **kw})

    # ------------------------------------------------------------------ general builder (any size)
    @classmethod
    def build(cls, dim: int, n_layers: int, *, vocab_size: int = 32000, attn_every: int = 4,
              attn_layers: Optional[Iterable[int]] = None, head_dim: int = 128, mixer_head_dim: Optional[int] = None,
              gqa_groups: int = 4, position: str = "alibi", tie_embeddings: Optional[bool] = None,
              **overrides) -> "DeepaConfig":
        """Modern large-model recipe at an explicit shape, as simple as defining a Transformer:

            DeepaConfig.build(dim=2048, n_layers=24)                 # ~1.5B, 3:1 Sheaf-Delta:attention
            DeepaConfig.build(dim=8192, n_layers=80, attn_every=8)   # 70B-class, 7:1

        Defaults: head dim 128 for both attention and Sheaf-Delta, grouped-query attention (gqa_groups query heads
        per K/V head), SwiGLU MLPs (8/3 * dim) in every block, scaled residual init, untied embeddings when the
        vocabulary is large. attn_every=0 -> pure Sheaf-Delta. attn_layers=[...] places attention explicitly.
        Every field can still be overridden by keyword (e.g. mlp_hidden_mixer=5632, n_kv_heads=8)."""
        if dim % head_dim:
            raise ValueError(f"dim={dim} must be a multiple of head_dim={head_dim}")
        mixer_head_dim = mixer_head_dim or head_dim
        if dim % mixer_head_dim:
            raise ValueError(f"dim={dim} must be a multiple of mixer_head_dim={mixer_head_dim}")
        heads = dim // head_dim
        kv = max(1, heads // gqa_groups)
        while heads % kv:
            kv -= 1
        pattern = pattern_from_layers(n_layers, attn_layers) if attn_layers is not None else spread_pattern(n_layers, attn_every)
        cfg = dict(vocab_size=vocab_size, dim=dim, n_layers=n_layers, pattern=pattern, attn_heads=heads, n_kv_heads=kv,
                   mixer_heads=dim // mixer_head_dim, position=position, mlp_mult_mixer=4.0, attn_mlp="swiglu",
                   mlp_mult_attn=4.0, scaled_init=True,
                   tie_embeddings=(vocab_size <= 65536 and dim < 2048) if tie_embeddings is None else tie_embeddings,
                   max_position_embeddings=4096)
        cfg.update(overrides)
        return cls(**cfg)

    @classmethod
    def from_params(cls, target: Union[str, int, float], *, vocab_size: int = 32000, aspect: float = 112.0,
                    attn_every: int = 4, head_dim: int = 128, **kw) -> "DeepaConfig":
        """Pick dim and depth for a target size, e.g. from_params("2B") / ("350M") / (7e9) / ("200b").
        Searches dim in multiples of head_dim with depth ~ dim / aspect (rounded to a multiple of attn_every) and
        returns the shape whose exact parameter count is closest to the target. Extra kwargs go to build()."""
        target_n = _parse_count(target)
        step = max(attn_every, 1)
        dim_step = head_dim * kw.get("gqa_groups", 4)          # heads divisible by the GQA group -> clean kv heads
        best = None
        for dim in range(dim_step, 131072 + 1, dim_step):
            L0 = dim / aspect
            for L in range(step, int(L0 * 3.0) + step + 1, step):
                if L < 0.5 * L0:                                 # keep depth within ~[0.5, 3] x dim / aspect
                    continue
                cfg = cls.build(dim, L, vocab_size=vocab_size, attn_every=attn_every, head_dim=head_dim, **kw)
                # prefer the closest size; tie-break towards the target aspect ratio
                err = abs(math.log(cfg.num_parameters() / target_n)) + 0.02 * abs(math.log(L / L0))
                if best is None or err < best[0]:
                    best = (err, cfg)
            if cls.build(dim, step, vocab_size=vocab_size, attn_every=attn_every, head_dim=head_dim, **kw).num_parameters() > 2 * target_n:
                break
        return best[1]

    def summary(self) -> str:
        n = self.num_parameters()
        return (f"DEEPA {n / 1e9:.3g}B params | dim {self.dim} x {self.n_layers} layers | pattern {self.pattern} "
                f"({self.pattern.count('a')} attention, {self.pattern.count('d')} Sheaf-Delta) | attn heads "
                f"{self.attn_heads} (kv {self.kv_heads}), mixer heads {self.mixer_heads} x {self.mixer_head_dim} | "
                f"vocab {self.vocab_size} | {self.position}")

    def __repr__(self) -> str:  # compact
        return (f"DeepaConfig(pattern={self.pattern!r}, dim={self.dim}, position={self.position!r}, "
                f"attn_heads={self.attn_heads}, kv_heads={self.kv_heads}, mixer_heads={self.mixer_heads}, "
                f"vocab={self.vocab_size}, params={self.num_parameters() / 1e6:.1f}M)")


__all__ = ["DeepaConfig", "spread_pattern", "pattern_from_layers", "NAMED_SIZES"]
