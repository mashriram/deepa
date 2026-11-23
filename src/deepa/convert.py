"""Load checkpoints produced by the research code (hsnlm / benchmarks.deepa_common).

The research models use the same module tree (embed, blocks.i.{n1,n2,mix,mlp}, norm_f; mixer params qkv/ba/g/out/edge/V/
onorm; attention params qkv/o), so conversion is a key-for-key copy plus a config built from the run's hyper-parameters.
"""
from __future__ import annotations

from typing import Optional

import torch

from .config import DeepaConfig
from .model import DeepaLM

_KIND = {"alibi-hybrid-h6": "deepa_alibi", "rope-hybrid-h6": "deepa_rope", "sheafdelta-pure": "sheafdelta"}


def from_research_checkpoint(path_or_state, kind: str = "alibi-hybrid-h6", dim: int = 384, layers: int = 8,
                             heads: int = 8, mixer_heads: int = 6, vocab: int = 256, beta_max: float = 2.0,
                             map_location: str = "cpu", strict: bool = True,
                             cfg: Optional[DeepaConfig] = None) -> DeepaLM:
    """kind: 'alibi-hybrid-h6' (paper model), 'rope-hybrid-h6', or 'sheafdelta-pure' (the names used in the benchmarks)."""
    sd = torch.load(path_or_state, map_location=map_location) if isinstance(path_or_state, str) else path_or_state
    if cfg is None:
        base = dict(dim=dim, n_layers=layers, attn_heads=heads, mixer_heads=mixer_heads, vocab_size=vocab, beta_max=beta_max)
        from .config import spread_pattern
        if kind == "sheafdelta-pure":
            cfg = DeepaConfig.sheafdelta(**base)
        else:
            cfg = getattr(DeepaConfig, _KIND[kind])(**{**base, "pattern": spread_pattern(layers)})
    model = DeepaLM(cfg)
    model.load_state_dict(sd, strict=strict)
    return model


__all__ = ["from_research_checkpoint"]
