"""DEEPA: hybrid gated-delta-rule / attention language models with constant-size recurrent state.

    from deepa import DeepaConfig, DeepaLM
    model = DeepaLM(DeepaConfig.deepa_alibi())      # paper model (ALiBi-DEEPA 3:1, 14.4M params)
    model = DeepaLM(DeepaConfig.sheafdelta())       # pure Sheaf-Delta (no attention, no KV cache)
"""
from .config import DeepaConfig, spread_pattern
from .layers.attention import Attention
from .layers.sheaf_delta import SheafDeltaMixer
from .model import Block, DeepaCache, DeepaLM
from .ops.delta_rule import fla_available, gated_delta_rule
from .tokenizer import ByteTokenizer

__version__ = "0.1.0"
__all__ = ["DeepaConfig", "DeepaLM", "DeepaCache", "Block", "SheafDeltaMixer", "Attention", "ByteTokenizer",
           "gated_delta_rule", "fla_available", "spread_pattern", "__version__"]
