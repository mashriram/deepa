"""``diffusers`` integration: DEEPA as a ModelMixin/ConfigMixin sequence backbone.

Gives a DEEPA stack the diffusers model API (``save_pretrained`` / ``from_pretrained`` / ``register_to_config``,
device/dtype helpers) so it can be used inside diffusers pipelines, e.g. as a causal byte-level *text encoder*
producing ``encoder_hidden_states`` for cross-attention conditioning, or as a causal 1-D sequence backbone.

Scope note: DEEPA is causal (left-to-right). It has been evaluated as a language model, not as a bidirectional
denoiser; using it as a diffusion denoising backbone is untested.

    from deepa.integrations.diffusers import DeepaSequenceModel
    enc = DeepaSequenceModel(pattern="dddaddda")
    hidden = enc(input_ids).last_hidden_state          # [B, T, dim]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

try:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.utils import BaseOutput
except ImportError as e:  # pragma: no cover
    raise ImportError("deepa.integrations.diffusers requires `diffusers` (pip install 'deepa-lm[diffusers]')") from e

from ..config import DeepaConfig
from ..model import DeepaLM


@dataclass
class DeepaSequenceOutput(BaseOutput):
    last_hidden_state: torch.Tensor
    logits: Optional[torch.Tensor] = None


class DeepaSequenceModel(ModelMixin, ConfigMixin):
    config_name = "config.json"
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self, vocab_size: int = 256, dim: int = 384, n_layers: int = 8, pattern: str = "dddaddda",
                 attn_heads: int = 8, mixer_heads: int = 6, position: str = "alibi", beta_max: float = 2.0,
                 mlp_mult_mixer: float = 3.65, mlp_mult_attn: float = 3.65, max_position_embeddings: int = 1024):
        super().__init__()
        self.model = DeepaLM(DeepaConfig(vocab_size=vocab_size, dim=dim, n_layers=n_layers, pattern=pattern,
                                         attn_heads=attn_heads, mixer_heads=mixer_heads, position=position,
                                         beta_max=beta_max, mlp_mult_mixer=mlp_mult_mixer, mlp_mult_attn=mlp_mult_attn,
                                         max_position_embeddings=max_position_embeddings))

    def forward(self, input_ids: torch.LongTensor, return_logits: bool = False, return_dict: bool = True):
        h = self.model(input_ids, return_hidden=True)
        hidden = self.model.norm_f(h)
        logits = self.model.head(h) if return_logits else None
        if not return_dict:
            return (hidden, logits)
        return DeepaSequenceOutput(last_hidden_state=hidden, logits=logits)


__all__ = ["DeepaSequenceModel", "DeepaSequenceOutput"]
