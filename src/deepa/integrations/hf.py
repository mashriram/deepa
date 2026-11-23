"""Hugging Face ``transformers`` integration.

    import deepa.integrations.hf                     # registers model_type "deepa" with the Auto classes
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = deepa.integrations.hf.DeepaHFConfig.from_preset("deepa_alibi")
    model = AutoModelForCausalLM.from_config(cfg)
    model.save_pretrained("out/"); AutoModelForCausalLM.from_pretrained("out/")
    model.generate(input_ids, max_new_tokens=50)     # standard HF generation, DEEPA recurrent cache under the hood

The HF model wraps ``deepa.DeepaLM`` (same parameters, prefixed ``model.``); nothing is re-implemented.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

try:
    from transformers import AutoConfig, AutoModelForCausalLM, GenerationMixin, PretrainedConfig, PreTrainedModel
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.tokenization_utils import PreTrainedTokenizer
except ImportError as e:  # pragma: no cover
    raise ImportError("deepa.integrations.hf requires `transformers` (pip install 'deepa-lm[hf]')") from e

from ..config import DeepaConfig
from ..model import DeepaCache, DeepaLM

_FIELDS = list(DeepaConfig.__dataclass_fields__)


class DeepaHFConfig(PretrainedConfig):
    model_type = "deepa"

    def __init__(self, **kwargs):
        own = {k: kwargs.pop(k) for k in list(kwargs) if k in _FIELDS}
        base = DeepaConfig(**own) if own else DeepaConfig()
        for k, v in base.to_dict().items():
            setattr(self, k, v)
        kwargs.setdefault("tie_word_embeddings", base.tie_embeddings)
        kwargs.setdefault("bos_token_id", None)
        kwargs.setdefault("eos_token_id", None)
        super().__init__(**kwargs)

    @property
    def hidden_size(self) -> int:
        return self.dim

    @property
    def num_hidden_layers(self) -> int:
        return self.n_layers

    def to_deepa(self) -> DeepaConfig:
        return DeepaConfig.from_dict({k: getattr(self, k) for k in _FIELDS})

    @classmethod
    def from_preset(cls, preset: str = "deepa_alibi", **kw) -> "DeepaHFConfig":
        return cls(**getattr(DeepaConfig, preset)(**kw).to_dict())


class DeepaForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = DeepaHFConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Block"]
    _tied_weights_keys = {}  # tying is internal to DeepaLM (head reuses embed.weight)
    _is_stateful = True       # recurrent state: no beam-search cache reordering

    def __init__(self, config: DeepaHFConfig):
        super().__init__(config)
        self.model = DeepaLM(config.to_deepa())
        self.post_init()

    def _init_weights(self, module):  # DeepaLM initialises itself; keep HF from re-initialising loaded weights
        pass

    def get_input_embeddings(self):
        return self.model.embed

    def set_input_embeddings(self, value):
        self.model.embed = value

    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func=None):
        self.model.gradient_checkpointing = enable

    def forward(self, input_ids: Optional[torch.LongTensor] = None, past_key_values: Optional[DeepaCache] = None,
                use_cache: Optional[bool] = None, labels: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None, return_dict: Optional[bool] = True, **kwargs):
        """attention_mask: only all-ones masks are supported (no left padding); pass equal-length or unpadded batches."""
        if attention_mask is not None and not bool(attention_mask.all()):
            raise ValueError("DEEPA's recurrent state does not support padded batches; use equal-length sequences.")
        if not isinstance(past_key_values, DeepaCache):  # HF may pre-create its own (empty) attention cache; ignore it
            past_key_values = None
        use_cache = bool(use_cache) or past_key_values is not None
        if use_cache:
            if past_key_values is not None and input_ids.shape[1] == 1 and past_key_values.seen_tokens > 0:
                logits = self.model.step(input_ids[:, 0], past_key_values)[:, None]
                cache = past_key_values
            else:
                logits, cache = self.model(input_ids, cache=past_key_values, use_cache=True)
        else:
            logits, cache = self.model(input_ids), None
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), labels[:, 1:].reshape(-1), ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=cache)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, use_cache=True, **kwargs):
        if not isinstance(past_key_values, DeepaCache):
            past_key_values = None
        if past_key_values is not None and past_key_values.seen_tokens > 0:
            input_ids = input_ids[:, past_key_values.seen_tokens:]
        return {"input_ids": input_ids, "past_key_values": past_key_values, "use_cache": use_cache}

    def _reorder_cache(self, past_key_values, beam_idx):  # beam search: reorder every state tensor along batch
        past_key_values.states = [tuple(t.index_select(0, beam_idx) for t in s) if s is not None else None
                                  for s in past_key_values.states]
        return past_key_values


class DeepaByteTokenizer(PreTrainedTokenizer):
    """256-symbol byte tokenizer (id == byte). No special tokens by default."""
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def vocab_size(self) -> int:
        return 256

    def get_vocab(self) -> Dict[str, int]:
        return {chr(i) if 32 < i < 127 else f"<0x{i:02X}>": i for i in range(256)}

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return [f"<0x{b:02X}>" for b in text.encode("utf-8")]

    def _convert_token_to_id(self, token: str) -> int:
        return int(token[3:5], 16) if token.startswith("<0x") else ord(token)

    def _convert_id_to_token(self, index: int) -> str:
        return f"<0x{index:02X}>"

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return bytes(self._convert_token_to_id(t) for t in tokens).decode("utf-8", errors="replace")

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        return ()


def register() -> None:
    """Register DEEPA with the transformers Auto classes (idempotent; called on import)."""
    try:
        AutoConfig.register("deepa", DeepaHFConfig)
    except ValueError:
        pass
    try:
        AutoModelForCausalLM.register(DeepaHFConfig, DeepaForCausalLM)
    except ValueError:
        pass


register()

__all__ = ["DeepaHFConfig", "DeepaForCausalLM", "DeepaByteTokenizer", "register"]
