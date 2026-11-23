"""DEEPA language model (plain PyTorch). Framework integrations wrap this module; they do not re-implement it."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import DeepaConfig
from .layers.attention import Attention
from .layers.common import MLP, RMSNorm
from .layers.sheaf_delta import SheafDeltaMixer


class Block(nn.Module):
    """Pre-norm residual block: x + mix(norm(x)), then + MLP(norm(.)). kind 'd' -> Sheaf-Delta + SwiGLU, 'a' -> attention + GELU."""

    def __init__(self, cfg: DeepaConfig, kind: str):
        super().__init__()
        self.kind = kind
        self.n1, self.n2 = RMSNorm(cfg.dim, cfg.norm_eps), RMSNorm(cfg.dim, cfg.norm_eps)
        if kind == "d":
            self.mix = SheafDeltaMixer(cfg)
            self.mlp = MLP(cfg.dim, cfg.mlp_mult_mixer, gated=True, hidden=cfg.mlp_hidden("d"))
        else:
            self.mix = Attention(cfg)
            self.mlp = MLP(cfg.dim, cfg.mlp_mult_attn, gated=cfg.attn_mlp == "swiglu", hidden=cfg.mlp_hidden("a"))

    def forward(self, x, state=None, return_state: bool = False):
        if return_state:
            y, state = self.mix(self.n1(x), state, return_state=True)
            x = x + y
            return x + self.mlp(self.n2(x)), state
        x = x + self.mix(self.n1(x), state)
        return x + self.mlp(self.n2(x))

    def step(self, x, state):
        """Single-token update. For attention blocks this is just forward() on a length-1 sequence with the KV cache."""
        if self.kind == "d":
            y, state = self.mix.step(self.n1(x), state)
            x = x + y
            return x + self.mlp(self.n2(x)), state
        y, state = self.forward(x[:, None], state, return_state=True)
        return y[:, 0], state


@dataclass
class DeepaCache:
    """Per-layer decode state: Sheaf-Delta layers hold (S, last values) of constant size; attention layers hold (K, V)."""
    states: List[Optional[tuple]] = field(default_factory=list)
    seen_tokens: int = 0

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for s in self.states if s is not None for t in s)


class DeepaLM(nn.Module):
    def __init__(self, cfg: DeepaConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg, c) for c in cfg.pattern])
        self.norm_f = RMSNorm(cfg.dim, cfg.norm_eps)
        if not cfg.tie_embeddings:
            self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.gradient_checkpointing = False
        self.apply(self._init)
        if cfg.scaled_init:  # residual-stream output projections: std / sqrt(2 * n_layers)  (GPT-2 / Megatron)
            for name, p in self.named_parameters():
                if name.endswith(("mix.out.weight", "mix.o.weight", "mlp.down.weight")):
                    nn.init.normal_(p, std=cfg.init_std / (2 * cfg.n_layers) ** 0.5)

    def _init(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=self.cfg.init_std)

    # ------------------------------------------------------------------ core
    def head(self, h: torch.Tensor) -> torch.Tensor:
        h = self.norm_f(h)
        return self.lm_head(h) if not self.cfg.tie_embeddings else F.linear(h, self.embed.weight)

    def forward(self, input_ids: torch.Tensor, cache: Optional[DeepaCache] = None, use_cache: bool = False,
                return_hidden: bool = False):
        """input_ids [B, T]. Returns logits [B, T, V] (and the updated cache if use_cache or a cache was passed).
        With a cache, input_ids continue the cached sequence (chunked prefill / multi-token decode)."""
        h = self.embed(input_ids)
        want_state = use_cache or cache is not None
        if want_state:
            cache = cache or DeepaCache(states=[None] * len(self.blocks))
            for i, blk in enumerate(self.blocks):
                h, cache.states[i] = blk(h, cache.states[i], return_state=True)
            cache.seen_tokens += input_ids.shape[1]
        else:
            for blk in self.blocks:
                h = checkpoint(blk, h, use_reentrant=False) if (self.gradient_checkpointing and self.training) else blk(h)
        out = h if return_hidden else self.head(h)
        return (out, cache) if want_state else out

    @torch.no_grad()
    def step(self, token: torch.Tensor, cache: DeepaCache) -> torch.Tensor:
        """One decode step. token [B] -> logits [B, V]. Sheaf-Delta layers cost O(1); attention layers O(T)."""
        h = self.embed(token)
        for i, blk in enumerate(self.blocks):
            h, cache.states[i] = blk.step(h, cache.states[i])
        cache.seen_tokens += 1
        return self.head(h)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 128, temperature: float = 1.0,
                 top_k: Optional[int] = None, top_p: Optional[float] = None, eos_token_id: Optional[int] = None,
                 generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Prefill the prompt in parallel, then decode token by token. temperature=0 -> greedy."""
        logits, cache = self.forward(input_ids, use_cache=True)
        nxt_logits, out = logits[:, -1], [input_ids]
        for _ in range(max_new_tokens):
            tok = _sample(nxt_logits.float(), temperature, top_k, top_p, generator)
            out.append(tok[:, None])
            if eos_token_id is not None and bool((tok == eos_token_id).all()):
                break
            nxt_logits = self.step(tok, cache)
        return torch.cat(out, 1)

    # ------------------------------------------------------------------ utilities
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def set_backend(self, delta: str = "auto", alibi: str = "auto") -> "DeepaLM":
        """delta in {auto, fla, chunked, recurrent}; alibi in {auto, folded, dense}."""
        for m in self.modules():
            if isinstance(m, SheafDeltaMixer):
                m.impl = delta
            elif isinstance(m, Attention):
                m.alibi_impl = alibi
        return self

    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({"model_type": "deepa", **self.cfg.to_dict()}, f, indent=2)
        sd = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        try:
            from safetensors.torch import save_file
            save_file(sd, os.path.join(path, "model.safetensors"))
        except ImportError:
            torch.save(sd, os.path.join(path, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, path: str, map_location: Union[str, torch.device] = "cpu") -> "DeepaLM":
        with open(os.path.join(path, "config.json")) as f:
            cfg = DeepaConfig.from_dict(json.load(f))
        model = cls(cfg)
        st = os.path.join(path, "model.safetensors")
        if os.path.exists(st):
            from safetensors.torch import load_file
            sd = load_file(st, device=str(map_location))
        else:
            sd = torch.load(os.path.join(path, "pytorch_model.bin"), map_location=map_location)
        model.load_state_dict(sd)
        return model


def _sample(logits: torch.Tensor, temperature: float, top_k, top_p, generator) -> torch.Tensor:
    if temperature == 0:
        return logits.argmax(-1)
    logits = logits / temperature
    if top_k:
        kth = torch.topk(logits, top_k, -1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        srt, idx = torch.sort(logits, -1, descending=True)
        drop = srt.softmax(-1).cumsum(-1) - srt.softmax(-1) > top_p
        logits = logits.scatter(-1, idx, srt.masked_fill(drop, float("-inf")))
    return torch.multinomial(logits.softmax(-1), 1, generator=generator)[:, 0]


__all__ = ["DeepaLM", "DeepaCache", "Block"]
