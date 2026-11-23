"""Byte-level tokenizer (vocab 256): token id == byte value. Lossless for any UTF-8 text or raw binary data."""
from __future__ import annotations

from typing import List, Sequence, Union

import torch


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text: Union[str, bytes]) -> List[int]:
        return list(text.encode("utf-8") if isinstance(text, str) else bytes(text))

    def decode(self, ids: Union[Sequence[int], torch.Tensor], errors: str = "replace") -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return bytes(int(i) for i in ids).decode("utf-8", errors=errors)

    def __call__(self, text: Union[str, bytes], return_tensors: str = "pt") -> torch.Tensor:
        return torch.tensor([self.encode(text)], dtype=torch.long)


__all__ = ["ByteTokenizer"]
