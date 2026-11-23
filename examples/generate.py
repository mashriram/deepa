"""Generate text from a saved DEEPA model (or a randomly initialised one for a smoke test).

    python examples/generate.py --model deepa-ckpt --prompt "In 1905, " --tokens 200
"""
import argparse

import torch

from deepa import ByteTokenizer, DeepaConfig, DeepaLM

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=None); ap.add_argument("--prompt", default="Hello")
ap.add_argument("--tokens", type=int, default=200); ap.add_argument("--temperature", type=float, default=0.8)
ap.add_argument("--top_k", type=int, default=50)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = (DeepaLM.from_pretrained(a.model) if a.model else DeepaLM(DeepaConfig.deepa_alibi())).to(dev).eval()
tok = ByteTokenizer()
out = model.generate(tok(a.prompt).to(dev), max_new_tokens=a.tokens, temperature=a.temperature, top_k=a.top_k)
print(tok.decode(out[0]))
