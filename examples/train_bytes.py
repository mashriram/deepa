"""Minimal training loop for a byte-level DEEPA model on any file (the paper's recipe, scaled down).

    python examples/train_bytes.py path/to/text_or_binary_file --preset deepa_alibi --steps 1500
"""
import argparse, math, time

import torch
import torch.nn.functional as F

from deepa import DeepaConfig, DeepaLM

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("--preset", default="deepa_alibi", choices=["deepa_alibi", "deepa_rope", "sheafdelta"])
ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--ctx", type=int, default=1024); ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--compile", action="store_true"); ap.add_argument("--out", default="deepa-ckpt")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
raw = torch.frombuffer(bytearray(open(a.data, "rb").read()), dtype=torch.uint8).long()
n_val = max(len(raw) // 40, a.ctx + 1); train, val = raw[:-n_val], raw[-n_val:]
model = DeepaLM(getattr(DeepaConfig, a.preset)()).to(dev)
fwd = torch.compile(model) if a.compile else model
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.05)
print(f"{a.preset}: {model.num_parameters():,} params on {dev}")

def batch(src):
    i = torch.randint(0, len(src) - a.ctx - 1, (a.batch,))
    x = torch.stack([src[j:j + a.ctx] for j in i]); return x.to(dev), torch.stack([src[j + 1:j + a.ctx + 1] for j in i]).to(dev)

t0 = time.time()
for s in range(a.steps):
    lr = a.lr * min(1, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / a.steps)))
    for g in opt.param_groups: g["lr"] = lr
    x, y = batch(train)
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        loss = F.cross_entropy(fwd(x).flatten(0, 1).float(), y.flatten())
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if (s + 1) % 100 == 0:
        print(f"step {s+1}  loss {loss.item() / math.log(2):.3f} bits/byte  {time.time() - t0:.0f}s", flush=True)

model.eval()
with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
    x, y = batch(val); print("val bits/byte:", F.cross_entropy(model(x).flatten(0, 1).float(), y.flatten()).item() / math.log(2))
model.save_pretrained(a.out); print("saved to", a.out)
