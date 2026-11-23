# deepa: hybrid delta-rule / attention language models

`deepa` is a clean, tested implementation of the **DEEPA** architecture (*Dyadic-Delta Enhanced-sheaf, Efficient
Pairing with ALiBi*). DEEPA is a hybrid sequence model. Most of its layers are **Sheaf-Delta** layers, which keep a fixed-size
memory instead of a growing KV cache, and a few layers are ordinary attention. The library ships three models:

| Preset | Layer pattern (d = Sheaf-Delta, a = attention) | Decode memory per sequence | Use it when |
|---|---|---|---|
| `DeepaConfig.deepa_alibi()` (**paper model**) | `dddaddda` | ¼ of a Transformer's KV cache | default; length-extrapolates (trained 1k → tested 16k) |
| `DeepaConfig.deepa_rope()` | `dddaddda` | ¼ of a Transformer's KV cache | 2-D raster data (images); does **not** extrapolate as well |
| `DeepaConfig.sheafdelta()` | `dddddddd` | **constant** (0.49 MB at 14M params, any length) | memory-bound / streaming; no KV cache at all |

All three load the checkpoints of the original research code and reproduce its outputs (see *Verification*).

## DEEPA in plain English

**What a language model does.** It reads text one piece at a time and guesses what comes next. To guess well it has to
remember what it has already read.

**How a standard Transformer remembers.** Think of a student who copies every word into a stack of notes. To write
the next word, the student re-reads the whole stack. Nothing gets lost, but the stack grows with every word. After a
long conversation the notes fill the computer's memory, each new word takes longer, and fewer conversations fit on one
machine at once. That growing stack is called the *KV cache*.

**What DEEPA does instead.** Most of DEEPA's layers (6 of every 8) keep **one fixed-size notebook** instead of a growing
stack. Each layer does three things.

1. **It writes and erases.** For each new piece of text, the layer files a note under a "label". If a note with that
   label already exists, it erases the old one before writing the new one. So it overwrites stale facts instead of
   piling them up. This is the *delta rule*. The notebook never grows; old notes slowly fade unless they keep getting
   refreshed.
2. **It can flip, not just fade.** The notebook can turn a note into its exact opposite, like flipping a light switch.
   This lets DEEPA keep track of things like "is this an odd or even number of brackets so far?" or "after this
   sequence of swaps, which cup is the ball under?". Standard Transformers of the same size could not learn these
   tasks in our tests. When we removed the flipping ability, DEEPA lost the skill too. That experiment is how we know
   this feature is the reason.
3. **It glances at the last few letters.** Alongside the notebook, the layer looks back exactly 1, 2, 4, 8 and 16
   letters. Spelling, multi-byte characters, code indentation and DNA patterns live in that short range.

**Why it still has some attention.** A fixed notebook is great for the gist, but it can't hold every exact word of a
long passage. So 2 of every 8 layers are ordinary Transformer layers that can re-read the text word for word, for
example to copy a name or a line of code exactly. They use a rule called *ALiBi*: nearby words count more than distant
ones. That rule lets the model read texts far longer than it was trained on.

**What this buys you** (measured on models with 2–38 million parameters):

- **Better text prediction than a same-size Transformer** on most data we tried: English Wikipedia text, Python code,
  Hindi, DNA and speech audio (Chinese text was the exception). It also stays ahead when both models get the same
  training time. It also judged the sentiment of long movie reviews far more accurately (76% vs 64%).
- **A quarter of the memory while writing.** Only 2 of 8 layers keep a growing stack, so one GPU can serve about
  **4× as many conversations at once** and generate text faster on long inputs.
- **Reads 16× longer texts than it was trained on** without getting confused. A standard RoPE Transformer breaks down
  as soon as the text is longer than what it saw in training.
- **Pure Sheaf-Delta** (no attention layers at all) uses the *same* tiny amount of memory whether the text is 100
  words or 100,000 words long.

**Where it is weaker** (we report these openly):

- **Short texts train slower.** On texts of about 1,000 letters, training is roughly 26% slower than an equally
  optimised Transformer, because many small computations are needed. It becomes *faster* than the Transformer from
  about 4,000 letters onward.
- **Copying long exact text.** Pure attention models learn to copy text word for word faster, and they are better at
  copying passages longer than anything they saw in training.
- **Images stored row by row.** The pixel directly above another sits far back in the sequence, and ALiBi
  under-weights distant positions. Swapping ALiBi for RoPE inside DEEPA fixes this, at the cost of length
  extrapolation.
- **Only tested at small scale.** All results come from models of up to 38 million parameters trained on one 8 GB GPU.
  Whether the advantages hold at billions of parameters is not yet known.

## Install

```bash
pip install deepa-lm               # from PyPI (once released); or, from a clone:
pip install -e .                    # core: PyTorch only (runs on CPU or GPU)
pip install -e ".[fast]"            # + Triton gated-delta-rule kernels (CUDA): the fast training path
pip install -e ".[hf]"              # + Hugging Face transformers integration
pip install -e ".[diffusers]"       # + diffusers integration
pip install -e ".[dev]"             # tests
```

## Quick start (plain PyTorch)

```python
import torch
from deepa import DeepaConfig, DeepaLM, ByteTokenizer

model = DeepaLM(DeepaConfig.deepa_alibi()).cuda()        # 14.44M params, byte-level (vocab 256)
tok = ByteTokenizer()

x = tok("The quick brown fox").cuda()                    # [1, T] byte ids
logits = model(x)                                         # [1, T, 256]
loss = torch.nn.functional.cross_entropy(logits[:, :-1].flatten(0, 1), x[:, 1:].flatten())

out = model.generate(x, max_new_tokens=100, temperature=0.8, top_k=50)
print(tok.decode(out[0]))
```

Generation prefills the prompt in parallel, then decodes one token at a time. Sheaf-Delta layers update a fixed-size
state, so each step costs the same at token 100 as at token 100,000. Only the attention layers grow with context.

### Use the mixer inside your own model

```python
from deepa import SheafDeltaMixer
mixer = SheafDeltaMixer.standalone(dim=512, heads=8)     # causal token mixer, [B, T, 512] -> [B, T, 512]
y = mixer(x)                                             # parallel (training)
y, state = mixer(x, return_state=True)                   # prefill and keep the state
y_t, state = mixer.step(x_t, state)                      # O(1) per token afterwards
```

### Sizes from the paper's scaling study

```python
DeepaConfig.from_size("38m")                  # 2m, 6m, 14m, 38m (ALiBi hybrid)
DeepaConfig.from_size("14m", kind="sheafdelta")
```

## Hugging Face transformers

```python
import deepa.integrations.hf as dhf               # registers model_type "deepa" with the Auto classes
from transformers import AutoModelForCausalLM

cfg = dhf.DeepaHFConfig.from_preset("deepa_alibi")
model = AutoModelForCausalLM.from_config(cfg)
model.generate(input_ids, max_new_tokens=50)       # standard HF generation; DEEPA's recurrent cache under the hood
model.save_pretrained("my-deepa"); AutoModelForCausalLM.from_pretrained("my-deepa")
tokenizer = dhf.DeepaByteTokenizer()
```

It works with the `Trainer` and passes `labels` through to the loss. Batches must be unpadded or equal-length, because
the recurrent state has no notion of padding.

## diffusers

```python
from deepa.integrations.diffusers import DeepaSequenceModel
encoder = DeepaSequenceModel(pattern="dddaddda")
cond = encoder(input_ids).last_hidden_state         # e.g. encoder_hidden_states for cross-attention
encoder.save_pretrained("enc"); DeepaSequenceModel.from_pretrained("enc")
```

DEEPA is causal and was evaluated as a language model. Using it as a text encoder or causal backbone is supported.
Using it as a bidirectional diffusion denoiser has not been tested.

## Loading research checkpoints

```python
from deepa.convert import from_research_checkpoint
model = from_research_checkpoint("checkpoints/extrap16k/alibi-hybrid-h6_seed0.pt", kind="alibi-hybrid-h6")
```

## Backends

`model.set_backend(delta=..., alibi=...)` chooses the implementation for each part of the model:

| Component | Options | Default (`auto`) |
|---|---|---|
| Gated delta rule | `fla` (Triton, CUDA), `chunked` (pure PyTorch, exact), `recurrent` (reference loop) | `fla` on CUDA if installed, otherwise `chunked` |
| ALiBi attention | `folded` (exact ALiBi inside the fused SDPA/FlashAttention kernel, O(T) memory), `dense` (explicit bias) | `folded` for bf16/fp16 CUDA prefill, otherwise `dense` |

Note on β: the delta-rule step size is passed already squashed (`b = 2·sigmoid(·) ∈ (0, 2)`), so it can produce negative
eigenvalues. fla's `allow_neg_eigval` flag only applies to its in-kernel sigmoid and has no effect here.

## Architecture (exact definition)

For a Sheaf-Delta layer, with input `u` after RMSNorm, per head:

```
q, k, v   = W_qkv u                    q, k L2-normalised
b_t       = 2 · σ(·)          ∈ (0, 2)   step size (b > 1 gives a negative eigenvalue along k)
a_t       = 0.8 + 0.2 · σ(·)  ∈ (0.8, 1) decay (bias init 3.0 → a ≈ 0.9905)
S_t       = a_t · S_{t-1} (I − b_t k_t k_tᵀ) + b_t v_t k_tᵀ        o_t = d_h^{-1/2} · S_t q_t
dyadic_t  = (1/√5) Σ_{d∈{1,2,4,8,16}} σ(W_e u)_{t,d} · U_d v_{t−d}    U_d = 2 Householder reflections (fixed per lag)
y_t       = W_out [ (RMSNorm_head(o_t) + dyadic_t) ⊙ SiLU(W_g u_t) ]
```

Each Sheaf-Delta block is followed by a SwiGLU MLP (multiplier 3.65). Each attention block uses 8 heads with ALiBi or
RoPE, followed by a GELU MLP (multiplier 3.65). The embeddings are tied and there is no positional embedding.

## Verification

`pytest` runs the following checks:

- **Kernels:** chunked and Triton (`fla`) delta rule match the token-by-token reference, including b ∈ (1, 2) and a non-zero initial state.
- **Mixer vs research code:** the forward pass, step-by-step decode and chunked prefill continuation all match the research implementation to fp32 precision.
- **Generation:** prefill plus O(1) decode gives the same logits as full recompute, for all three presets.
- **ALiBi:** the folded kernel matches the dense-bias reference.
- **Integrations:** Hugging Face `generate()` matches greedy full recompute; `save_pretrained`/`from_pretrained` is bit-exact; the diffusers save/load round-trip works.
- **Research checkpoints (GPU, if present):** logits match `hsnlm`.

## Reproducing the paper's numbers

The benchmarks, raw logs and per-benchmark write-ups live in the research repository (`hsnlm/benchmarks`,
`hsnlm/benchmark_results`). This library is the clean, importable implementation of the architecture they evaluate.

## License

Apache License 2.0. See [LICENSE](LICENSE).
