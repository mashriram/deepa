import tempfile

import pytest
import torch

from deepa import DeepaConfig, DeepaLM, SheafDeltaMixer
from deepa.layers.attention import alibi_attention_dense, alibi_attention_folded

PRESETS = ["deepa_alibi", "deepa_rope", "sheafdelta"]


def small(preset, **kw):
    base = dict(dim=64, n_layers=4, attn_heads=4, mixer_heads=2, **kw)
    if preset != "sheafdelta":
        base["pattern"] = "ddda"
    return getattr(DeepaConfig, preset)(**base)


def test_paper_parameter_counts():
    assert DeepaLM(DeepaConfig.deepa_alibi()).num_parameters() == 14_436_348
    assert DeepaLM(DeepaConfig.sheafdelta()).num_parameters() == 14_771_408


@pytest.mark.parametrize("preset", PRESETS)
def test_decode_matches_full_forward(preset):
    torch.manual_seed(0)
    m = DeepaLM(small(preset)).double().eval()
    x = torch.randint(0, 256, (2, 30))
    with torch.no_grad():
        full = m(x)
        logits, cache = m(x[:, :12], use_cache=True)            # parallel prefill
        steps = [logits]
        for t in range(12, 30):
            steps.append(m.step(x[:, t], cache)[:, None])        # O(1) recurrent decode
    dec = torch.cat(steps, 1)                                     # logits at every position 0..29
    assert dec.shape == full.shape and (dec - full).abs().max() < 1e-6


@pytest.mark.parametrize("preset", PRESETS)
def test_chunked_prefill_matches_full(preset):
    torch.manual_seed(0)
    m = DeepaLM(small(preset)).double().eval()
    x = torch.randint(0, 256, (2, 40))
    with torch.no_grad():
        full = m(x)
        a, cache = m(x[:, :17], use_cache=True)
        b, cache = m(x[:, 17:], cache=cache)
    assert (torch.cat([a, b], 1) - full).abs().max() < 1e-6
    assert cache.seen_tokens == 40


def test_sheafdelta_state_is_constant_size():
    m = DeepaLM(small("sheafdelta")).eval()
    with torch.no_grad():
        _, c1 = m(torch.randint(0, 256, (1, 10)), use_cache=True)
        _, c2 = m(torch.randint(0, 256, (1, 500)), use_cache=True)
    assert c1.nbytes() == c2.nbytes()


def test_alibi_folded_matches_dense():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 8, 300, 48) for _ in range(3))
    ref = alibi_attention_dense(q, k, v)
    out = alibi_attention_folded(q, k, v)                       # fp32 here: exact up to fp rounding
    assert (out - ref).abs().max() < 1e-4


def test_generate_greedy_matches_recompute():
    torch.manual_seed(0)
    m = DeepaLM(small("deepa_alibi")).eval()
    x = torch.randint(0, 256, (2, 8))
    out = m.generate(x, max_new_tokens=10, temperature=0)
    ref = x
    with torch.no_grad():
        for _ in range(10):
            ref = torch.cat([ref, m(ref)[:, -1].argmax(-1, keepdim=True)], 1)
    assert torch.equal(out, ref)


def test_save_load_roundtrip():
    m = DeepaLM(small("deepa_rope")).eval()
    x = torch.randint(0, 256, (1, 16))
    with tempfile.TemporaryDirectory() as d:
        m.save_pretrained(d)
        m2 = DeepaLM.from_pretrained(d).eval()
    with torch.no_grad():
        assert torch.equal(m(x), m2(x))


def test_backward_and_standalone_mixer():
    m = DeepaLM(small("deepa_alibi"))
    x = torch.randint(0, 256, (2, 20))
    m(x).logsumexp(-1).mean().backward()
    assert all(p.grad is not None for p in m.parameters())
    mix = SheafDeltaMixer.standalone(32, 4)
    y, st = mix(torch.randn(2, 9, 32), return_state=True)
    y1, st = mix.step(torch.randn(2, 32), st)
    assert y.shape == (2, 9, 32) and y1.shape == (2, 32)


def test_config_validation():
    with pytest.raises(ValueError):
        DeepaConfig(pattern="dda", n_layers=4)
    with pytest.raises(ValueError):
        DeepaConfig(position="nope")
