import tempfile

import pytest
import torch

transformers = pytest.importorskip("transformers")


def _hf_cfg(preset):
    import deepa.integrations.hf as H
    kw = dict(dim=64, n_layers=4, attn_heads=4, mixer_heads=2)
    if preset != "sheafdelta":
        kw["pattern"] = "ddda"
    return H.DeepaHFConfig.from_preset(preset, **kw)


@pytest.mark.parametrize("preset", ["deepa_alibi", "deepa_rope", "sheafdelta"])
def test_hf_generate_save_load_train(preset):
    from transformers import AutoModelForCausalLM
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(_hf_cfg(preset)).eval()
    ids = torch.randint(0, 256, (2, 12))
    with torch.no_grad():
        full = m(ids).logits
        gen = m.generate(ids, max_new_tokens=8, do_sample=False)
        ref = ids
        for _ in range(8):
            ref = torch.cat([ref, m(ref).logits[:, -1].argmax(-1, keepdim=True)], 1)
    assert torch.equal(gen, ref)
    with tempfile.TemporaryDirectory() as d:
        m.save_pretrained(d)
        m2 = AutoModelForCausalLM.from_pretrained(d).eval()
        with torch.no_grad():
            assert torch.equal(m2(ids).logits, full)
    m.train()
    out = m(ids, labels=ids)
    out.loss.backward()
    assert torch.isfinite(out.loss)


def test_hf_tokenizer_roundtrip():
    import deepa.integrations.hf as H
    tok = H.DeepaByteTokenizer()
    text = "héllo, 世界"
    ids = tok(text)["input_ids"]
    assert ids == list(text.encode("utf-8")) and tok.decode(ids) == text


def test_diffusers_model():
    pytest.importorskip("diffusers")
    from deepa.integrations.diffusers import DeepaSequenceModel
    m = DeepaSequenceModel(dim=64, n_layers=4, pattern="ddda", attn_heads=4, mixer_heads=2).eval()
    ids = torch.randint(0, 256, (2, 10))
    with torch.no_grad():
        h = m(ids).last_hidden_state
    assert h.shape == (2, 10, 64)
    with tempfile.TemporaryDirectory() as d:
        m.save_pretrained(d)
        m2 = DeepaSequenceModel.from_pretrained(d).eval()
        with torch.no_grad():
            assert torch.equal(m2(ids).last_hidden_state, h)
