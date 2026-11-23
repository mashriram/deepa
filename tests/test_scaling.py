"""Any-size models: builder, size solver, exact analytic parameter counts, GQA, meta-device construction."""
import pytest
import torch

from deepa import DeepaConfig, DeepaLM
from deepa.config import NAMED_SIZES, pattern_from_layers, spread_pattern

SMALL_SHAPES = [
    dict(dim=128, n_layers=4, attn_every=4, head_dim=32, vocab_size=1000),
    dict(dim=256, n_layers=6, attn_every=3, head_dim=64, vocab_size=50257, gqa_groups=2),
    dict(dim=192, n_layers=5, attn_layers=[1, 4], head_dim=32, mixer_head_dim=64, vocab_size=300, position="rope"),
    dict(dim=128, n_layers=4, attn_every=0, head_dim=32, vocab_size=500),                         # pure Sheaf-Delta
    dict(dim=128, n_layers=4, attn_every=2, head_dim=32, vocab_size=500, mlp_hidden_mixer=200, mlp_hidden_attn=300,
         tie_embeddings=False),
]


@pytest.mark.parametrize("kw", SMALL_SHAPES)
def test_analytic_param_count_is_exact(kw):
    kw = dict(kw)
    cfg = DeepaConfig.build(kw.pop("dim"), kw.pop("n_layers"), **kw)
    assert cfg.num_parameters() == DeepaLM(cfg).num_parameters()


def test_paper_presets_unchanged():
    for cfg, n in ((DeepaConfig.deepa_alibi(), 14_436_348), (DeepaConfig.sheafdelta(), 14_771_408)):
        assert cfg.num_parameters() == n == DeepaLM(cfg).num_parameters()


@pytest.mark.parametrize("name", ["125m", "350m", "1b", "2b", "7b", "13b", "70b", "200b"])
def test_from_params_hits_target(name):
    cfg = DeepaConfig.from_params(name)
    assert abs(cfg.num_parameters() / NAMED_SIZES[name] - 1) < 0.12, cfg.summary()
    assert cfg.attn_heads % cfg.kv_heads == 0 and cfg.dim % cfg.mixer_heads == 0


def test_from_params_accepts_numbers_and_strings():
    a, b = DeepaConfig.from_params(2e9), DeepaConfig.from_params("2B")
    assert a.to_dict() == b.to_dict()


def test_200b_builds_on_meta_device_without_memory():
    cfg = DeepaConfig.from_params("200b")
    with torch.device("meta"):
        model = DeepaLM(cfg)
    assert model.num_parameters() == cfg.num_parameters()
    assert all(p.device.type == "meta" for p in model.parameters())


@pytest.mark.parametrize("position", ["alibi", "rope"])
def test_gqa_decode_matches_full_forward(position):
    torch.manual_seed(0)
    cfg = DeepaConfig.build(128, 4, attn_every=2, head_dim=32, gqa_groups=4, vocab_size=300, position=position)
    assert cfg.kv_heads == 1 and cfg.attn_heads == 4
    m = DeepaLM(cfg).double().eval()
    x = torch.randint(0, 300, (2, 25))
    with torch.no_grad():
        full = m(x)
        logits, cache = m(x[:, :10], use_cache=True)
        steps = [logits] + [m.step(x[:, t], cache)[:, None] for t in range(10, 25)]
    assert (torch.cat(steps, 1) - full).abs().max() < 1e-6
    k, v = cache.states[1]                                        # attention layer: only kv heads are cached
    assert k.shape[1] == 1


def test_scaled_init_shrinks_residual_projections():
    cfg = DeepaConfig.build(256, 12, head_dim=64, vocab_size=300)
    m = DeepaLM(cfg)
    down = torch.cat([b.mlp.down.weight.flatten() for b in m.blocks]).std().item()
    up = torch.cat([b.mlp.up.weight.flatten() for b in m.blocks]).std().item()
    assert down < up / 3


def test_patterns():
    assert spread_pattern(8, 4) == "dddaddda" and spread_pattern(6, 0) == "dddddd"
    assert pattern_from_layers(5, [0, 3]) == "addad"
    cfg = DeepaConfig.build(128, 8, attn_layers=[2, 5], head_dim=32)
    assert cfg.pattern == "ddaddadd"


def test_memory_estimate_reports_hybrid_saving():
    cfg = DeepaConfig.from_params("7b")
    est = cfg.estimate_memory(seq_len=32768, batch=8)
    assert est["decode_state_gb"] < est["transformer_kv_cache_gb_same_shape"] / 3
