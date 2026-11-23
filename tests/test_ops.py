import pytest
import torch
import torch.nn.functional as F

from deepa.ops.delta_rule import fla_available, gated_delta_rule, gated_delta_rule_chunked, gated_delta_rule_recurrent


def _inputs(B=2, T=77, H=3, d=16, device="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)
    q = F.normalize(torch.randn(B, T, H, d, generator=g), dim=-1)
    k = F.normalize(torch.randn(B, T, H, d, generator=g), dim=-1)
    v = torch.randn(B, T, H, d, generator=g)
    a = 0.8 + 0.2 * torch.rand(B, T, H, generator=g)
    b = 2 * torch.rand(B, T, H, generator=g)          # includes b in (1, 2): negative eigenvalues
    S0 = torch.randn(B, H, d, d, generator=g)
    return [t.to(device) for t in (q, k, v, a, b, S0)]


@pytest.mark.parametrize("chunk", [8, 16, 32])
def test_chunked_matches_recurrent(chunk):
    q, k, v, a, b, S0 = _inputs()
    o1, s1 = gated_delta_rule_recurrent(q, k, v, a, b, initial_state=S0)
    o2, s2 = gated_delta_rule_chunked(q, k, v, a, b, initial_state=S0, chunk=chunk)
    assert torch.allclose(o1, o2, atol=1e-4) and torch.allclose(s1, s2, atol=1e-4)


def test_negative_eigenvalue_reflects():
    """b = 2 with k = e_0 and a = 1 must flip the sign of S's first row (Householder reflection) before the write."""
    S0 = torch.eye(4)[None, None]
    k = torch.zeros(1, 1, 1, 4); k[..., 0] = 1
    v = torch.zeros(1, 1, 1, 4)
    _, S = gated_delta_rule_recurrent(k, k, v, torch.ones(1, 1, 1), 2 * torch.ones(1, 1, 1), initial_state=S0)
    assert torch.allclose(S[0, 0, 0], -torch.eye(4)[0]) and torch.allclose(S[0, 0, 1:], torch.eye(4)[1:])


@pytest.mark.skipif(not fla_available(), reason="needs CUDA + flash-linear-attention")
def test_fla_matches_reference_with_beta_up_to_2():
    q, k, v, a, b, S0 = _inputs(T=200, H=4, d=64, device="cuda")
    o1, s1 = gated_delta_rule_recurrent(q, k, v, a, b, initial_state=S0)
    o2, s2 = gated_delta_rule(q, k, v, a, b, initial_state=S0, output_final_state=True, impl="fla")
    assert (o1 - o2.float()).abs().max() < 2e-3 * o1.abs().max()
    assert (s1 - s2.float()).abs().max() < 2e-3 * s1.abs().max()
