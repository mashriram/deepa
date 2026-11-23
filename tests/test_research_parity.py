"""End-to-end parity with the research implementation (hsnlm) on its trained checkpoints.

Runs only when CUDA is available and the research repo is reachable (env DEEPA_RESEARCH_DIR, default ../hsnlm).
"""
import os
import sys

import pytest
import torch

RESEARCH = os.path.abspath(os.environ.get("DEEPA_RESEARCH_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "hsnlm")))
CKPT = os.path.join(RESEARCH, "checkpoints", "extrap16k")

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and os.path.isdir(CKPT)),
                                reason="needs CUDA and the research repo's checkpoints/extrap16k")


@pytest.mark.parametrize("kind", ["alibi-hybrid-h6"])
@pytest.mark.parametrize("alibi_impl", ["dense", "flash"])
def test_logits_match_research_model(kind, alibi_impl):
    sys.path.insert(0, RESEARCH)
    import benchmarks.deepa_common as C
    from deepa.convert import from_research_checkpoint

    C.ALIBI_IMPL = alibi_impl
    path = os.path.join(CKPT, f"{kind}_seed0.pt")
    torch.manual_seed(0)
    ref = C.build(kind).eval()
    ref.load_state_dict(torch.load(path, map_location="cuda"))
    mine = from_research_checkpoint(path, kind=kind, map_location="cuda").cuda().eval()
    mine.set_backend(delta="fla", alibi="folded" if alibi_impl == "flash" else "dense")
    x = torch.randint(0, 256, (2, 1024), device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        a, b = ref(x).float(), mine(x).float()
    # identical math, identical kernels; differences are bf16 kernel-order noise only
    assert (a - b).abs().max() < 2e-2 * a.abs().max()
    ce = lambda lg: torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1)).item()  # noqa: E731
    assert abs(ce(a) - ce(b)) < 1e-3
