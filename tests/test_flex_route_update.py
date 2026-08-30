import pytest
import torch

from attention_backends.flex import FlexReuseBackend, FlexReuseConfig


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FlexAttention route updates require CUDA",
)
def test_periodic_route_update_replaces_block_mask():
    import types
    import sys

    model = types.ModuleType("wan.modules.model")
    model._CURRENT_ATTN_ID = 0
    model._CURRENT_DENOISE_STEP = 1
    model._CURRENT_CFG_BRANCH = 0
    sys.modules["wan.modules.model"] = model

    tokens, heads, head_dim = 256, 2, 32
    q = torch.randn(1, tokens, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    backend = FlexReuseBackend(FlexReuseConfig(
        block_size=128, keep=.5, mass_target=.9,
        compile_kernel=False, update_interval=2))
    backend.state[(0, 0, tokens, heads, head_dim, q.device)] = {
        "route": torch.zeros(heads, 2, 2, device="cuda", dtype=torch.bool),
        "block_mask": object(),
        "step": -1,
    }

    output = backend(q, k, v)
    assert output.shape == q.shape[:3] + (head_dim,)
    state = next(iter(backend.state.values()))
    assert state["step"] == 1
    assert state["route"].any()
    assert state["block_mask"] is not None
