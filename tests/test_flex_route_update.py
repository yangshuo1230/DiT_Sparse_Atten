import pytest
import torch

from attention_backends.flex import FlexReuseBackend, FlexReuseConfig


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FlexAttention route updates require CUDA",
)
def test_periodic_route_update_replaces_block_mask(monkeypatch):
    import attention_backends.flex as flex_module

    monkeypatch.setattr(
        flex_module, "routing_context",
        lambda include_grid=False: (0, 1, 0, None)
        if include_grid else (0, 1, 0))

    tokens, heads, head_dim = 256, 2, 32
    q = torch.randn(1, tokens, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    backend = FlexReuseBackend(FlexReuseConfig(
        block_size=128, keep=.5, mass_target=.9,
        compile_kernel=False, update_interval=2))
    backend.state[(0, 0, tokens, heads, head_dim, q.device, None)] = {
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


def test_sampled_update_can_switch_route_to_dense(monkeypatch):
    import attention_backends.flex as flex_module

    monkeypatch.setattr(
        flex_module, "routing_context",
        lambda include_grid=False: (0, 1, 0, None)
        if include_grid else (0, 1, 0))
    monkeypatch.setattr(flex_module, "_build_block_mask", lambda *args: object())
    tokens, heads, head_dim = 8, 1, 4
    q = torch.randn(1, tokens, heads, head_dim)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    backend = FlexReuseBackend(FlexReuseConfig(
        sampled_update_interval=1, prefetch_sampled_update=False,
        dense_route_threshold=.75, total_steps=3))
    backend.state[(0, 0, tokens, heads, head_dim, q.device, None)] = {
        "route": torch.eye(1, dtype=torch.bool),
        "block_mask": object(), "step": 0, "dense": False,
    }
    monkeypatch.setattr(
        backend, "_sampled_route",
        lambda state, query, key, scale: {
            "route": torch.ones(1, 1, 1, dtype=torch.bool),
            "frontier": None,
        })

    output = backend(q, k, v, dtype=torch.float32)
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
    ).transpose(1, 2)
    torch.testing.assert_close(output, reference)
    assert backend.state[next(iter(backend.state))]["dense"] is True
    assert backend.phase_counts["dense_dispatch"] == 1


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Spatial Flex bootstrap requires CUDA",
)
def test_spatial_bootstrap_restores_original_token_order(monkeypatch):
    import attention_backends.flex as flex_module

    monkeypatch.setattr(
        flex_module, "routing_context",
        lambda include_grid=False: (0, 0, 0, (1, 4, 64))
        if include_grid else (0, 0, 0))
    torch.manual_seed(8)
    q = torch.randn(1, 256, 2, 32, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    backend = FlexReuseBackend(FlexReuseConfig(
        bootstrap_mode="sampled", prefetch_sampled_bootstrap=False,
        spatial_reorder=True, spatial_microtile_tokens=32,
        route_samples=2, total_steps=2, compile_kernel=False))

    output = backend(q, k, v)
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
    ).transpose(1, 2)
    torch.testing.assert_close(output, reference, atol=2e-3, rtol=2e-2)
    state = next(iter(backend.state.values()))
    assert state["layout"].microtile_shape == (4, 8)
    assert state["frontier"] is not None
