import pytest
import torch

from attention_backends.flex import (
    _build_block_mask,
    _dense_bootstrap,
    _flex_output,
    _top_mass_route,
    flex_attention,
)


def test_top_mass_route_respects_row_cap():
    mass = torch.tensor([[[0.50, 0.30, 0.15, 0.05]]])
    route = _top_mass_route(mass, target=0.90, keep=0.50)
    assert route.tolist() == [[[True, True, False, False]]]


@pytest.mark.skipif(
    not torch.cuda.is_available() or flex_attention is None,
    reason="FlexAttention CUDA support is required",
)
def test_dense_bootstrap_matches_sdpa_with_partial_last_block():
    torch.manual_seed(0)
    q = torch.randn(1, 250, 2, 32, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    output, route = _dense_bootstrap(
        q, k, v, block_size=128, query_chunk=256,
        mass_target=0.90, keep=0.625)
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
    ).transpose(1, 2)

    assert route.shape == (2, 2, 2)
    torch.testing.assert_close(output, reference, atol=2e-3, rtol=2e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or flex_attention is None,
    reason="FlexAttention CUDA support is required",
)
def test_flex_output_matches_explicit_block_mask():
    torch.manual_seed(1)
    tokens, heads, head_dim, block = 250, 2, 32, 128
    q = torch.randn(1, tokens, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    route = torch.eye(2, device="cuda", dtype=torch.bool).expand(heads, -1, -1)

    block_mask = _build_block_mask(route, tokens, block)
    output = _flex_output(
        q, k, v, block_mask, block, compile_kernel=False)

    qh = q.transpose(1, 2).float()
    kh = k.transpose(1, 2).float()
    vh = v.transpose(1, 2).float()
    block_ids = torch.arange(tokens, device="cuda") // block
    allowed = route[:, block_ids[:, None], block_ids[None, :]].unsqueeze(0)
    scores = (qh @ kh.transpose(-1, -2)) * head_dim**-0.5
    reference = scores.masked_fill(~allowed, -torch.inf).softmax(-1) @ vh

    torch.testing.assert_close(
        output.transpose(1, 2).float(), reference,
        atol=5e-3, rtol=3e-2)
