import pytest
import torch

from attention_backends.flex import (
    _block_attention_mass,
    _build_block_mask,
    _dense_bootstrap,
    _flex_output,
    _pack_route,
    _sampled_block_mass,
    _top_mass_route,
    flex_attention,
)


def test_pack_route_preserves_selected_indices():
    route = torch.tensor([
        [[True, False, True], [False, True, False]],
        [[False, True, True], [True, False, True]],
    ])
    counts, indices = _pack_route(route)
    assert counts.tolist() == [[[2, 1], [2, 2]]]
    for head in range(route.shape[0]):
        for query in range(route.shape[1]):
            count = counts[0, head, query]
            selected = indices[0, head, query, :count]
            assert selected.tolist() == torch.where(route[head, query])[0].tolist()


def test_full_sampling_matches_exact_block_route_mass():
    torch.manual_seed(4)
    tokens, heads, head_dim, block = 12, 2, 8, 4
    q = torch.randn(1, tokens, heads, head_dim)
    k = torch.randn_like(q)
    scores = torch.einsum(
        "bqhd,bkhd->bhqk", q.float(), k.float()) * head_dim**-0.5
    lse = scores.logsumexp(-1)
    exact = _block_attention_mass(q, k, lse, block)
    sampled = _sampled_block_mass(q, k, block, samples=block)
    torch.testing.assert_close(sampled, exact / block, atol=1e-6, rtol=1e-6)


def test_top_mass_route_respects_row_cap():
    mass = torch.tensor([[[0.50, 0.30, 0.15, 0.05]]])
    route = _top_mass_route(mass, target=0.90, keep=0.50)
    assert route.tolist() == [[[True, True, False, False]]]


def test_reference_block_mass_matches_explicit_softmax():
    torch.manual_seed(2)
    tokens, heads, head_dim, block = 259, 3, 16, 128
    q = torch.randn(1, tokens, heads, head_dim)
    k = torch.randn_like(q)

    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * head_dim**-0.5
    lse = scores.logsumexp(-1)
    actual = _block_attention_mass(q, k, lse, block_size=block)
    probabilities = scores.softmax(-1)[0]
    padded = torch.nn.functional.pad(probabilities, (0, 125, 0, 125))
    expected = padded.reshape(heads, 3, block, 3, block).sum((2, 4))

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton exact-mass kernel requires CUDA",
)
def test_triton_block_mass_matches_explicit_softmax():
    torch.manual_seed(3)
    tokens, heads, head_dim, block = 250, 2, 32, 128
    q = torch.randn(1, tokens, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)

    scores = (
        q.transpose(1, 2).float()
        @ k.transpose(1, 2).float().transpose(-1, -2)
    ) * head_dim**-0.5
    lse = scores.logsumexp(-1)
    actual = _block_attention_mass(q, k, lse, block_size=block)
    probabilities = scores.softmax(-1)[0]
    padded = torch.nn.functional.pad(probabilities, (0, 6, 0, 6))
    expected = padded.reshape(heads, 2, block, 2, block).sum((2, 4))

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


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
        q, k, v, block_size=128, mass_target=0.90, keep=0.625,
        compile_kernel=False)
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
