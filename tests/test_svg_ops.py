import pytest
import torch

from attention_backends.svg_ops import (
    apply_rope,
    gated_residual,
    layer_norm,
    modulate,
    rms_norm,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_svg_norm_and_elementwise_ops_match_torch():
    torch.manual_seed(23)
    x = torch.randn(2, 17, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(1, 128, device="cuda")
    shift = torch.randn(1, 128, device="cuda")
    gate = torch.randn(1, 128, device="cuda")

    rms_reference = torch.nn.functional.rms_norm(
        x.float(), (128,), weight.float(), 1e-6
    ).to(x.dtype)
    torch.testing.assert_close(
        rms_norm(x, weight, 1e-6), rms_reference, atol=3e-2, rtol=3e-2
    )

    layer_reference = torch.nn.functional.layer_norm(
        x.float(), (128,), weight.float(), bias.float(), 1e-6
    )
    torch.testing.assert_close(
        layer_norm(x, weight, bias, 1e-6), layer_reference, atol=3e-3, rtol=3e-3
    )

    modulated = modulate(x, scale, shift, x.dtype)
    torch.testing.assert_close(
        modulated, (x.float() * (1 + scale) + shift).to(x.dtype), atol=2e-2, rtol=2e-2
    )
    residual = torch.randn_like(x)
    torch.testing.assert_close(
        gated_residual(residual, x, gate, x.dtype),
        (residual.float() + x.float() * gate).to(x.dtype),
        atol=2e-2,
        rtol=2e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_svg_fused_rope_matches_diffusers_reference():
    torch.manual_seed(29)
    query = torch.randn(1, 4, 33, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    rotary = torch.polar(
        torch.ones(1, 1, 33, 64, device="cuda", dtype=torch.float64),
        torch.randn(1, 1, 33, 64, device="cuda", dtype=torch.float64),
    )

    def reference(tensor):
        values = torch.view_as_complex(tensor.to(torch.float64).unflatten(3, (-1, 2)))
        return torch.view_as_real(values * rotary).flatten(3, 4).to(tensor.dtype)

    expected_query = reference(query)
    expected_key = reference(key)
    actual_query, actual_key = apply_rope(query.contiguous(), key.contiguous(), rotary)
    torch.testing.assert_close(actual_query, expected_query, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(actual_key, expected_key, atol=3e-2, rtol=3e-2)
