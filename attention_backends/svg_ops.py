"""Small, self-contained ports of the SVG fast normalization/elementwise ops."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - GPU inference requires Triton.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _norm_kernel(
        x,
        y,
        weight,
        bias,
        rows,
        cols,
        eps,
        RMS: tl.constexpr,
        AFFINE: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_N)
        mask = offsets < cols
        values = tl.load(x + row * cols + offsets, mask=mask, other=0.0).to(tl.float32)
        if RMS:
            variance = tl.sum(values * values, axis=0) / cols
            output = values * tl.rsqrt(variance + eps)
        else:
            mean = tl.sum(values, axis=0) / cols
            centered = values - mean
            variance = tl.sum(centered * centered, axis=0) / cols
            output = centered * tl.rsqrt(variance + eps)
        if AFFINE:
            output = output * tl.load(weight + offsets, mask=mask)
            if not RMS:
                output += tl.load(bias + offsets, mask=mask)
        tl.store(y + row * cols + offsets, output, mask=mask)

    @triton.jit
    def _modulate_kernel(x, y, scale, shift, rows, cols, BLOCK_N: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_N)
        mask = offsets < cols
        values = tl.load(x + row * cols + offsets, mask=mask).to(tl.float32)
        scales = tl.load(scale + offsets, mask=mask)
        shifts = tl.load(shift + offsets, mask=mask)
        tl.store(y + row * cols + offsets, values * (1 + scales) + shifts, mask=mask)

    @triton.jit
    def _gate_residual_kernel(residual, x, y, gate, rows, cols, BLOCK_N: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_N)
        mask = offsets < cols
        residual_values = tl.load(residual + row * cols + offsets, mask=mask).to(
            tl.float32
        )
        values = tl.load(x + row * cols + offsets, mask=mask).to(tl.float32)
        gates = tl.load(gate + offsets, mask=mask)
        tl.store(y + row * cols + offsets, residual_values + values * gates, mask=mask)


def _matrix(tensor):
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor.reshape(-1, tensor.shape[-1])


def _norm(tensor, weight, bias, eps, *, rms, output_dtype):
    if triton is None or not tensor.is_cuda:
        if rms:
            values = tensor.float()
            output = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
            if weight is not None:
                output = output * weight
        else:
            output = torch.nn.functional.layer_norm(
                tensor.float(), (tensor.shape[-1],), weight, bias, eps
            )
        return output.to(output_dtype)
    original_shape = tensor.shape
    matrix = _matrix(tensor)
    rows, cols = matrix.shape
    output = torch.empty_like(matrix, dtype=output_dtype)
    affine = weight is not None
    pointer = weight if affine else matrix
    bias_pointer = bias if bias is not None else matrix
    _norm_kernel[(rows,)](
        matrix,
        output,
        pointer,
        bias_pointer,
        rows,
        cols,
        eps,
        RMS=rms,
        AFFINE=affine,
        BLOCK_N=triton.next_power_of_2(cols),
        num_warps=8,
    )
    return output.reshape(original_shape)


def rms_norm(tensor, weight, eps):
    return _norm(tensor, weight, None, eps, rms=True, output_dtype=tensor.dtype)


def layer_norm(tensor, weight, bias, eps):
    return _norm(tensor, weight, bias, eps, rms=False, output_dtype=torch.float32)


def modulate(tensor, scale, shift, output_dtype):
    if triton is None or not tensor.is_cuda:
        return (tensor.float() * (1 + scale) + shift).to(output_dtype)
    original_shape = tensor.shape
    matrix = _matrix(tensor)
    rows, cols = matrix.shape
    output = torch.empty_like(matrix, dtype=output_dtype)
    _modulate_kernel[(rows,)](
        matrix,
        output,
        scale,
        shift,
        rows,
        cols,
        BLOCK_N=triton.next_power_of_2(cols),
        num_warps=8,
    )
    return output.reshape(original_shape)


def gated_residual(residual, tensor, gate, output_dtype):
    if triton is None or not tensor.is_cuda:
        return (residual.float() + tensor.float() * gate).to(output_dtype)
    original_shape = tensor.shape
    residual_matrix = _matrix(residual)
    matrix = _matrix(tensor)
    rows, cols = matrix.shape
    output = torch.empty_like(matrix, dtype=output_dtype)
    _gate_residual_kernel[(rows,)](
        residual_matrix,
        matrix,
        output,
        gate,
        rows,
        cols,
        BLOCK_N=triton.next_power_of_2(cols),
        num_warps=8,
    )
    return output.reshape(original_shape)


def _load_rope_extension():
    try:
        module = importlib.import_module("_kernels")
    except ImportError:
        module = None
    if module is not None and hasattr(module, "apply_qk_rope_inplace_cossin_complex"):
        return module

    configured = os.getenv("SVG_KERNELS_BUILD")
    if configured and Path(configured).is_dir():
        sys.path.insert(0, configured)
        try:
            module = importlib.import_module("_kernels")
        except ImportError:
            return None
        if hasattr(module, "apply_qk_rope_inplace_cossin_complex"):
            return module
    return None


_ROPE_EXTENSION = _load_rope_extension()


def apply_rope(query, key, rotary_emb):
    """Apply SVG's fused Q/K RoPE, with an equivalent PyTorch fallback."""
    if _ROPE_EXTENSION is not None and query.is_cuda:
        real = rotary_emb.real.squeeze(0).squeeze(0).contiguous().float()
        imag = rotary_emb.imag.squeeze(0).squeeze(0).contiguous().float()
        _ROPE_EXTENSION.apply_qk_rope_inplace_cossin_complex(query, key, real, imag, 0)
        return query, key

    def reference(tensor):
        dtype = torch.float32 if tensor.device.type == "mps" else torch.float64
        values = torch.view_as_complex(tensor.to(dtype).unflatten(3, (-1, 2)))
        return torch.view_as_real(values * rotary_emb).flatten(3, 4).type_as(tensor)

    return reference(query), reference(key)


def operator_status():
    return {
        "triton_ops": triton is not None,
        "fused_rope": _ROPE_EXTENSION is not None,
    }
