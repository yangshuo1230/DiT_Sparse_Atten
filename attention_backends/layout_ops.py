"""CUDA data-movement kernels shared by sparse attention backends.

These kernels only change tensor placement.  They deliberately do not encode
any routing policy so the PyTorch fallbacks are also useful as executable
specifications for their semantics.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # Keep CPU-only and older environments importable.
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _reorder_qkv_kernel(
        q_ptr, k_ptr, v_ptr,
        q_out_ptr, k_out_ptr, v_out_ptr,
        permutation_ptr,
        tokens, hidden,
        BLOCK_S: tl.constexpr, BLOCK_HIDDEN: tl.constexpr,
    ):
        """Gather Q/K/V through one shared permutation in one launch."""
        sequence_block = tl.program_id(0)
        hidden_block = tl.program_id(1)
        batch = tl.program_id(2)
        destination = sequence_block * BLOCK_S + tl.arange(0, BLOCK_S)
        features = hidden_block * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
        source = tl.load(
            permutation_ptr + destination,
            mask=destination < tokens,
            other=0,
        )
        valid = (destination[:, None] < tokens) & (
            features[None, :] < hidden)
        batch_offset = batch * tokens * hidden
        input_offset = batch_offset + source[:, None] * hidden + features[None, :]
        output_offset = (
            batch_offset + destination[:, None] * hidden + features[None, :])

        q_value = tl.load(q_ptr + input_offset, mask=valid, other=0.0)
        tl.store(q_out_ptr + output_offset, q_value, mask=valid)
        k_value = tl.load(k_ptr + input_offset, mask=valid, other=0.0)
        tl.store(k_out_ptr + output_offset, k_value, mask=valid)
        v_value = tl.load(v_ptr + input_offset, mask=valid, other=0.0)
        tl.store(v_out_ptr + output_offset, v_value, mask=valid)


    @triton.jit
    def _pack_boolean_rows_kernel(
        route_ptr, counts_ptr, indices_ptr,
        columns: tl.constexpr, BLOCK_COLUMNS: tl.constexpr,
    ):
        """Stable-partition each boolean row into selected/unselected IDs."""
        row = tl.program_id(0)
        columns_offset = tl.arange(0, BLOCK_COLUMNS)
        valid = columns_offset < columns
        selected = tl.load(
            route_ptr + row * columns + columns_offset,
            mask=valid,
            other=0,
        ).to(tl.int32)
        selected_prefix = tl.cumsum(selected, axis=0)
        count = tl.sum(selected, axis=0)
        # This is the same stable order as argsort(bool, descending=True):
        # ascending selected IDs, followed by ascending unselected IDs.
        destination = tl.where(
            selected != 0,
            selected_prefix - 1,
            count + columns_offset - selected_prefix,
        )
        tl.store(
            indices_ptr + row * columns + destination,
            columns_offset,
            mask=valid,
        )
        tl.store(counts_ptr + row, count)


def reorder_qkv(q, k, v, permutation):
    """Return Q/K/V reordered on axis 1, fusing CUDA data movement.

    The fallback intentionally mirrors three ``index_select`` calls exactly.
    """
    compatible = (
        triton is not None
        and q.is_cuda and k.is_cuda and v.is_cuda
        and q.ndim == k.ndim == v.ndim == 4
        and q.shape == k.shape == v.shape
        and q.dtype == k.dtype == v.dtype
        and q.device == k.device == v.device == permutation.device
        and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    )
    if not compatible:
        return tuple(tensor.index_select(1, permutation) for tensor in (q, k, v))

    batch, tokens, heads, head_dim = q.shape
    if permutation.ndim != 1 or permutation.numel() != tokens:
        raise ValueError("permutation must contain one index per token")
    permutation = permutation.to(dtype=torch.int32).contiguous()
    outputs = tuple(torch.empty_like(tensor) for tensor in (q, k, v))
    # One complete 1024-feature segment gives coalesced accesses across HxD.
    # Keeping BLOCK_S=1 avoids multiplying live Q/K/V vectors and register use.
    block_s = 1
    block_hidden = 1024
    hidden = heads * head_dim
    grid = (
        triton.cdiv(tokens, block_s),
        triton.cdiv(hidden, block_hidden),
        batch,
    )
    _reorder_qkv_kernel[grid](
        q, k, v, *outputs, permutation,
        tokens, hidden,
        BLOCK_S=block_s,
        BLOCK_HIDDEN=block_hidden,
        num_warps=8,
    )
    return outputs


def pack_boolean_rows(route):
    """Pack selected column IDs first in every row, preserving ID order."""
    if route.dtype != torch.bool or route.ndim < 2:
        raise ValueError("route must be a boolean tensor with at least two axes")
    columns = route.shape[-1]
    use_triton = (
        triton is not None and route.is_cuda and 0 < columns <= 4096)
    if not use_triton:
        counts = route.sum(-1, dtype=torch.int32)
        indices = torch.argsort(
            route.to(torch.int8), dim=-1, descending=True, stable=True)
        return counts, indices.to(torch.int32)

    route = route.contiguous()
    counts = torch.empty(route.shape[:-1], device=route.device, dtype=torch.int32)
    indices = torch.empty(route.shape, device=route.device, dtype=torch.int32)
    rows = route.numel() // columns
    _pack_boolean_rows_kernel[(rows,)](
        route, counts, indices,
        columns=columns,
        BLOCK_COLUMNS=triton.next_power_of_2(columns),
        num_warps=8 if columns >= 256 else 4,
    )
    return counts, indices
