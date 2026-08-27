"""Minimal FlexAttention backend with a route reused across denoising steps.

The first observation of every layer and CFG branch runs exact dense attention
and builds a 128-token top-mass route. Later steps execute that unchanged route
through FlexAttention's BlockMask. This intentionally small backend isolates
the output-kernel question from route prediction and update policies.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .dense import DenseBackend, install

try:
    import triton
    import triton.language as tl
except ImportError:  # Flex remains importable; exact GPU mass needs Triton.
    triton = None
    tl = None

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except ImportError:  # Kept importable on older PyTorch installations.
    create_block_mask = None
    flex_attention = None


@dataclass(frozen=True)
class FlexReuseConfig:
    block_size: int = 128
    keep: float = 0.625
    mass_target: float = 0.90
    compile_kernel: bool = True


def _routing_context():
    model = sys.modules.get("wan.modules.model")
    return (
        getattr(model, "_CURRENT_ATTN_ID", -1),
        getattr(model, "_CURRENT_DENOISE_STEP", -1),
        getattr(model, "_CURRENT_CFG_BRANCH", -1),
    )


def _top_mass_route(mass, target, keep):
    """Select K blocks independently for every (head, query-block) row."""
    order = mass.argsort(-1, descending=True)
    sorted_mass = torch.gather(mass, -1, order)
    thresholds = mass.sum(-1, keepdim=True) * target
    counts = (torch.cumsum(sorted_mass, -1) < thresholds).sum(-1) + 1
    counts.clamp_(max=max(1, math.ceil(mass.shape[-1] * keep)))

    ranks = torch.empty_like(order)
    rank_values = torch.arange(order.shape[-1], device=mass.device).expand_as(order)
    ranks.scatter_(-1, order, rank_values)
    return ranks < counts.unsqueeze(-1)


@torch.no_grad()
def _reference_block_mass(q, k, lse, block_size, scale):
    """Small CPU/reference implementation used when Triton is unavailable."""
    tokens, heads, head_dim = q.shape[1:]
    blocks = math.ceil(tokens / block_size)
    scores = torch.einsum(
        "bqhd,bkhd->bhqk", q.float(), k.float()) * scale
    probabilities = torch.exp(scores - lse.float().unsqueeze(-1))[0]
    padding = blocks * block_size - tokens
    probabilities = F.pad(probabilities, (0, padding, 0, padding))
    return probabilities.reshape(
        heads, blocks, block_size, blocks, block_size).sum((2, 4))


if triton is not None:
    @triton.jit
    def _exact_block_mass_kernel(
        q_ptr, k_ptr, lse_ptr, mass_ptr,
        tokens, head_dim, route_blocks, score_scale,
        stride_qt, stride_qh, stride_qd,
        stride_kt, stride_kh, stride_kd,
        stride_lh, stride_lt,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr, ROUTE_BLOCK: tl.constexpr,
    ):
        """Compute exact dense probability mass for one 128x64 QK tile."""
        pair = tl.program_id(0)
        head = tl.program_id(1)
        key_micro_blocks = tl.cdiv(tokens, BLOCK_N)
        query_block = pair // key_micro_blocks
        key_micro_block = pair % key_micro_blocks

        query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        key_offsets = key_micro_block * BLOCK_N + tl.arange(0, BLOCK_N)
        dimensions = tl.arange(0, BLOCK_D)
        query_valid = query_offsets < tokens
        key_valid = key_offsets < tokens

        query = tl.load(
            q_ptr + query_offsets[:, None] * stride_qt + head * stride_qh
            + dimensions[None, :] * stride_qd,
            mask=query_valid[:, None] & (dimensions[None, :] < head_dim),
            other=0.0,
        )
        key = tl.load(
            k_ptr + key_offsets[:, None] * stride_kt + head * stride_kh
            + dimensions[None, :] * stride_kd,
            mask=key_valid[:, None] & (dimensions[None, :] < head_dim),
            other=0.0,
        )
        scores = tl.dot(query, tl.trans(key)) * score_scale
        lse = tl.load(
            lse_ptr + head * stride_lh + query_offsets * stride_lt,
            mask=query_valid,
            other=float("inf"),
        )
        probabilities = tl.exp(scores - lse[:, None])
        probabilities = tl.where(
            query_valid[:, None] & key_valid[None, :], probabilities, 0.0)
        partial_mass = tl.sum(tl.sum(probabilities, axis=1), axis=0)

        key_route_block = (key_micro_block * BLOCK_N) // ROUTE_BLOCK
        output_offset = (
            head * route_blocks * route_blocks
            + query_block * route_blocks
            + key_route_block
        )
        tl.atomic_add(mass_ptr + output_offset, partial_mass)


@torch.no_grad()
def _block_attention_mass(q, k, lse, block_size, scale=None):
    """Recover exact dense block mass from Q, K, and per-query logsumexp.

    The GPU path recomputes QK once with tensor-core tiles, normalizes each
    score with the LSE returned by dense FlexAttention, and writes only one
    scalar per 128x128 route block. No probability tensor is materialized.
    """
    if q.shape[0] != 1 or q.shape[1] != k.shape[1]:
        raise ValueError("Flex bootstrap supports batch-1 self-attention")
    tokens, heads, head_dim = q.shape[1:]
    blocks = math.ceil(tokens / block_size)
    score_scale = scale if scale is not None else head_dim**-0.5
    if triton is None or not q.is_cuda:
        return _reference_block_mass(
            q, k, lse, block_size, score_scale)
    if block_size != 128:
        raise ValueError("The exact Triton mass kernel currently requires block_size=128")

    mass = torch.zeros((heads, blocks, blocks), device=q.device)
    block_m, block_n = 128, 64
    key_micro_blocks = triton.cdiv(tokens, block_n)
    grid = (blocks * key_micro_blocks, heads)
    _exact_block_mass_kernel[grid](
        q, k, lse, mass,
        tokens, head_dim, blocks, score_scale,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        lse.stride(1), lse.stride(2),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=max(16, triton.next_power_of_2(head_dim)),
        ROUTE_BLOCK=block_size,
        num_warps=8,
    )
    return mass


@torch.no_grad()
def _dense_bootstrap(q, k, v, block_size, mass_target, keep,
                     compile_kernel, scale=None):
    """Return exact dense output and its block-quantized top-mass route."""
    if q.shape[0] != 1 or q.shape[1] != k.shape[1]:
        raise ValueError("Flex bootstrap supports batch-1 self-attention")

    output, lse = _flex_kernel(compile_kernel)(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        scale=scale,
        return_lse=True,
    )
    mass = _block_attention_mass(
        q, k, lse, block_size, scale=scale)

    return (
        output.transpose(1, 2).contiguous(),
        _top_mass_route(mass, mass_target, keep),
    )


def _build_block_mask(route, tokens, block_size):
    """Convert a [head, Q-block, K-block] route to a Flex BlockMask."""
    if create_block_mask is None:
        raise RuntimeError("FlexAttention requires a PyTorch build with BlockMask support")
    if route.ndim != 3 or route.shape[-2] != route.shape[-1]:
        raise ValueError("route must have shape [head, Q-block, K-block]")

    route = route.unsqueeze(0).contiguous()
    padded_tokens = route.shape[-1] * block_size

    def mask_mod(batch, head, query_index, key_index):
        selected = route[
            batch, head, query_index // block_size, key_index // block_size]
        valid = (query_index < tokens) & (key_index < tokens)
        # Padding query rows are discarded after the kernel, but still need a
        # valid key to keep online softmax numerically well-defined.
        padding_fallback = (query_index >= tokens) & (key_index == 0)
        return (valid & selected) | padding_fallback

    return create_block_mask(
        mask_mod,
        B=1,
        H=route.shape[1],
        Q_LEN=padded_tokens,
        KV_LEN=padded_tokens,
        device=route.device,
        BLOCK_SIZE=block_size,
        _compile=True,
    )


_COMPILED_FLEX_ATTENTION = None


def _flex_kernel(compile_kernel):
    global _COMPILED_FLEX_ATTENTION
    if flex_attention is None:
        raise RuntimeError("FlexAttention is unavailable in this PyTorch build")
    if not compile_kernel:
        return flex_attention
    if _COMPILED_FLEX_ATTENTION is None:
        _COMPILED_FLEX_ATTENTION = torch.compile(
            flex_attention, fullgraph=True, dynamic=False)
    return _COMPILED_FLEX_ATTENTION


def _flex_output(q, k, v, block_mask, block_size, compile_kernel, scale=None):
    """Run FlexAttention on padded [batch, head, token, dim] tensors."""
    tokens = q.shape[1]
    padded_tokens = math.ceil(tokens / block_size) * block_size
    padding = padded_tokens - tokens

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    if padding:
        q = F.pad(q, (0, 0, 0, padding))
        k = F.pad(k, (0, 0, 0, padding))
        v = F.pad(v, (0, 0, 0, padding))

    output = _flex_kernel(compile_kernel)(
        q, k, v, block_mask=block_mask, scale=scale)
    return output[:, :, :tokens].transpose(1, 2).contiguous()


class FlexReuseBackend:
    """Dense bootstrap followed by an unchanged FlexAttention route."""

    name = "flex_reuse"

    def __init__(self, config=None):
        self.config = config or FlexReuseConfig()
        if self.config.block_size != 128:
            raise ValueError(
                "The minimal flex_reuse backend currently supports only 128-token blocks")
        self.dense = DenseBackend()
        self.state = {}

    def __call__(
        self,
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.0,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=torch.bfloat16,
        version=None,
        **kwargs,
    ):
        # Cross-attention and uncommon attention modes remain on the mature
        # dense backend in this minimal version.
        supported = (
            q.shape[0] == 1
            and q.shape[1] == k.shape[1]
            and dropout_p == 0
            and not causal
            and window_size == (-1, -1)
        )
        if not supported:
            return self.dense(
                q, k, v, q_lens=q_lens, k_lens=k_lens,
                dropout_p=dropout_p, softmax_scale=softmax_scale,
                q_scale=q_scale, causal=causal, window_size=window_size,
                deterministic=deterministic, dtype=dtype, version=version,
                **kwargs)

        output_dtype = q.dtype
        q = q.to(dtype)
        k = k.to(dtype)
        v = v.to(dtype)
        if q_scale is not None:
            q = q * q_scale

        attention_id, step, branch = _routing_context()
        key = (
            attention_id, branch, q.shape[1], q.shape[2], q.shape[3],
            q.device,
        )
        if step == 0 or key not in self.state:
            output, route = _dense_bootstrap(
                q, k, v,
                block_size=self.config.block_size,
                mass_target=self.config.mass_target,
                keep=self.config.keep,
                compile_kernel=self.config.compile_kernel,
                scale=softmax_scale,
            )
            self.state[key] = {
                "route": route,
                "block_mask": _build_block_mask(
                    route, q.shape[1], self.config.block_size),
            }
            return output.to(output_dtype)

        output = _flex_output(
            q, k, v,
            self.state[key]["block_mask"],
            self.config.block_size,
            self.config.compile_kernel,
            scale=softmax_scale,
        )
        return output.to(output_dtype)


def install_flex_reuse(config=None):
    return install(FlexReuseBackend(config))
