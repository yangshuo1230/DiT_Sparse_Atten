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
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except ImportError:  # Kept importable on older PyTorch installations.
    create_block_mask = None
    flex_attention = None


@dataclass(frozen=True)
class FlexReuseConfig:
    block_size: int = 128
    keep: float = 0.625
    mass_target: float = 0.90
    query_chunk: int = 256
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
def _dense_bootstrap(q, k, v, block_size, query_chunk, mass_target, keep,
                     scale=None):
    """Return exact dense output and its block-quantized top-mass route."""
    if q.shape[0] != 1 or q.shape[1] != k.shape[1]:
        raise ValueError("Flex bootstrap supports batch-1 self-attention")

    tokens, heads, head_dim = q.shape[1:]
    blocks = math.ceil(tokens / block_size)
    padded_tokens = blocks * block_size
    mass = torch.zeros((heads, blocks, blocks), device=q.device)
    output = torch.empty(
        (1, tokens, heads, v.shape[-1]), device=q.device, dtype=q.dtype)

    k_float = k.float()
    v_float = v[0].permute(1, 0, 2).float()
    score_scale = scale if scale is not None else head_dim**-0.5
    query_chunk = max(block_size, query_chunk // block_size * block_size)

    for start in range(0, tokens, query_chunk):
        end = min(start + query_chunk, tokens)
        scores = torch.einsum(
            "bqhd,bkhd->bhqk", q[:, start:end].float(), k_float)
        probabilities = (scores * score_scale).softmax(-1)[0]
        chunk_output = probabilities.matmul(v_float)
        output[:, start:end] = chunk_output.permute(1, 0, 2).unsqueeze(0).to(q.dtype)

        query_blocks = math.ceil((end - start) / block_size)
        padded = F.pad(
            probabilities,
            (0, padded_tokens - tokens,
             0, query_blocks * block_size - (end - start)))
        shape = (heads, query_blocks, block_size, blocks, block_size)
        destination = slice(start // block_size, start // block_size + query_blocks)
        mass[:, destination] = padded.reshape(shape).sum((2, 4))

    return output, _top_mass_route(mass, mass_target, keep)


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
                query_chunk=self.config.query_chunk,
                mass_target=self.config.mass_target,
                keep=self.config.keep,
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
