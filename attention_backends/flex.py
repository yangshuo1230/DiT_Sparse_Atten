"""Hybrid FlexAttention backend with exact or sampled reusable routes.

Step 0 always returns complete dense attention output. Route measurement can be
exact or sampled; later steps can reuse, lightly update, or dispatch high-keep
routes back to dense SDPA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .context import routing_context
from .dense import DenseBackend, install
from .layout_ops import pack_boolean_rows
from .routing import directional_budget_update, frontier_mask
from .spatial import build_spatial_layout

try:
    import triton
    import triton.language as tl
except ImportError:  # Flex remains importable; exact GPU mass needs Triton.
    triton = None
    tl = None

try:
    from torch.nn.attention.flex_attention import (
        BlockMask,
        create_block_mask,
        flex_attention,
        noop_mask,
    )
except ImportError:  # Kept importable on older PyTorch installations.
    BlockMask = None
    create_block_mask = None
    flex_attention = None
    noop_mask = None


@dataclass(frozen=True)
class FlexReuseConfig:
    block_size: int = 128
    keep: float = 0.625
    mass_target: float = 0.90
    compile_kernel: bool = True
    # 0 keeps the step-0 route static. A positive value performs an expensive
    # exact dense refresh every N denoising steps.
    update_interval: int = 0
    # A positive value updates from sampled Q/K blocks without dense attention.
    sampled_update_interval: int = 0
    route_samples: int = 2
    sample_query_block_chunk: int = 16
    route_persistence: float = 0.5
    prefetch_sampled_update: bool = True
    prefetch_sampled_bootstrap: bool = True
    total_steps: int | None = None
    # exact: dense Flex+LSE and a second full QK mass pass; sampled: dense SDPA
    # output plus lightweight route estimation, optionally prefetched.
    bootstrap_mode: str = "exact"
    # Routes at or above this keep fraction are faster on dense SDPA. None
    # disables automatic per-layer/branch dispatch.
    dense_route_threshold: float | None = None
    release_after_final_step: bool = True
    spatial_reorder: bool = False
    spatial_microtile_tokens: int = 32
    directional_update: bool = False
    direction_min_ratio: float = 0.0
    direction_candidate_bonus: float = 0.0
    route_budget_scale: float = 1.0
    route_exploration_fraction: float = 0.0
    direction_expand_q: bool = True
    direction_expand_k: bool = True
    direction_expand_joint: bool = True


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
    tokens, heads, _head_dim = q.shape[1:]
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
def _sampled_block_mass(q, k, block_size, samples=2, scale=None,
                        query_block_chunk=16):
    """Estimate block mass from evenly sampled Q/K tokens per block.

    This never runs dense attention or forms full token-level QK. At 57,600
    tokens, block size 128 and two samples, the score matrix has 1/4096 as many
    elements as full QK. Query blocks are chunked to bound temporary memory.
    """
    if q.shape[0] != 1 or q.shape[1] != k.shape[1]:
        raise ValueError("sampled route update supports batch-1 self-attention")
    if samples <= 0 or samples > block_size:
        raise ValueError("route_samples must be in [1, block_size]")
    if query_block_chunk <= 0:
        raise ValueError("sample_query_block_chunk must be positive")

    tokens, heads, head_dim = q.shape[1:]
    blocks = math.ceil(tokens / block_size)
    block_ids = torch.arange(blocks, device=q.device)
    sizes = (tokens - block_ids * block_size).clamp(max=block_size)
    sample_ids = torch.arange(samples, device=q.device)
    local_indices = torch.div(
        (2 * sample_ids + 1)[None] * sizes[:, None],
        2 * samples,
        rounding_mode="floor",
    ).clamp_max(sizes[:, None] - 1)
    indices = block_ids[:, None] * block_size + local_indices

    sampled_q = q[0, indices].permute(2, 0, 1, 3).contiguous()
    sampled_k = k[0, indices].permute(2, 0, 1, 3).contiguous()
    flat_k = sampled_k.reshape(heads, blocks * samples, head_dim)
    key_log_weights = (sizes.float() / samples).log()
    key_log_weights = key_log_weights[:, None].expand(-1, samples).reshape(-1)
    score_scale = scale if scale is not None else head_dim**-0.5
    mass = torch.empty((heads, blocks, blocks), device=q.device)

    for start in range(0, blocks, query_block_chunk):
        end = min(start + query_block_chunk, blocks)
        query = sampled_q[:, start:end].reshape(
            heads, (end - start) * samples, head_dim)
        scores = torch.matmul(query, flat_k.transpose(1, 2)) * score_scale
        probabilities = torch.softmax(
            scores.float() + key_log_weights[None, None], dim=-1)
        mass[:, start:end] = probabilities.reshape(
            heads, end - start, samples, blocks, samples,
        ).sum(-1).mean(2)
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


def _pack_route(route):
    """Pack a boolean [H,Q,K] route into counts and compact K indices."""
    # Keep selected indices first and every unselected index afterward.  The
    # CUDA path performs a stable linear compaction instead of a full sort.
    # Unselected IDs are still retained for PyTorch's KV-to-Q metadata builder.
    counts, indices = pack_boolean_rows(route)
    return counts.unsqueeze(0), indices.to(torch.int32).unsqueeze(0)


_EMPTY_BLOCK_METADATA = {}


def _empty_block_metadata(route):
    """Return shared zero-count partial-block metadata for a route shape."""
    heads, query_blocks, _key_blocks = route.shape
    key = (route.device, heads, query_blocks)
    metadata = _EMPTY_BLOCK_METADATA.get(key)
    if metadata is None:
        counts = torch.zeros(
            (1, heads, query_blocks), device=route.device, dtype=torch.int32)
        # FlexAttention uses the final index dimension to infer the logical
        # number of K blocks even when every count is zero, so it must retain
        # full width.  Sharing this immutable tensor still avoids two dense
        # BxQxK int32 allocations in every persistent BlockMask.
        indices = torch.arange(
            query_blocks, device=route.device, dtype=torch.int32,
        ).view(1, 1, 1, query_blocks).expand(
            1, heads, query_blocks, query_blocks).contiguous()
        metadata = (counts, indices)
        _EMPTY_BLOCK_METADATA[key] = metadata
    return metadata


def _build_block_mask(route, tokens, block_size):
    """Convert [H,Q,K] route directly to FlexAttention's compressed form."""
    if BlockMask is None:
        raise RuntimeError("FlexAttention requires a PyTorch build with BlockMask support")
    if route.ndim != 3 or route.shape[-2] != route.shape[-1]:
        raise ValueError("route must have shape [head, Q-block, K-block]")

    counts, indices = _pack_route(route)
    padded_tokens = route.shape[-1] * block_size

    if tokens == padded_tokens:
        # Construct both traversal directions directly from the boolean route.
        # ``from_kv_blocks`` densifies and sorts once more to derive Q metadata;
        # packing the transpose with the same stable linear kernel is equivalent
        # and keeps BlockMask construction entirely O(H*Q*K).
        q_counts, q_indices = _pack_route(
            route.transpose(-2, -1).contiguous())
        empty_counts, empty_indices = _empty_block_metadata(route)
        return BlockMask(
            seq_lengths=(padded_tokens, padded_tokens),
            kv_num_blocks=empty_counts,
            kv_indices=empty_indices,
            full_kv_num_blocks=counts,
            full_kv_indices=indices,
            q_num_blocks=empty_counts,
            q_indices=empty_indices,
            full_q_num_blocks=q_counts,
            full_q_indices=q_indices,
            BLOCK_SIZE=(block_size, block_size),
            mask_mod=noop_mask,
        )

    # A partial final Q/K block needs element-level masking. Retain the mature
    # builder for this uncommon path; Wan's 57,600-token target is divisible.
    route_with_batch = route.unsqueeze(0).contiguous()

    def boundary_mask(batch, head, query_index, key_index):
        selected = route_with_batch[
            batch, head, query_index // block_size, key_index // block_size]
        valid = (query_index < tokens) & (key_index < tokens)
        padding_fallback = (query_index >= tokens) & (key_index == 0)
        return (valid & selected) | padding_fallback

    return create_block_mask(
        boundary_mask, B=1, H=route.shape[0],
        Q_LEN=padded_tokens, KV_LEN=padded_tokens,
        device=route.device, BLOCK_SIZE=block_size, _compile=True,
    )


_COMPILED_FLEX_ATTENTION = None

# Every route producer in this backend retains at least one K block per query
# block. Partial-tail masks also route padded queries to the fallback key. This
# lets the generated FlexAttention kernel omit empty-row guards and denominator
# fixups without changing attention semantics.
_SPARSE_KERNEL_OPTIONS = {"ROWS_GUARANTEED_SAFE": True}


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
        q, k, v,
        block_mask=block_mask,
        scale=scale,
        kernel_options=_SPARSE_KERNEL_OPTIONS,
    )
    return output[:, :, :tokens].transpose(1, 2).contiguous()


class FlexReuseBackend:
    """Dense bootstrap followed by reuse with optional exact route refresh."""

    name = "flex_reuse"

    def __init__(self, config=None):
        self.config = config or FlexReuseConfig()
        if self.config.block_size != 128:
            raise ValueError(
                "The minimal flex_reuse backend currently supports only 128-token blocks")
        if self.config.update_interval < 0:
            raise ValueError("update_interval must be zero or a positive integer")
        if self.config.sampled_update_interval < 0:
            raise ValueError(
                "sampled_update_interval must be zero or a positive integer")
        if self.config.update_interval and self.config.sampled_update_interval:
            raise ValueError("exact and sampled route updates are mutually exclusive")
        if not 0 < self.config.route_samples <= self.config.block_size:
            raise ValueError("route_samples must be in [1, block_size]")
        if self.config.route_persistence < 0:
            raise ValueError("route_persistence must be non-negative")
        if self.config.bootstrap_mode not in ("exact", "sampled"):
            raise ValueError("bootstrap_mode must be 'exact' or 'sampled'")
        if (self.config.dense_route_threshold is not None and
                not 0 < self.config.dense_route_threshold <= 1):
            raise ValueError("dense_route_threshold must be in (0, 1]")
        if not 0 < self.config.keep <= 1:
            raise ValueError("keep must be in (0, 1]")
        if not 0 < self.config.mass_target <= 1:
            raise ValueError("mass_target must be in (0, 1]")
        if self.config.spatial_microtile_tokens <= 0:
            raise ValueError("spatial_microtile_tokens must be positive")
        if self.config.direction_min_ratio < 0:
            raise ValueError("direction_min_ratio must be non-negative")
        if self.config.direction_candidate_bonus < 0:
            raise ValueError("direction_candidate_bonus must be non-negative")
        if self.config.route_budget_scale <= 0:
            raise ValueError("route_budget_scale must be positive")
        if not 0 <= self.config.route_exploration_fraction <= 1:
            raise ValueError("route_exploration_fraction must be in [0, 1]")
        if self.config.directional_update and not self.config.spatial_reorder:
            raise ValueError("directional_update requires spatial_reorder")
        self.dense = DenseBackend()
        self.state = {}
        self.route_streams = {}
        self.spatial_layouts = {}
        self.phase_counts = {
            "dense_bootstrap": 0,
            "sampled_bootstrap": 0,
            "exact_refresh": 0,
            "sampled_update": 0,
            "sparse_reuse": 0,
            "dense_dispatch": 0,
        }
        self.finished_routes = []

    def _activate_pending_route(self, state):
        pending = state.pop("pending", None)
        if pending is None:
            return
        torch.cuda.current_stream(pending["route"].device).wait_event(
            pending["ready"])
        state.update({
            "route": pending["route"],
            "frontier": pending.get("frontier"),
            "block_mask": pending["block_mask"],
            "step": pending["source_step"],
        })
        self._update_dispatch(state)

    def _update_dispatch(self, state):
        threshold = self.config.dense_route_threshold
        keep = (float(state["route"].float().mean().item())
                if state.get("route") is not None else None)
        state["keep"] = keep
        frontier = state.get("frontier")
        state["frontier_fraction"] = (
            float(frontier.float().mean().item())
            if frontier is not None else None)
        state["dense"] = bool(
            threshold is not None
            and keep is not None
            and keep >= threshold
        )
        if state["dense"]:
            # Permanently dense-dispatched states never consume their route or
            # BlockMask again. At 57K these tensors are large enough that
            # retaining them can prevent a later sparse route replacement.
            state["route"] = None
            state["block_mask"] = None
            state["frontier"] = None

    def _is_final_step(self, step):
        return (self.config.release_after_final_step
                and self.config.total_steps is not None
                and step >= self.config.total_steps - 1)

    def _release_route(self, key, state):
        self.finished_routes.append({
            "keep": state.get("keep"),
            "frontier_fraction": state.get("frontier_fraction"),
            "dense": bool(state.get("dense")),
        })
        self.state.pop(key, None)

    def _sampled_route(self, state, q, k, softmax_scale):
        mass = _sampled_block_mass(
            q, k, self.config.block_size,
            samples=self.config.route_samples,
            scale=softmax_scale,
            query_block_chunk=self.config.sample_query_block_chunk,
        )
        previous = state.get("route")
        layout = state.get("layout")
        if (self.config.directional_update and previous is not None
                and layout is not None):
            return directional_budget_update(
                mass,
                previous,
                layout.neighbors(q.device),
                keep=self.config.keep,
                persistence=self.config.route_persistence,
                min_ratio=self.config.direction_min_ratio,
                candidate_bonus=self.config.direction_candidate_bonus,
                budget_scale=self.config.route_budget_scale,
                exploration_fraction=self.config.route_exploration_fraction,
                expand_q=self.config.direction_expand_q,
                expand_k=self.config.direction_expand_k,
                expand_joint=self.config.direction_expand_joint,
                previous_frontier=state.get("frontier"),
            )
        if self.config.route_persistence and previous is not None:
            mass = (
                mass
                + self.config.route_persistence
                * mass.mean(-1, keepdim=True)
                * previous.float()
            )
        route = _top_mass_route(
            mass, self.config.mass_target, self.config.keep)
        frontier = None
        if layout is not None:
            frontier = frontier_mask(route, layout.neighbors(q.device))
        return {"route": route, "frontier": frontier}

    def _install_route(self, state, result, tokens):
        route = result["route"]
        state.update({
            "route": route,
            "frontier": result.get("frontier"),
            "block_mask": _build_block_mask(
                route, tokens, self.config.block_size),
        })

    def _spatial_layout(self, grid, tokens):
        if not self.config.spatial_reorder:
            return None
        if grid is None or math.prod(grid) != tokens:
            return None
        key = (tuple(grid), self.config.block_size,
               self.config.spatial_microtile_tokens)
        if key not in self.spatial_layouts:
            self.spatial_layouts[key] = build_spatial_layout(
                grid, self.config.block_size,
                self.config.spatial_microtile_tokens)
        return self.spatial_layouts[key]

    def _prefetch_sampled_route(self, state, q, k, step, softmax_scale,
                                phase="sampled_update"):
        stream = self.route_streams.setdefault(
            q.device, torch.cuda.Stream(device=q.device))
        attention_done = torch.cuda.Event()
        attention_done.record(torch.cuda.current_stream(q.device))
        stream.wait_event(attention_done)
        with torch.cuda.stream(stream):
            q.record_stream(stream)
            k.record_stream(stream)
            result = self._sampled_route(state, q, k, softmax_scale)
            route = result["route"]
            block_mask = _build_block_mask(
                route, q.shape[1], self.config.block_size)
            ready = torch.cuda.Event()
            ready.record(stream)
        state["pending"] = {
            "route": route,
            "frontier": result.get("frontier"),
            "block_mask": block_mask,
            "source_step": step,
            "ready": ready,
        }
        self.phase_counts[phase] += 1

    def profile_summary(self):
        active = [state for state in self.state.values()
                  if state.get("route") is not None]
        keep = [state.get("keep") for state in active] + [
            state["keep"] for state in self.finished_routes
            if state["keep"] is not None]
        frontier = [state.get("frontier_fraction") for state in active] + [
            state["frontier_fraction"] for state in self.finished_routes
            if state["frontier_fraction"] is not None]
        return {
            "phase_counts": dict(self.phase_counts),
            "active_routes": len(active),
            "released_routes": len(self.finished_routes),
            "mean_route_keep": sum(keep) / len(keep) if keep else None,
            "mean_frontier_fraction": (
                sum(frontier) / len(frontier) if frontier else None),
            "dense_dispatched_routes": sum(
                bool(state.get("dense")) for state in self.state.values())
                + sum(state["dense"] for state in self.finished_routes),
        }

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

        attention_id, step, branch, grid = routing_context(include_grid=True)
        layout = self._spatial_layout(grid, q.shape[1])
        if self.config.spatial_reorder and layout is None:
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
        if layout is not None:
            q, k, v = layout.reorder_qkv(q, k, v)

        def finish(output):
            if layout is not None:
                output = layout.restore(output)
            return output.to(output_dtype)

        key = (
            attention_id, branch, q.shape[1], q.shape[2], q.shape[3],
            q.device, layout.signature if layout is not None else None,
        )
        if step == 0 or key not in self.state:
            if self.config.bootstrap_mode == "exact":
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
                    "frontier": (frontier_mask(
                        route, layout.neighbors(q.device))
                        if layout is not None else None),
                    "block_mask": _build_block_mask(
                        route, q.shape[1], self.config.block_size),
                    "step": step,
                    "layout": layout,
                }
                self._update_dispatch(self.state[key])
                self.phase_counts["dense_bootstrap"] += 1
            else:
                # The generated output is still complete dense attention. Only
                # the route estimator is sampled, avoiding a second full QK.
                output = self.dense(
                    q, k, v, dropout_p=0.0,
                    softmax_scale=softmax_scale, q_scale=None,
                    causal=False, window_size=(-1, -1), dtype=dtype)
                state = {
                    "route": None,
                    "frontier": None,
                    "block_mask": None,
                    "step": step,
                    "layout": layout,
                }
                self.state[key] = state
                if (self.config.prefetch_sampled_bootstrap
                        and (self.config.total_steps is None
                             or step < self.config.total_steps - 1)):
                    self._prefetch_sampled_route(
                        state, q, k, step, softmax_scale,
                        phase="sampled_bootstrap")
                else:
                    result = self._sampled_route(
                        state, q, k, softmax_scale)
                    self._install_route(state, result, q.shape[1])
                    self._update_dispatch(state)
                    self.phase_counts["sampled_bootstrap"] += 1
            if self._is_final_step(step):
                self._release_route(key, self.state[key])
            return finish(output)

        state = self.state[key]
        self._activate_pending_route(state)
        if state.get("dense"):
            self.phase_counts["dense_dispatch"] += 1
            output = self.dense(
                q, k, v, dropout_p=0.0,
                softmax_scale=softmax_scale, q_scale=None,
                causal=False, window_size=(-1, -1), dtype=dtype,
            ).to(output_dtype)
            if self._is_final_step(step):
                self._release_route(key, state)
            return finish(output)
        if (self.config.update_interval > 0 and
                (step - state["step"]) >= self.config.update_interval and
                not self._is_final_step(step)):
            output, lse = _flex_kernel(self.config.compile_kernel)(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                scale=softmax_scale,
                return_lse=True,
            )
            mass = _block_attention_mass(
                q, k, lse, self.config.block_size, scale=softmax_scale)
            route = _top_mass_route(
                mass, self.config.mass_target, self.config.keep)
            state.update({
                "route": route,
                "frontier": (frontier_mask(
                    route, layout.neighbors(q.device))
                    if layout is not None else None),
                "block_mask": _build_block_mask(
                    route, q.shape[1], self.config.block_size),
                "step": step,
            })
            self._update_dispatch(state)
            self.phase_counts["exact_refresh"] += 1
            return finish(output.transpose(1, 2).contiguous())

        sampled_update_due = (
            self.config.sampled_update_interval > 0
            and (step - state["step"]) >= self.config.sampled_update_interval
            and (self.config.total_steps is None
                 or step < self.config.total_steps - 1)
        )
        if sampled_update_due and not self.config.prefetch_sampled_update:
            # The old mask will not execute this step; release its large
            # bidirectional index tensors before allocating sampled mass and
            # the replacement BlockMask.
            state["block_mask"] = None
            result = self._sampled_route(
                state, q, k, softmax_scale)
            self._install_route(state, result, q.shape[1])
            state["step"] = step
            self._update_dispatch(state)
            self.phase_counts["sampled_update"] += 1
            if state.get("dense"):
                self.phase_counts["dense_dispatch"] += 1
                output = self.dense(
                    q, k, v, dropout_p=0.0,
                    softmax_scale=softmax_scale, q_scale=None,
                    causal=False, window_size=(-1, -1), dtype=dtype,
                )
                return finish(output)

        output = _flex_output(
            q, k, v,
            state["block_mask"],
            self.config.block_size,
            self.config.compile_kernel,
            scale=softmax_scale,
        )
        self.phase_counts["sparse_reuse"] += 1
        should_prefetch = (
            self.config.prefetch_sampled_update
            and sampled_update_due
        )
        if should_prefetch:
            # FlexAttention has already enqueued all reads from the current
            # BlockMask on the main stream. Dropping the Python reference lets
            # the caching allocator reclaim it once those reads complete.
            state["block_mask"] = None
            self._prefetch_sampled_route(
                state, q, k, step, softmax_scale)
        if self._is_final_step(step):
            self._release_route(key, state)
        return finish(output)


def install_flex_reuse(config=None):
    return install(FlexReuseBackend(config))
