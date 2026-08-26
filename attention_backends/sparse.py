"""Full QxK matrix sparse reference backend."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import torch

from .dense import DenseBackend, install


@dataclass(frozen=True)
class SparseConfig:
    """Routing and chunking controls for the matrix-tile reference backend."""

    # Target tokens per spatial tile. The actual HxW shape is chosen from exact
    # divisors of the current Wan grid, so spatial tiles never need padding.
    tile: int = 64
    # Select enough key tiles to reach mass_target, but never more than keep.
    keep: float = 0.625
    mass_target: float = 0.90
    query_chunk: int = 256
    # reuse: previous mask; directional: independent one-step Q/K directions;
    # all: at most eight coupled Q/K spatial neighbors.
    policy: str = "reuse"
    centroid_threshold: float = 0.15
    # Directional routes discard blocks below this fraction of the head's
    # average selected mass, while preserving at least one K block per Q row.
    drop_factor: float = 0.1
    huge_factor: float = 2.0


@dataclass(frozen=True)
class _SpatialTileLayout:
    """An exact per-frame partition of Wan's flattened (F, H, W) token grid."""

    frames: int
    height: int
    width: int
    tile_h: int
    tile_w: int

    @property
    def tokens(self):
        return self.frames * self.height * self.width

    @property
    def tile_tokens(self):
        return self.tile_h * self.tile_w

    @property
    def tile_count(self):
        return self.tokens // self.tile_tokens

    @property
    def block_h(self):
        return self.height // self.tile_h

    @property
    def block_w(self):
        return self.width // self.tile_w

    def to_tile_major(self, tensor):
        """Move each spatial HxW block next to itself on the sequence axis."""
        batch, tokens, heads, dim = tensor.shape
        if tokens != self.tokens:
            raise ValueError(f"Grid has {self.tokens} tokens, tensor has {tokens}")
        block_h = self.height // self.tile_h
        block_w = self.width // self.tile_w
        tensor = tensor.reshape(
            batch, self.frames, block_h, self.tile_h,
            block_w, self.tile_w, heads, dim)
        return tensor.permute(0, 1, 2, 4, 3, 5, 6, 7).reshape(
            batch, tokens, heads, dim)

    def from_tile_major(self, tensor):
        """Restore Wan's original W-fastest flattened token order."""
        batch, tokens, heads, dim = tensor.shape
        block_h = self.height // self.tile_h
        block_w = self.width // self.tile_w
        tensor = tensor.reshape(
            batch, self.frames, block_h, block_w,
            self.tile_h, self.tile_w, heads, dim)
        return tensor.permute(0, 1, 2, 4, 3, 5, 6, 7).reshape(
            batch, tokens, heads, dim)


def _divisors(value):
    small = [candidate for candidate in range(1, math.isqrt(value) + 1)
             if value % candidate == 0]
    return small + [value // candidate for candidate in reversed(small)
                    if candidate * candidate != value]


def _spatial_layout(grid, target_tile_tokens, tokens):
    """Choose a near-target spatial tile whose sides exactly divide H and W."""
    if grid is None or len(grid) != 3 or target_tile_tokens <= 0:
        return None
    frames, height, width = (int(value) for value in grid)
    if frames * height * width != tokens:
        # A padded or sequence-parallel Wan layout cannot be spatially reordered
        # without additional ownership metadata, so it falls back to dense.
        return None
    candidates = [
        (tile_h, tile_w)
        for tile_h in _divisors(height)
        for tile_w in _divisors(width)
    ]
    tile_h, tile_w = min(
        candidates,
        key=lambda shape: (
            abs(math.log(
                (shape[0] * shape[1]) / target_tile_tokens))
            + 0.25 * abs(math.log(shape[1] / shape[0])),
            abs(shape[0] * shape[1] - target_tile_tokens),
            -shape[0] * shape[1],
        ),
    )
    return _SpatialTileLayout(
        frames, height, width, tile_h, tile_w)


def _routing_context():
    model = sys.modules.get("wan.modules.model")
    return (
        getattr(model, "_CURRENT_ATTN_ID", -1),
        getattr(model, "_CURRENT_DENOISE_STEP", -1),
        getattr(model, "_CURRENT_CFG_BRANCH", -1),
        getattr(model, "_CURRENT_GRID_SIZE", None),
    )


def _axis_offsets(length, device):
    """Return normalized coordinates in [-1, 1] for one tile axis."""
    positions = torch.arange(length, device=device, dtype=torch.float32)
    center = (length - 1) / 2
    scale = max((length - 1) / 2, .5)
    return (positions - center) / scale


def _spatial_offsets(layout, device):
    """Return tile-major y/x coordinates for every token in the full grid."""
    local_y, local_x = torch.meshgrid(
        _axis_offsets(layout.tile_h, device),
        _axis_offsets(layout.tile_w, device),
        indexing="ij",
    )
    repeats = layout.tile_count
    return local_y.flatten().repeat(repeats), local_x.flatten().repeat(repeats)


def _empty_route_stats(heads, tiles, device):
    mass = torch.zeros((heads, tiles, tiles), device=device)
    return {
        "mass": mass,
        "q_y": torch.zeros_like(mass),
        "q_x": torch.zeros_like(mass),
        "k_y": torch.zeros_like(mass),
        "k_x": torch.zeros_like(mass),
    }


def _row_mask(mass, target, keep):
    """Select key tiles independently for every (head, query-tile) row."""
    order = mass.argsort(-1, descending=True)
    sorted_mass = torch.gather(mass, -1, order)
    threshold = mass.sum(-1, keepdim=True) * target
    counts = (torch.cumsum(sorted_mass, -1) < threshold).sum(-1) + 1
    counts = counts.clamp(max=max(1, math.ceil(mass.shape[-1] * keep)))
    ranks = torch.empty_like(order)
    rank_values = torch.arange(order.shape[-1], device=mass.device).expand_as(order)
    ranks.scatter_(-1, order, rank_values)
    return ranks < counts.unsqueeze(-1)


@torch.no_grad()
def _dense_route(q, k, v, layout, query_chunk, target, keep, scale=None):
    """Measure an exact dense route and produce the first-step output together."""
    tokens, heads, head_dim = q.shape[1:]
    tile = layout.tile_tokens
    if tokens % tile:
        raise ValueError("Spatial tile layout must exactly divide the token count")
    tiles = tokens // tile
    stats = _empty_route_stats(heads, tiles, q.device)
    token_y, token_x = _spatial_offsets(layout, q.device)
    # K participates in every query chunk. Convert it once rather than allocate
    # a full float32 copy on every iteration.
    k_float = k.float()
    v_float = v[0].permute(1, 0, 2).float()
    score_scale = scale if scale is not None else head_dim**-0.5
    output = q.new_empty((q.shape[0], tokens, heads, v.shape[-1]))
    query_chunk = max(tile, query_chunk // tile * tile)
    for start in range(0, tokens, query_chunk):
        end = min(start + query_chunk, tokens)
        scores = torch.einsum(
            "bqhd,bkhd->bhqk", q[:, start:end].float(), k_float)
        probabilities = (scores * score_scale).softmax(-1)[0]
        # The probabilities are already materialized for route statistics, so
        # use them for PV instead of running a second dense SDPA afterward.
        chunk_output = probabilities.matmul(v_float)
        output[:, start:end] = chunk_output.permute(
            1, 0, 2).unsqueeze(0).to(q.dtype)
        query_tiles = (end - start) // tile
        shape = (heads, query_tiles, tile, tiles, tile)
        destination = slice(start // tile, start // tile + query_tiles)
        stats["mass"][:, destination] = probabilities.reshape(shape).sum((2, 4))
        for axis, offsets in (("y", token_y), ("x", token_x)):
            stats[f"q_{axis}"][:, destination] = (
                probabilities * offsets[start:end].view(1, -1, 1)
            ).reshape(shape).sum((2, 4))
            stats[f"k_{axis}"][:, destination] = (
                probabilities * offsets.view(1, 1, -1)
            ).reshape(shape).sum((2, 4))
    normalized = stats["mass"] / tokens
    denominator = stats["mass"].clamp_min(1e-12)
    route = {
        "mask": _row_mask(normalized, target, keep).cpu(),
        "mass": normalized.to(torch.float16).cpu(),
        "centroid_q_y": (stats["q_y"] / denominator).to(torch.float16).cpu(),
        "centroid_q_x": (stats["q_x"] / denominator).to(torch.float16).cpu(),
        "centroid_k_y": (stats["k_y"] / denominator).to(torch.float16).cpu(),
        "centroid_k_x": (stats["k_x"] / denominator).to(torch.float16).cpu(),
        "tile": tile,
    }
    return route, output


def _route_grid(tensor, layout):
    """View flat Q/K tile IDs as independent per-frame spatial block grids."""
    shape = (
        tensor.shape[0],
        layout.frames, layout.block_h, layout.block_w,
        layout.frames, layout.block_h, layout.block_w,
    )
    return tensor.reshape(shape)


def _shift_spatial_dimension(mask, dimension, delta):
    """Shift one spatial axis without wrapping across rows or frames."""
    output = torch.zeros_like(mask)
    source = [slice(None)] * mask.ndim
    target = [slice(None)] * mask.ndim
    size = mask.shape[dimension]
    source[dimension] = slice(max(0, -delta), size - max(0, delta))
    target[dimension] = slice(max(0, delta), size - max(0, -delta))
    output[tuple(target)] = mask[tuple(source)]
    return output


def _spatial_neighbors(mask):
    """Add at most eight coupled Q/K spatial neighbors per attention block.

    A neighbor uses the same (dy, dx) for the Q and K block coordinates. This
    preserves a two-dimensional eight-neighborhood instead of forming the
    3x3-Q by 3x3-K Cartesian product, which would add up to 80 blocks.
    """
    expanded = mask.clone()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = mask
            for dimension, delta in ((2, dy), (3, dx), (5, dy), (6, dx)):
                if delta:
                    shifted = _shift_spatial_dimension(
                        shifted, dimension, delta)
            expanded |= shifted
    return expanded


def _centroid_direction(route, name, layout, threshold):
    centroid = _route_grid(route[name], layout)
    return torch.where(
        centroid.abs() < threshold, 0, torch.sign(centroid)).long()


def _apply_directional_steps(predicted, directional, route, layout, threshold):
    """Add one Q-only, K-only, and joint Q/K step for every routed block."""
    source = torch.where(directional)
    if source[0].numel() == 0:
        return
    directions = {
        "q_y": _centroid_direction(
            route, "centroid_q_y", layout, threshold)[source],
        "q_x": _centroid_direction(
            route, "centroid_q_x", layout, threshold)[source],
        "k_y": _centroid_direction(
            route, "centroid_k_y", layout, threshold)[source],
        "k_x": _centroid_direction(
            route, "centroid_k_x", layout, threshold)[source],
    }
    modes = (
        (True, False),   # Move Q while keeping K fixed.
        (False, True),   # Move K while keeping Q fixed.
        (True, True),    # Move Q and K together.
    )
    for move_q, move_k in modes:
        target = list(source)
        valid = torch.ones_like(source[0], dtype=torch.bool)
        moved = torch.zeros_like(source[0], dtype=torch.bool)
        dimensions = []
        if move_q:
            dimensions.extend(((2, "q_y", layout.block_h),
                               (3, "q_x", layout.block_w)))
        if move_k:
            dimensions.extend(((5, "k_y", layout.block_h),
                               (6, "k_x", layout.block_w)))
        for dimension, name, size in dimensions:
            direction = directions[name]
            target[dimension] = source[dimension] + direction
            valid &= (target[dimension] >= 0) & (target[dimension] < size)
            moved |= direction != 0
        valid &= moved
        if not valid.any():
            continue
        predicted[tuple(index[valid] for index in target)] = True


def _preserve_one_per_row(kept, original, mass):
    """Restore the strongest original K block when dropping empties a Q row."""
    empty = ~kept.any(-1) & original.any(-1)
    if not empty.any():
        return kept
    fallback_scores = mass.masked_fill(~original, -torch.inf)
    fallback_keys = fallback_scores.argmax(-1)
    heads, query_tiles = torch.where(empty)
    kept[heads, query_tiles, fallback_keys[heads, query_tiles]] = True
    return kept


def _drop_low_mass(mask, mass, drop_factor):
    """Shrink a route only by dropping low-mass blocks; never re-rank/select."""
    if drop_factor <= 0:
        return mask
    selected_mass = torch.where(mask, mass, 0)
    mean_mass = selected_mass.sum((1, 2), keepdim=True) / mask.sum(
        (1, 2), keepdim=True).clamp_min(1)
    kept = mask & (mass / mean_mass.clamp_min(1e-12) >= drop_factor)
    return _preserve_one_per_row(kept, mask, mass)


def _expand(route, policy, centroid_threshold, huge_factor, layout):
    """Predict the next-step mask from the previous step's tile statistics."""
    mask = route["mask"]
    if policy == "reuse":
        # The caller immediately copies the mask to the accelerator and never
        # mutates the CPU route, so an intermediate CPU clone is unnecessary.
        return mask
    spatial_mask = _route_grid(mask, layout)
    if policy == "all":
        return _spatial_neighbors(spatial_mask).reshape_as(mask)
    mass = route["mass"].float()
    selected_mass = torch.where(mask, mass, 0)
    mean_mass = selected_mass.sum((1, 2), keepdim=True) / mask.sum(
        (1, 2), keepdim=True).clamp_min(1)
    ratio = mass / mean_mass.clamp_min(1e-12)
    huge = mask & (ratio >= huge_factor)
    all_mask = _route_grid(huge, layout)
    predicted = spatial_mask | _spatial_neighbors(all_mask)
    # Huge blocks already receive the full eight-neighborhood. Other retained
    # blocks may add up to three one-step targets, with no global count cap.
    directional_grid = _route_grid(mask & ~huge, layout)
    _apply_directional_steps(
        predicted, directional_grid, route, layout, centroid_threshold)
    return predicted.reshape_as(mask)


def _group_indices(group_mask, tile, tokens):
    """Expand per-group tile masks into padded token indices on the device.

    The old implementation iterated over every (query tile, head) pair in
    Python and fed a list of variable-length tensors to ``pad_sequence``.
    Building the same layout in a few tensor operations removes that hot-loop
    overhead and reduces device synchronization to one maximum-size read.
    """
    selected_counts = group_mask.sum(-1)
    max_selected = int(selected_counts.max().item())
    if max_selected == 0:
        raise ValueError("Every sparse attention group must select at least one key tile")

    # Number selected tiles within each row from left to right. These ranks are
    # used as compact columns in the padded tile-id matrix.
    selected_rows, tile_ids = torch.where(group_mask)
    selected_ranks = group_mask.cumsum(-1)[selected_rows, tile_ids] - 1
    padded_tiles = torch.zeros(
        (group_mask.shape[0], max_selected), dtype=torch.long,
        device=group_mask.device)
    padded_tiles[selected_rows, selected_ranks] = tile_ids

    valid_tiles = torch.arange(max_selected, device=group_mask.device)[None] < (
        selected_counts[:, None])
    token_in_tile = torch.arange(tile, device=group_mask.device)
    indices = padded_tiles[:, :, None] * tile + token_in_tile
    valid_keys = valid_tiles[:, :, None].expand_as(indices)
    # Gather happens before logits are masked, so batch padding still needs an
    # in-range placeholder even though spatial tiles themselves are exact.
    indices.masked_fill_(~valid_keys, 0)
    return indices.flatten(1), valid_keys.flatten(1)


def sparse_chunk(q, k, v, route_mask, tile, start, end, scale,
                 stats, token_offsets=None):
    """Gather all selected keys for one aligned query chunk and run attention.

    Each (query tile, head) is a separate batch item with its own gathered K/V
    sequence. Softmax therefore remains independent per query and head.
    """
    tokens, heads, head_dim = q.shape[1:]
    if start % tile or end % tile:
        raise ValueError("Query chunks must align to exact spatial tiles")
    first_tile = start // tile
    query_tiles = (end - start) // tile
    q_chunk = q[:, start:end]
    grouped_q = q_chunk[0].reshape(query_tiles, tile, heads, head_dim)
    grouped_q = grouped_q.permute(0, 2, 1, 3).reshape(-1, tile, head_dim)
    group_heads = torch.arange(heads, device=q.device).repeat(query_tiles)
    group_mask = route_mask[:, first_tile:first_tile + query_tiles]
    group_mask = group_mask.permute(1, 0, 2).reshape(query_tiles * heads, -1)
    padded_indices, valid_keys = _group_indices(group_mask, tile, tokens)
    selected_k = k[0, padded_indices, group_heads[:, None]].to(q.dtype)
    selected_v = v[0, padded_indices, group_heads[:, None]].to(q.dtype)
    logits = torch.matmul(grouped_q, selected_k.transpose(1, 2)) * (
        scale if scale is not None else head_dim**-.5)
    logits = logits.masked_fill(
        ~valid_keys[:, None], torch.finfo(logits.dtype).min)
    probabilities = torch.softmax(logits, -1)
    grouped_output = probabilities.matmul(selected_v)
    output = grouped_output.reshape(query_tiles, heads, tile, -1)
    output = output.permute(0, 2, 1, 3).reshape(
        1, end - start, heads, -1)

    key_mass = probabilities.sum(1).float()
    key_tiles = padded_indices // tile
    groups = query_tiles * heads
    group_mass = torch.zeros(
        (groups, stats["mass"].shape[-1]), device=q.device)
    group_mass.scatter_add_(1, key_tiles, key_mass * valid_keys)
    if token_offsets is None:
        # The standalone kernel benchmark has no Wan grid. Treat its tile as a
        # single spatial row; inference always supplies real y/x coordinates.
        token_offsets = (
            torch.zeros(tokens, device=q.device),
            _axis_offsets(tile, q.device).repeat(tokens // tile),
        )
    shape = (query_tiles, heads, -1)
    destination = slice(first_tile, first_tile + query_tiles)
    stats["mass"][:, destination] = group_mass.reshape(shape).permute(1, 0, 2)
    for axis, offsets in zip(("y", "x"), token_offsets):
        local_q = offsets[start:end].reshape(query_tiles, tile)
        local_q = local_q[:, None].expand(
            -1, heads, -1).reshape(-1, tile)
        q_key_mass = (
            probabilities.float() * local_q[:, :, None]).sum(1)
        group_q_moment = torch.zeros_like(group_mass)
        group_q_moment.scatter_add_(
            1, key_tiles, q_key_mass * valid_keys)
        local_k = offsets[padded_indices]
        group_k_moment = torch.zeros_like(group_mass)
        group_k_moment.scatter_add_(
            1, key_tiles, key_mass * local_k * valid_keys)
        stats[f"q_{axis}"][:, destination] = (
            group_q_moment.reshape(shape).permute(1, 0, 2))
        stats[f"k_{axis}"][:, destination] = (
            group_k_moment.reshape(shape).permute(1, 0, 2))
    return output


class MatrixSparseBackend:
    """Reuse the previous denoising step's matrix-tile route for self-attention."""

    name = "sparse"

    def __init__(self, config=None):
        self.config = config or SparseConfig()
        self.dense = DenseBackend()
        self.state = {}

    def _sparse_attention(self, q, k, v, scale, layout):
        # RoPE was applied in Wan's original order. Jointly permuting Q/K/V is
        # attention-equivalent and only changes which tokens form a tile.
        q = layout.to_tile_major(q)
        k = layout.to_tile_major(k)
        v = layout.to_tile_major(v)
        tokens, heads = q.shape[1:3]
        attention_id, step, branch, _ = _routing_context()
        # Layers and CFG branches follow different attention trajectories and
        # must never share a route, even when their tensor shapes match.
        key = (attention_id, branch, heads, layout)
        tile = layout.tile_tokens
        if step == 0 or key not in self.state:
            # The first observation is dense so later steps have an exact route.
            # Route measurement also returns PV, avoiding a duplicate dense SDPA.
            route, output = _dense_route(
                q, k, v, layout, self.config.query_chunk,
                self.config.mass_target, self.config.keep, scale)
            route["grid"] = (layout.frames, layout.height, layout.width)
            route["spatial_tile"] = (layout.tile_h, layout.tile_w)
            self.state[key] = route
            return layout.from_tile_major(output)
        route = _expand(
            self.state[key], self.config.policy, self.config.centroid_threshold,
            self.config.huge_factor, layout).to(q.device)
        tiles = route.shape[-1]
        stats = _empty_route_stats(heads, tiles, q.device)
        # Write chunks directly into their final positions. Keeping a list and
        # concatenating it would require another full output-sized allocation.
        output = q.new_empty((q.shape[0], tokens, heads, v.shape[-1]))
        # Tile-local coordinates are invariant across chunks and are used for
        # both Q and K centroid updates.
        token_offsets = _spatial_offsets(layout, q.device)
        chunk = max(
            tile,
            self.config.query_chunk // tile * tile)
        for start in range(0, tokens, chunk):
            end = min(start + chunk, tokens)
            output[:, start:end] = sparse_chunk(
                q, k, v, route, tile, start, end, scale,
                stats, token_offsets)
        normalized = stats["mass"] / tokens
        denominator = stats["mass"].clamp_min(1e-12)
        # The first dense step uses top-mass selection to bootstrap the route.
        # Later steps retain the route actually executed this step and contract
        # it only through the explicit low-mass drop policy.
        next_mask = _drop_low_mass(
            route, normalized, self.config.drop_factor)
        # Keep the large, persistent route on CPU. Only the predicted boolean
        # mask is transferred back to the accelerator on the next step.
        self.state[key] = {
            "mask": next_mask.cpu(),
            "mass": normalized.to(torch.float16).cpu(),
            "centroid_q_y": (
                stats["q_y"] / denominator).to(torch.float16).cpu(),
            "centroid_q_x": (
                stats["q_x"] / denominator).to(torch.float16).cpu(),
            "centroid_k_y": (
                stats["k_y"] / denominator).to(torch.float16).cpu(),
            "centroid_k_x": (
                stats["k_x"] / denominator).to(torch.float16).cpu(),
            "tile": tile,
            "grid": (layout.frames, layout.height, layout.width),
            "spatial_tile": (layout.tile_h, layout.tile_w),
        }
        return layout.from_tile_major(output)

    def __call__(self, q, k, v, softmax_scale=None, **kwargs):
        if q.shape[1] == k.shape[1] and q.shape[0] == 1:
            _, _, _, grid = _routing_context()
            layout = _spatial_layout(grid, self.config.tile, q.shape[1])
            if layout is not None:
                return self._sparse_attention(
                    q, k, v, softmax_scale, layout)
        return self.dense(q, k, v, softmax_scale=softmax_scale, **kwargs)


def install_sparse(config=None):
    return install(MatrixSparseBackend(config))
