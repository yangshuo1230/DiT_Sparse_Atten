"""Frontier-aware directional updates for spatial FlexAttention routes."""

from __future__ import annotations

import math

import torch


def _neighbor_values(tensor, neighbors, axis):
    """Gather [H,Q,K] values at each source block's four neighbours."""
    heads, blocks, _ = tensor.shape
    values = []
    valid_values = []
    for direction in range(neighbors.shape[1]):
        target = neighbors[:, direction]
        valid = target >= 0
        if axis == 1:
            indices = target.clamp_min(0).view(1, blocks, 1).expand(
                heads, blocks, blocks)
            gathered = torch.gather(tensor, 1, indices)
            valid_mask = valid.view(1, blocks, 1)
        elif axis == 2:
            indices = target.clamp_min(0).view(1, 1, blocks).expand(
                heads, blocks, blocks)
            gathered = torch.gather(tensor, 2, indices)
            valid_mask = valid.view(1, 1, blocks)
        else:
            raise ValueError("axis must be the Q axis (1) or K axis (2)")
        values.append(gathered)
        valid_values.append(valid_mask.expand_as(tensor))
    return torch.stack(values, -1), torch.stack(valid_values, -1)


def frontier_mask(route, neighbors):
    """Return selected route cells adjacent to an unselected spatial cell."""
    q_selected, q_valid = _neighbor_values(route, neighbors, axis=1)
    k_selected, k_valid = _neighbor_values(route, neighbors, axis=2)
    q_frontier = route & (q_valid & ~q_selected).any(-1)
    k_frontier = route & (k_valid & ~k_selected).any(-1)
    return q_frontier | k_frontier


def _scatter_points(heads, query, key, shape):
    output = torch.zeros(
        (shape[0], shape[1] * shape[2]),
        device=heads.device, dtype=torch.bool)
    flat = query * shape[2] + key
    output[heads, flat] = True
    return output.reshape(shape)


def _sparse_best_targets(mass, route, neighbors, frontier, axis, min_ratio):
    """Choose one neighbour only for cells in the maintained frontier mask."""
    heads, query, key = torch.where(frontier)
    if not heads.numel():
        empty = torch.empty(0, device=mass.device, dtype=torch.long)
        return torch.zeros_like(route), (heads, query, key, empty, empty.bool())
    source = query if axis == 1 else key
    targets = neighbors[source].long()
    valid = targets >= 0
    safe_targets = targets.clamp_min(0)
    if axis == 1:
        selected = route[heads[:, None], safe_targets, key[:, None]]
        values = mass[heads[:, None], safe_targets, key[:, None]]
    else:
        selected = route[heads[:, None], query[:, None], safe_targets]
        values = mass[heads[:, None], query[:, None], safe_targets]
    values = values.masked_fill(~valid | selected, -torch.inf)
    best_values, directions = values.max(-1)
    best_targets = safe_targets.gather(1, directions[:, None]).squeeze(1)
    eligible = torch.isfinite(best_values) & (
        best_values >= mass[heads, query, key] * min_ratio)
    if axis == 1:
        candidates = _scatter_points(
            heads[eligible], best_targets[eligible], key[eligible], route.shape)
    else:
        candidates = _scatter_points(
            heads[eligible], query[eligible], best_targets[eligible], route.shape)
    return candidates, (heads, query, key, best_targets, eligible)


def _select_fixed_budget(score, pool, counts):
    order = score.masked_fill(~pool, -torch.inf).argsort(-1, descending=True)
    ranks = torch.empty_like(order)
    values = torch.arange(order.shape[-1], device=order.device).expand_as(order)
    ranks.scatter_(-1, order, values)
    counts = torch.minimum(counts, pool.sum(-1)).clamp_min(1)
    return (ranks < counts.unsqueeze(-1)) & pool


def directional_budget_update(
    mass,
    previous,
    neighbors,
    *,
    keep,
    persistence=0.5,
    min_ratio=0.0,
    candidate_bonus=0.0,
    budget_scale=1.0,
    exploration_fraction=0.0,
    expand_q=True,
    expand_k=True,
    expand_joint=True,
    previous_frontier=None,
):
    """Expand only route frontiers, then prune back to a per-row budget."""
    if previous.shape != mass.shape:
        raise ValueError("previous route and mass must have identical shapes")
    if neighbors.shape != (mass.shape[-1], 4):
        raise ValueError("neighbors must have shape [blocks, 4]")
    if budget_scale <= 0:
        raise ValueError("budget_scale must be positive")
    if not 0 <= exploration_fraction <= 1:
        raise ValueError("exploration_fraction must be in [0, 1]")

    source_frontier = (previous_frontier if previous_frontier is not None
                       else frontier_mask(previous, neighbors))
    q_candidates, q_values = _sparse_best_targets(
        mass, previous, neighbors, source_frontier, axis=1,
        min_ratio=min_ratio)
    k_candidates, k_values = _sparse_best_targets(
        mass, previous, neighbors, source_frontier, axis=2,
        min_ratio=min_ratio)
    candidates = torch.zeros_like(previous)
    if expand_q:
        candidates |= q_candidates
    if expand_k:
        candidates |= k_candidates
    if expand_joint:
        heads, _query, _key, q_targets, q_eligible = q_values
        _, _, _, k_targets, k_eligible = k_values
        joint = q_eligible & k_eligible
        candidates |= _scatter_points(
            heads[joint], q_targets[joint], k_targets[joint], previous.shape)
    candidates &= ~previous

    pool = previous | candidates
    if exploration_fraction:
        explore_count = max(1, math.ceil(
            mass.shape[-1] * exploration_fraction))
        explore_order = mass.masked_fill(pool, -torch.inf).argsort(
            -1, descending=True)
        exploration = torch.zeros_like(pool)
        exploration.scatter_(
            -1, explore_order[..., :explore_count], True)
        pool |= exploration
    else:
        exploration = torch.zeros_like(pool)

    row_scale = mass.mean(-1, keepdim=True)
    score = mass + persistence * row_scale * previous.float()
    if candidate_bonus:
        score = score + candidate_bonus * row_scale * candidates.float()
    cap = max(1, math.ceil(mass.shape[-1] * keep))
    counts = torch.ceil(
        previous.sum(-1).float() * budget_scale).to(torch.long)
    counts.clamp_(min=1, max=cap)
    route = _select_fixed_budget(score, pool, counts)
    return {
        "route": route,
        "frontier": frontier_mask(route, neighbors),
        "candidates": candidates,
        "exploration": exploration,
        "dropped": previous & ~route,
        "added": route & ~previous,
    }
