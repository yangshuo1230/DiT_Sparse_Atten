"""Sampled single-query sparsity ceilings for real Wan self-attention."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


def _quantiles(values):
    ordered = sorted(values)
    if not ordered:
        return {}
    return {
        f"p{int(q * 100)}": ordered[min(len(ordered) - 1, int(q * len(ordered)))]
        for q in (0.1, 0.5, 0.9)
    }


def _summary(values):
    return {
        "mean": sum(values) / len(values) if values else None,
        "quantiles": _quantiles(values),
        "observations": len(values),
    }


def _pool_linear(probabilities, tile):
    tokens = probabilities.shape[-1]
    tiles = math.ceil(tokens / tile)
    pooled = F.pad(probabilities, (0, tiles * tile - tokens))
    pooled = pooled.reshape(*probabilities.shape[:-1], tiles, tile).sum(-1)
    sizes = torch.full((tiles,), tile, device=probabilities.device)
    sizes[-1] = tokens - (tiles - 1) * tile
    return pooled, sizes


def _pool_spatial(probabilities, grid, tile_h, tile_w):
    frames, height, width = grid
    rows, cols = math.ceil(height / tile_h), math.ceil(width / tile_w)
    values = probabilities.reshape(*probabilities.shape[:-1], frames, height, width)
    values = F.pad(values, (0, cols * tile_w - width, 0, rows * tile_h - height))
    shape = (*values.shape[:-2], rows, tile_h, cols, tile_w)
    pooled = values.reshape(shape).sum((-1, -3)).flatten(-3)

    y_sizes = torch.full((rows,), tile_h, device=probabilities.device)
    x_sizes = torch.full((cols,), tile_w, device=probabilities.device)
    y_sizes[-1] = height - (rows - 1) * tile_h
    x_sizes[-1] = width - (cols - 1) * tile_w
    sizes = (y_sizes[:, None] * x_sizes[None, :]).flatten().repeat(frames)
    return pooled, sizes


def _select(mass, target):
    order = mass.argsort(-1, descending=True)
    sorted_mass = torch.gather(mass, -1, order)
    counts = (torch.cumsum(sorted_mass, -1) < target).sum(-1) + 1
    ranks = torch.empty_like(order)
    rank_values = torch.arange(order.shape[-1], device=order.device).expand_as(order)
    ranks.scatter_(-1, order, rank_values)
    return ranks < counts.unsqueeze(-1), counts, sorted_mass


def _route_metrics(mass, sizes, target, padded_tile_size):
    selected, counts, _ = _select(mass, target)
    tokens = int(sizes.sum())
    useful = (selected * sizes).sum(-1).float() / tokens
    padded = counts.float() * padded_tile_size / tokens
    return {
        "tile_keep": (counts.float() / mass.shape[-1]).flatten().cpu().tolist(),
        "useful_key_fraction": useful.flatten().cpu().tolist(),
        "padded_matmul_fraction": padded.flatten().cpu().tolist(),
    }


def _query_indices(grid, layer, branch, device):
    frames, height, width = grid
    full_rows, full_cols = height // 4, width // 4
    blocks = []
    for offset in (0, 1):
        frame = (layer + branch + offset) % frames
        tile_y = (layer * 3 + branch * 5 + offset * max(1, full_rows // 2)) % full_rows
        tile_x = (layer * 7 + branch * 3 + offset * max(1, full_cols // 2)) % full_cols
        y0, x0 = tile_y * 4, tile_x * 4
        blocks.extend(
            (frame * height + y) * width + x
            for y in range(y0, y0 + 4)
            for x in range(x0, x0 + 4)
        )
    return torch.tensor(blocks, device=device, dtype=torch.long)


class SingleQuerySpatialProbe:
    def __init__(self, path: Path, mass_target=0.90):
        self.path = Path(path)
        self.mass_target = mass_target
        self.min_steps = int(os.getenv("WAN_PROBE_MIN_STEPS", "5"))
        self.step = -1
        self.timestep = None
        self.branch = -1
        self.calls = 0
        self.cross_calls = 0
        self.grid = None
        self.snapshot = None
        self.snapshot_distance = float("inf")
        self.snapshot_target_keep = float(os.getenv("WAN_SNAPSHOT_TARGET_KEEP", ".39"))
        self.snapshot_target_same_frame = float(
            os.getenv("WAN_SNAPSHOT_TARGET_SAME_FRAME", ".59"))
        self.values = defaultdict(list)
        self.step_values = defaultdict(lambda: defaultdict(list))
        self.layer_head_spatial_keep = defaultdict(lambda: defaultdict(list))

    def begin_forward(self, timestep):
        value = float(timestep.detach().flatten()[0].item())
        if self.timestep is None or not math.isclose(value, self.timestep):
            self.step += 1
            self.timestep = value
            self.branch = 0
        else:
            self.branch += 1

    def _add(self, name, values):
        self.values[name].extend(values)
        self.step_values[self.step][name].extend(values)

    @torch.no_grad()
    def record(self, q, k, attention_id, grid):
        if q.shape[1] != k.shape[1]:
            self.cross_calls += 1
            return
        if grid is None or math.prod(grid) != q.shape[1]:
            raise RuntimeError(f"Invalid token grid {grid} for sequence length {q.shape[1]}")
        self.grid = tuple(int(value) for value in grid)
        layer = int(attention_id) // 2
        indices = _query_indices(self.grid, layer, self.branch, q.device)
        scores = torch.einsum("bqhd,bkhd->bhqk", q[:, indices].float(), k.float())
        probabilities = (scores / math.sqrt(q.shape[-1])).softmax(-1)[0]
        heads, queries, tokens = probabilities.shape

        token_mask, token_counts, sorted_tokens = _select(probabilities, self.mass_target)
        self._add("token_oracle_keep", (token_counts.float() / tokens).flatten().cpu().tolist())

        linear, linear_sizes = _pool_linear(probabilities, 16)
        spatial2, spatial2_sizes = _pool_spatial(probabilities, self.grid, 2, 2)
        spatial4, spatial4_sizes = _pool_spatial(probabilities, self.grid, 4, 4)
        spatial2x4, spatial2x4_sizes = _pool_spatial(probabilities, self.grid, 2, 4)
        for name, mass, sizes, padded in (
            ("linear16", linear, linear_sizes, 16),
            ("spatial2x2_q1", spatial2, spatial2_sizes, 4),
            ("spatial4x4_q1", spatial4, spatial4_sizes, 16),
            ("spatial2x4_q1", spatial2x4, spatial2x4_sizes, 8),
        ):
            for metric, values in _route_metrics(
                    mass, sizes, self.mass_target, padded).items():
                self._add(f"{name}_{metric}", values)

        # The 32 sampled queries are two aligned 4x4 spatial blocks. Compare
        # one route per q, per spatial row (4 q), and per full block (16 q).
        for group_size in (4, 16):
            grouped = spatial4.reshape(heads, queries // group_size, group_size, -1).mean(2)
            metrics = _route_metrics(grouped, spatial4_sizes, self.mass_target, 16)
            for metric, values in metrics.items():
                self._add(f"spatial4x4_q{group_size}_{metric}", values)

        # Form exact 2x2 query blocks within each sampled 4x4 block. The
        # row-major query sample needs an explicit gather rather than reshape.
        local_2x2 = []
        for base in (0, 16):
            for y0 in (0, 2):
                for x0 in (0, 2):
                    local_2x2.append([
                        base + (y0 + dy) * 4 + x0 + dx
                        for dy in (0, 1) for dx in (0, 1)
                    ])
        query_groups = torch.tensor(local_2x2, device=q.device)
        grouped_2x2 = spatial2[:, query_groups, :].mean(2)
        for metric, values in _route_metrics(
                grouped_2x2, spatial2_sizes, self.mass_target, 4).items():
            self._add(f"spatial2x2_q4_{metric}", values)
        selected_2x2, counts_2x2, _ = _select(grouped_2x2, self.mass_target)
        keeps_2x2 = counts_2x2.float() / grouped_2x2.shape[-1]
        group_query_tokens = indices[query_groups]
        group_frames = group_query_tokens[:, 0] // (self.grid[1] * self.grid[2])
        grouped_frame_mass = grouped_2x2.reshape(
            heads, grouped_2x2.shape[1], self.grid[0], -1).sum(-1)
        same_frame_mass_2x2 = torch.gather(
            grouped_frame_mass, 2,
            group_frames.view(1, -1, 1).expand(heads, -1, -1)).squeeze(-1)
        distance_2x2 = (
            (keeps_2x2 - self.snapshot_target_keep).abs() +
            (same_frame_mass_2x2 - self.snapshot_target_same_frame).abs()
        )
        flat_candidate = int(distance_2x2.argmin())
        candidate_distance = float(distance_2x2.flatten()[flat_candidate])
        if candidate_distance < self.snapshot_distance:
            candidate_head = flat_candidate // grouped_2x2.shape[1]
            candidate_group = flat_candidate % grouped_2x2.shape[1]
            candidate_queries = query_groups[candidate_group]
            query_tokens = indices[candidate_queries]
            query_frames = query_tokens // (self.grid[1] * self.grid[2])
            query_remainder = query_tokens % (self.grid[1] * self.grid[2])
            self.snapshot_distance = candidate_distance
            self.snapshot = {
                "step": self.step,
                "cfg_branch": self.branch,
                "layer": layer,
                "head": candidate_head,
                "query_token_indices": query_tokens.cpu().tolist(),
                "query_coordinates_tyx": torch.stack((
                    query_frames,
                    query_remainder // self.grid[2],
                    query_remainder % self.grid[2],
                ), -1).cpu().tolist(),
                "key_block_shape": [1, 2, 2],
                "key_block_grid": [self.grid[0], self.grid[1] // 2, self.grid[2] // 2],
                "route_keep_fraction": float(keeps_2x2[candidate_head, candidate_group]),
                "same_frame_mass": float(same_frame_mass_2x2[
                    candidate_head, candidate_group]),
                "route_mass_covered": float((
                    grouped_2x2[candidate_head, candidate_group] *
                    selected_2x2[candidate_head, candidate_group]).sum()),
                "individual_query_block_mass": spatial2[
                    candidate_head, candidate_queries].cpu().tolist(),
                "aggregate_query_block_mass": grouped_2x2[
                    candidate_head, candidate_group].cpu().tolist(),
                "selected_key_blocks": selected_2x2[
                    candidate_head, candidate_group].cpu().tolist(),
            }

        entropy = -(probabilities.clamp_min(1e-30).log() * probabilities).sum(-1)
        self._add("normalized_entropy", (entropy / math.log(tokens)).flatten().cpu().tolist())
        self._add("effective_token_fraction", (entropy.exp() / tokens).flatten().cpu().tolist())
        top1_count = max(1, math.ceil(tokens * .01))
        top10_count = max(1, math.ceil(tokens * .10))
        self._add("top1pct_mass", sorted_tokens[..., :top1_count].sum(-1).flatten().cpu().tolist())
        self._add("top10pct_mass", sorted_tokens[..., :top10_count].sum(-1).flatten().cpu().tolist())

        frames, height, width = self.grid
        key_indices = torch.arange(tokens, device=q.device)
        key_frames = key_indices // (height * width)
        key_remainder = key_indices % (height * width)
        key_y, key_x = key_remainder // width, key_remainder % width
        query_frames = indices // (height * width)
        query_remainder = indices % (height * width)
        query_y, query_x = query_remainder // width, query_remainder % width
        same_frame = key_frames.view(1, 1, -1) == query_frames.view(1, -1, 1)
        self._add("same_frame_mass", (probabilities * same_frame).sum(-1).flatten().cpu().tolist())
        distance = torch.maximum(
            (key_y.view(1, -1) - query_y.view(-1, 1)).abs(),
            (key_x.view(1, -1) - query_x.view(-1, 1)).abs(),
        )
        for radius in (0, 1, 2, 4, 8):
            spatial = distance <= radius
            local = spatial.unsqueeze(0) & same_frame
            self._add(
                f"same_frame_radius{radius}_mass",
                (probabilities * local).sum(-1).flatten().cpu().tolist(),
            )
            self._add(
                f"any_frame_radius{radius}_mass",
                (probabilities * spatial.unsqueeze(0)).sum(-1).flatten().cpu().tolist(),
            )
            self._add(
                f"uniform_same_frame_radius{radius}_fraction",
                local.float().mean(-1).flatten().cpu().tolist(),
            )
            self._add(
                f"top90_tokens_same_frame_radius{radius}_fraction",
                ((token_mask & local).sum(-1).float() /
                 token_counts.clamp_min(1)).flatten().cpu().tolist(),
            )

        # Measure K-to-K clustering, independent of the query's own position.
        flat_probabilities = probabilities.flatten(0, 1)
        rows = flat_probabilities.shape[0]
        top10_count = math.ceil(tokens * .10)
        top10_indices = flat_probabilities.topk(top10_count, dim=-1).indices
        top10_mask = torch.zeros_like(flat_probabilities, dtype=torch.bool)
        top10_mask.scatter_(1, top10_indices, True)
        top10_grid = top10_mask.reshape(rows * frames, 1, height, width).float()
        neighbour_kernel = torch.ones((1, 1, 3, 3), device=q.device)
        neighbour_kernel[..., 1, 1] = 0
        neighbour_counts = F.conv2d(top10_grid, neighbour_kernel, padding=1)
        valid_neighbours = F.conv2d(
            torch.ones_like(top10_grid), neighbour_kernel, padding=1)
        selected_neighbours = (neighbour_counts * top10_grid).reshape(rows, -1).sum(-1)
        selected_edges = (valid_neighbours * top10_grid).reshape(rows, -1).sum(-1)
        conditional = selected_neighbours / selected_edges.clamp_min(1)
        random_baseline = (top10_count - 1) / (tokens - 1)
        self._add("top10_token_neighbour_hit", conditional.cpu().tolist())
        self._add(
            "top10_token_neighbour_enrichment",
            (conditional / random_baseline).cpu().tolist(),
        )

        top1_indices = flat_probabilities.argmax(-1)
        top1_mass = flat_probabilities.gather(1, top1_indices[:, None]).squeeze(1)
        top1_frames = top1_indices // (height * width)
        top1_remainder = top1_indices % (height * width)
        top1_y, top1_x = top1_remainder // width, top1_remainder % width
        top1_distance = torch.maximum(
            (key_y.view(1, -1) - top1_y.view(-1, 1)).abs(),
            (key_x.view(1, -1) - top1_x.view(-1, 1)).abs(),
        )
        top1_same_frame = key_frames.view(1, -1) == top1_frames.view(-1, 1)
        for radius in (1, 2, 4):
            neighbour_mask = (
                top1_same_frame & (top1_distance <= radius) &
                (key_indices.view(1, -1) != top1_indices.view(-1, 1))
            )
            neighbour_mass = (flat_probabilities * neighbour_mask).sum(-1)
            remaining_uniform_mass = (
                (1 - top1_mass) * neighbour_mask.sum(-1) / (tokens - 1)
            ).clamp_min(1e-12)
            self._add(
                f"top1_token_radius{radius}_neighbour_mass",
                neighbour_mass.cpu().tolist(),
            )
            self._add(
                f"top1_token_radius{radius}_neighbour_enrichment",
                (neighbour_mass / remaining_uniform_mass).cpu().tolist(),
            )

        single_spatial_keep = _route_metrics(
            spatial4, spatial4_sizes, self.mass_target, 16)["tile_keep"]
        keep_tensor = torch.tensor(single_spatial_keep).reshape(heads, queries)
        for head in range(heads):
            self.layer_head_spatial_keep[str(layer)][str(head)].extend(
                keep_tensor[head].tolist())
        self.calls += 1

    def summary(self):
        steps = sorted(self.step_values)
        if len(steps) < self.min_steps:
            raise RuntimeError(f"Observed {len(steps)} steps; need at least {self.min_steps}")
        metrics = {name: _summary(values) for name, values in self.values.items()}
        per_step = [{
            "step": step,
            "metrics": {name: _summary(values) for name, values in self.step_values[step].items()},
        } for step in steps]
        layer_head = {
            layer: {
                head: sum(values) / len(values)
                for head, values in heads.items()
            } for layer, heads in self.layer_head_spatial_keep.items()
        }
        return {
            "schema_version": 1,
            "config": {
                "grid": self.grid,
                "sampled_queries_per_call": 32,
                "query_sampling": "two deterministic aligned 4x4 spatial blocks",
                "mass_target": self.mass_target,
                "minimum_required_steps": self.min_steps,
                "spatial_key_layouts": ["2x2", "4x4", "2x4"],
                "comparison_layout": "contiguous linear groups of 16 keys",
            },
            "calls": {
                "self_attention": self.calls,
                "cross_attention_skipped": self.cross_calls,
            },
            "denoising_steps_observed": len(steps),
            "metrics": metrics,
            "per_step": per_step,
            "per_layer_head_spatial4x4_q1_keep": layer_head,
            "representative_spatial2x2_snapshot": self.snapshot,
            "storage_note": "Only aggregate metrics are saved; Q, K, and sampled attention rows are discarded after each call.",
        }

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.summary(), indent=2) + "\n")


def install(probe):
    from profiles._runner import install_probe

    def attention_hook(original_attention, model_module, args, kwargs):
        q = kwargs.get("q", args[0] if args else None)
        k = kwargs.get("k", args[1] if len(args) > 1 else None)
        if q is not None and k is not None:
            probe.record(
                q,
                k,
                getattr(model_module, "_CURRENT_ATTN_ID", -1),
                getattr(model_module, "_CURRENT_GRID_SIZE", None),
            )
        return original_attention(*args, **kwargs)

    return install_probe(attention_hook, probe.begin_forward)

def main():
    from profiles._runner import add_model_args, run_probe

    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument(
        "--result", type=Path,
        default=Path("results/real_single_query_spatial_probe.json"))
    args = parser.parse_args()
    run_probe(SingleQuerySpatialProbe(args.result), args, install)


if __name__ == "__main__":
    main()
