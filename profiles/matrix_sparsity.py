"""Streaming probe for full two-dimensional Q x K attention-matrix tiles."""

from __future__ import annotations

import json
import argparse
import math
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


def _mean(values):
    return sum(values) / len(values) if values else None


def _quantiles(values):
    ordered = sorted(values)
    return {
        f"p{int(q * 100)}": ordered[min(len(ordered) - 1, int(q * len(ordered)))]
        for q in (.1, .5, .9)
    } if ordered else {}


def _local_offsets(length, tile, device):
    positions = torch.arange(length, device=device, dtype=torch.float32)
    starts = (positions // tile) * tile
    ends = torch.minimum(starts + tile, torch.tensor(length, device=device))
    centers = (starts + ends - 1) / 2
    scales = torch.clamp((ends - starts - 1) / 2, min=.5)
    return (positions - centers) / scales


@torch.no_grad()
def matrix_tile_statistics(q, k, tile=16, query_chunk=256):
    """Compute exact per-head tile mass and centroids without materializing LxL."""
    if q.shape[0] != 1 or q.shape[1] != k.shape[1]:
        raise ValueError("Matrix probe supports batch-1 self-attention")
    tokens, heads, head_dim = q.shape[1:]
    tiles = math.ceil(tokens / tile)
    mass = torch.zeros((heads, tiles, tiles), device=q.device)
    q_moment = torch.zeros_like(mass)
    k_moment = torch.zeros_like(mass)
    q_offsets = _local_offsets(tokens, tile, q.device)
    k_offsets = _local_offsets(tokens, tile, q.device)

    query_chunk = max(tile, query_chunk // tile * tile)
    for start in range(0, tokens, query_chunk):
        end = min(start + query_chunk, tokens)
        scores = torch.einsum(
            "bqhd,bkhd->bhqk", q[:, start:end].float(), k.float())
        probabilities = (scores / math.sqrt(head_dim)).softmax(-1)[0]
        query_tiles = math.ceil((end - start) / tile)
        padded = F.pad(
            probabilities,
            (0, tiles * tile - tokens, 0, query_tiles * tile - (end - start)))
        shape = (heads, query_tiles, tile, tiles, tile)
        mass[:, start // tile:start // tile + query_tiles] = padded.reshape(shape).sum((2, 4))

        q_weighted = probabilities * q_offsets[start:end].view(1, -1, 1)
        q_weighted = F.pad(
            q_weighted,
            (0, tiles * tile - tokens, 0, query_tiles * tile - (end - start)))
        q_moment[:, start // tile:start // tile + query_tiles] = q_weighted.reshape(shape).sum((2, 4))

        k_weighted = probabilities * k_offsets.view(1, 1, -1)
        k_weighted = F.pad(
            k_weighted,
            (0, tiles * tile - tokens, 0, query_tiles * tile - (end - start)))
        k_moment[:, start // tile:start // tile + query_tiles] = k_weighted.reshape(shape).sum((2, 4))

    # Every query row sums to one; normalize the full matrix mass to one.
    normalized_mass = mass / tokens
    denominator = mass.clamp_min(1e-12)
    return normalized_mass.cpu(), (q_moment / denominator).cpu(), (k_moment / denominator).cpu()


def _global_mask(mass, target):
    heads, query_tiles, key_tiles = mass.shape
    flat = mass.flatten(1)
    order = flat.argsort(-1, descending=True)
    cumulative = torch.cumsum(torch.gather(flat, -1, order), -1)
    counts = (cumulative < target).sum(-1) + 1
    ranks = torch.empty_like(order)
    rank_values = torch.arange(order.shape[-1]).expand_as(order)
    ranks.scatter_(-1, order, rank_values)
    mask = (ranks < counts.unsqueeze(-1)).reshape(heads, query_tiles, key_tiles)
    covered = (mass * mask).sum((1, 2))
    return mask, counts, covered


def _row_mask(mass, target):
    order = mass.argsort(-1, descending=True)
    sorted_mass = torch.gather(mass, -1, order)
    thresholds = mass.sum(-1, keepdim=True) * target
    counts = (torch.cumsum(sorted_mass, -1) < thresholds).sum(-1) + 1
    ranks = torch.empty_like(order)
    rank_values = torch.arange(order.shape[-1]).expand_as(order)
    ranks.scatter_(-1, order, rank_values)
    mask = ranks < counts.unsqueeze(-1)
    covered = (mass * mask).sum((1, 2))
    return mask, counts, covered


def _shift(mask, dy, dx):
    output = torch.zeros_like(mask)
    source_y = slice(max(0, -dy), mask.shape[-2] - max(0, dy))
    source_x = slice(max(0, -dx), mask.shape[-1] - max(0, dx))
    target_y = slice(max(0, dy), mask.shape[-2] - max(0, -dy))
    target_x = slice(max(0, dx), mask.shape[-1] - max(0, -dx))
    output[:, target_y, target_x] = mask[:, source_y, source_x]
    return output


def _expand(mask, mass, centroid_q, centroid_k, mode, centroid_threshold,
            mass_factor, huge_factor):
    if mode is None:
        return mask.clone()
    if mode == "all":
        return F.max_pool2d(mask.float(), 3, stride=1, padding=1).bool()

    selected_mass = torch.where(mask, mass, 0)
    mean_mass = selected_mass.sum((1, 2), keepdim=True) / mask.sum((1, 2), keepdim=True).clamp_min(1)
    ratio = mass / mean_mass.clamp_min(1e-12)
    all_mask = mask & (ratio >= huge_factor)
    predicted = mask | F.max_pool2d(all_mask.float(), 3, stride=1, padding=1).bool()
    directional = mask & ~all_mask & (ratio >= mass_factor)
    dy = torch.where(centroid_q.abs() < centroid_threshold, 0, torch.sign(centroid_q)).long()
    dx = torch.where(centroid_k.abs() < centroid_threshold, 0, torch.sign(centroid_k)).long()
    for y_direction in (-1, 0, 1):
        for x_direction in (-1, 0, 1):
            if y_direction == 0 and x_direction == 0:
                continue
            source = directional & (dy == y_direction) & (dx == x_direction)
            predicted |= _shift(source, y_direction, x_direction)
    return predicted


def _packing(mask):
    points = torch.where(mask)
    selected = int(mask.sum())
    if not selected:
        return {"bbox_utilization": 0.0, "row_runs": 0, "component_blocks": 0,
                "component_bbox_utilization": 0.0}
    rows, cols = mask.shape
    bbox_area = ((int(points[0].max()) - int(points[0].min()) + 1) *
                 (int(points[1].max()) - int(points[1].min()) + 1))
    row_runs = 0
    for row in mask:
        padded = F.pad(row, (1, 1))
        row_runs += int((~padded[:-1] & padded[1:]).sum())

    remaining = {(int(y), int(x)) for y, x in zip(*points)}
    components = []
    while remaining:
        seed = remaining.pop()
        component = {seed}
        stack = [seed]
        while stack:
            y, x = stack.pop()
            for neighbour in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    component.add(neighbour)
                    stack.append(neighbour)
        components.append(component)
    component_area = sum(
        (max(y for y, _ in component) - min(y for y, _ in component) + 1) *
        (max(x for _, x in component) - min(x for _, x in component) + 1)
        for component in components)
    return {
        "bbox_utilization": selected / bbox_area,
        "row_runs": row_runs,
        "component_blocks": len(components),
        "component_bbox_utilization": selected / component_area,
    }


class MatrixAttentionProbe:
    def __init__(self, path: Path, tile=16, query_chunk=256, mass_target=.90):
        self.path = Path(path)
        self.tile = tile
        self.query_chunk = query_chunk
        self.mass_target = mass_target
        self.min_steps = int(os.getenv("WAN_PROBE_MIN_STEPS", "5"))
        self.step = -1
        self.timestep = None
        self.branch = -1
        self.calls = 0
        self.cross_calls = 0
        self.previous = {}
        self.step_values = defaultdict(lambda: defaultdict(list))
        self.layer_head = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.policy_values = defaultdict(lambda: defaultdict(list))
        self.packing_values = defaultdict(lambda: defaultdict(list))
        self.policies = {
            "reuse-only": None,
            "all-eight-neighbours": "all",
            "directional-c0.00-m0.0-h1.0": (0.0, 0.0, 1.0),
            "directional-c0.00-m0.0-h1.5": (0.0, 0.0, 1.5),
            "directional-c0.15-m0.5-h2.0": (0.15, 0.5, 2.0),
        }

    def begin_forward(self, timestep):
        value = float(timestep.detach().flatten()[0].item())
        if self.timestep is None or not math.isclose(value, self.timestep):
            self.step += 1
            self.timestep = value
            self.branch = 0
        else:
            self.branch += 1

    @torch.no_grad()
    def record(self, q, k, attention_id):
        if q.shape[1] != k.shape[1]:
            self.cross_calls += 1
            return
        layer = int(attention_id) // 2
        mass, centroid_q, centroid_k = matrix_tile_statistics(
            q, k, self.tile, self.query_chunk)
        global_mask, global_counts, global_mass = _global_mask(mass, self.mass_target)
        row_mask, row_counts, row_mass = _row_mask(mass, self.mass_target)
        total_tiles = mass.shape[-1] * mass.shape[-2]
        global_keep = global_counts.float() / total_tiles
        row_keep = row_counts.float().sum(-1) / total_tiles

        self.calls += 1
        for head in range(mass.shape[0]):
            self.step_values[self.step]["global_keep"].append(float(global_keep[head]))
            self.step_values[self.step]["row_keep"].append(float(row_keep[head]))
            self.step_values[self.step]["global_mass"].append(float(global_mass[head]))
            self.step_values[self.step]["row_mass"].append(float(row_mass[head]))
            self.layer_head[str(layer)][str(head)]["global_keep"].append(float(global_keep[head]))
            self.layer_head[str(layer)][str(head)]["row_keep"].append(float(row_keep[head]))

        key = (self.branch, layer)
        if key in self.previous:
            previous = self.previous[key]
            for mask_kind, target in (("global", global_mask), ("row", row_mask)):
                previous_mask = previous[f"{mask_kind}_mask"]
                for name, policy in self.policies.items():
                    if policy is None:
                        predicted = previous_mask
                    elif policy == "all":
                        predicted = _expand(previous_mask, previous["mass"],
                                            previous["centroid_q"], previous["centroid_k"],
                                            "all", 0, 0, 0)
                    else:
                        predicted = _expand(previous_mask, previous["mass"],
                                            previous["centroid_q"], previous["centroid_k"],
                                            "directional", *policy)
                    intersection = (predicted & target).sum((1, 2)).float()
                    recall = intersection / target.sum((1, 2)).clamp_min(1)
                    covered_mass = (mass * predicted).sum((1, 2))
                    fraction = predicted.float().mean((1, 2))
                    key_policy = (mask_kind, name, self.step)
                    self.policy_values[key_policy]["recall"].extend(recall.tolist())
                    self.policy_values[key_policy]["mass"].extend(covered_mass.tolist())
                    self.policy_values[key_policy]["fraction"].extend(fraction.tolist())

                    for head in range(mass.shape[0]):
                        sample = (layer + self.branch + head + self.step) % 32 == 0
                        if (mask_kind == "global" and
                                name in ("reuse-only", "all-eight-neighbours",
                                         "directional-c0.00-m0.0-h1.0") and sample):
                            for metric, value in _packing(predicted[head]).items():
                                self.packing_values[name][metric].append(value)

        self.previous[key] = {
            "global_mask": global_mask,
            "row_mask": row_mask,
            "mass": mass.to(torch.float16),
            "centroid_q": centroid_q.to(torch.float16),
            "centroid_k": centroid_k.to(torch.float16),
        }

    def summary(self):
        steps = sorted(self.step_values)
        if len(steps) < self.min_steps:
            raise RuntimeError(f"Observed {len(steps)} steps; need at least {self.min_steps}")
        per_step = []
        for step in steps:
            values = self.step_values[step]
            per_step.append({
                "step": step,
                "mean_global_keep_fraction": _mean(values["global_keep"]),
                "global_keep_quantiles": _quantiles(values["global_keep"]),
                "mean_row_routed_keep_fraction": _mean(values["row_keep"]),
                "row_routed_keep_quantiles": _quantiles(values["row_keep"]),
                "mean_global_mass_covered": _mean(values["global_mass"]),
                "mean_row_routed_mass_covered": _mean(values["row_mass"]),
            })

        def summarize_policies(mask_kind):
            policies = []
            for name in self.policies:
                transitions = []
                for step in steps[1:]:
                    values = self.policy_values[(mask_kind, name, step)]
                    transitions.append({
                        "from_step": step - 1,
                        "to_step": step,
                        "mean_recall": _mean(values["recall"]),
                        "mean_mass_covered": _mean(values["mass"]),
                        "mean_predicted_tile_fraction": _mean(values["fraction"]),
                    })
                entry = {
                    "policy": name,
                    "mean_recall": _mean([x["mean_recall"] for x in transitions]),
                    "mean_mass_covered": _mean([x["mean_mass_covered"] for x in transitions]),
                    "mean_predicted_tile_fraction": _mean([
                        x["mean_predicted_tile_fraction"] for x in transitions]),
                    "transitions": transitions,
                }
                if mask_kind == "global" and name in self.packing_values:
                    entry["packing"] = {
                        metric: _mean(values)
                        for metric, values in self.packing_values[name].items()
                    }
                    entry["packing"]["observations"] = len(
                        self.packing_values[name]["component_blocks"])
                policies.append(entry)
            return policies

        policies = summarize_policies("global")
        row_policies = summarize_policies("row")

        feasible = [x for x in policies if x["mean_mass_covered"] >= self.mass_target]
        recommended = min(feasible, key=lambda x: x["mean_predicted_tile_fraction"])["policy"] if feasible else None
        row_feasible = [x for x in row_policies if x["mean_mass_covered"] >= self.mass_target]
        row_recommended = min(row_feasible, key=lambda x: x["mean_predicted_tile_fraction"])["policy"] if row_feasible else None
        layer_head_summary = {
            layer: {
                head: {metric: _mean(values) for metric, values in metrics.items()}
                for head, metrics in heads.items()
            } for layer, heads in self.layer_head.items()
        }
        return {
            "schema_version": 1,
            "config": {
                "matrix_tile_shape": [self.tile, self.tile],
                "query_chunk": self.query_chunk,
                "mass_target": self.mass_target,
                "minimum_required_steps": self.min_steps,
                "normalization": "softmax independently for every query row before tile summation",
                "routing_granularity": "independent per layer, head, CFG branch, query-matrix-tile, and denoising step",
                "tile_axes": ["query_token", "key_token"],
            },
            "calls": {"self_attention": self.calls, "cross_attention_skipped": self.cross_calls},
            "denoising_steps_observed": len(steps),
            "per_step": per_step,
            "per_layer_head": layer_head_summary,
            "locality_policy_sweep": policies,
            "recommended_policy_at_mass_target": recommended,
            "row_route_policy_sweep": row_policies,
            "recommended_row_route_policy_at_mass_target": row_recommended,
            "storage_note": "Only aggregates are saved; full QK or attention matrices are never written.",
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
            probe.record(q, k, getattr(model_module, "_CURRENT_ATTN_ID", -1))
        return original_attention(*args, **kwargs)

    return install_probe(attention_hook, probe.begin_forward)

def main():
    from profiles._runner import add_model_args, run_probe

    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument("--tile", type=int, default=16)
    parser.add_argument("--query-chunk", type=int, default=256)
    parser.add_argument(
        "--result", type=Path,
        default=Path("results/real_matrix_locality_probe.json"))
    args = parser.parse_args()
    run_probe(MatrixAttentionProbe(
        args.result, tile=args.tile, query_chunk=args.query_chunk), args, install)


if __name__ == "__main__":
    main()
