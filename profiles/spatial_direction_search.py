"""Search frontier-directional route parameters on real spatially packed Q/K."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from attention_backends.flex import (
    _block_attention_mass,
    _flex_kernel,
    _sampled_block_mass,
    _top_mass_route,
)
from attention_backends.routing import directional_budget_update, frontier_mask
from attention_backends.spatial import build_spatial_layout


def _mean(values):
    return sum(values) / len(values) if values else None


def _quantiles(values):
    ordered = sorted(values)
    return {
        f"p{int(q * 100)}": ordered[min(len(ordered) - 1, int(q * len(ordered)))]
        for q in (.1, .5, .9)
    } if ordered else {}


def _route_metrics(route, exact_mass, exact_route, dense):
    if dense:
        return 1.0, 1.0, 1.0
    covered = float((exact_mass * route).sum() / exact_mass.sum().clamp_min(1e-12))
    recall = float((route & exact_route).sum() / exact_route.sum().clamp_min(1))
    return covered, recall, float(route.float().mean())


def _evaluate(records, neighbors, *, mode, keep, target, update_interval,
              dense_threshold, persistence=.5, min_ratio=0.0,
              candidate_bonus=0.0, budget_scale=1.0, exploration=0.0,
              joint=True):
    state = {}
    coverage = []
    recall = []
    work = []
    route_coverage = []
    route_recall = []
    route_keep = []
    frontier_values = []
    candidate_values = []
    added_values = []
    dropped_values = []
    dense_calls = 0
    sparse_calls = 0
    updates = 0

    last_step = max(record["step"] for record in records)
    for record in records:
        key = (record["attention_id"], record["branch"])
        exact_mass = record["exact_mass"].float()
        sampled_mass = record["sampled_mass"].float()
        exact_route = _top_mass_route(exact_mass, target, keep)
        current = state.get(key)
        if current is None:
            route = _top_mass_route(sampled_mass, target, keep)
            current = {
                "route": route,
                "frontier": frontier_mask(route, neighbors),
                "step": record["step"],
                "dense": float(route.float().mean()) >= dense_threshold,
            }
            state[key] = current
            continue  # Step 0 always executes complete dense attention.

        update_due = (
            mode != "static"
            and record["step"] - current["step"] >= update_interval
            and record["step"] < last_step
            and not current["dense"]
        )
        if update_due:
            if mode == "global":
                score = (
                    sampled_mass
                    + persistence * sampled_mass.mean(-1, keepdim=True)
                    * current["route"].float()
                )
                route = _top_mass_route(score, target, keep)
                result = {
                    "route": route,
                    "frontier": frontier_mask(route, neighbors),
                    "candidates": torch.zeros_like(route),
                    "added": route & ~current["route"],
                    "dropped": current["route"] & ~route,
                }
            else:
                result = directional_budget_update(
                    sampled_mass, current["route"], neighbors,
                    keep=keep, persistence=persistence,
                    min_ratio=min_ratio,
                    candidate_bonus=candidate_bonus,
                    budget_scale=budget_scale,
                    exploration_fraction=exploration,
                    expand_q=True, expand_k=True, expand_joint=joint,
                    previous_frontier=current["frontier"])
            current.update({
                "route": result["route"],
                "frontier": result["frontier"],
                "step": record["step"],
                "dense": (float(result["route"].float().mean())
                          >= dense_threshold),
            })
            frontier_values.append(float(result["frontier"].float().mean()))
            candidate_values.append(float(result["candidates"].float().mean()))
            added_values.append(float(result["added"].float().mean()))
            dropped_values.append(float(result["dropped"].float().mean()))
            updates += 1

        covered, got_recall, got_work = _route_metrics(
            current["route"], exact_mass, exact_route, current["dense"])
        raw_covered, raw_recall, raw_keep = _route_metrics(
            current["route"], exact_mass, exact_route, False)
        coverage.append(covered)
        recall.append(got_recall)
        work.append(got_work)
        route_coverage.append(raw_covered)
        route_recall.append(raw_recall)
        route_keep.append(raw_keep)
        dense_calls += int(current["dense"])
        sparse_calls += int(not current["dense"])

    return {
        "mode": mode,
        "parameters": {
            "persistence": persistence,
            "min_ratio": min_ratio,
            "candidate_bonus": candidate_bonus,
            "budget_scale": budget_scale,
            "exploration": exploration,
            "joint": joint,
        },
        "mean_dense_mass_covered": _mean(coverage),
        "mass_covered_quantiles": _quantiles(coverage),
        "mean_exact_route_recall": _mean(recall),
        "mean_work_fraction": _mean(work),
        "mean_route_dense_mass_covered": _mean(route_coverage),
        "mean_route_exact_recall": _mean(route_recall),
        "mean_route_keep": _mean(route_keep),
        "dense_calls": dense_calls,
        "sparse_calls": sparse_calls,
        "updates": updates,
        "mean_frontier_fraction": _mean(frontier_values),
        "mean_candidate_fraction": _mean(candidate_values),
        "mean_added_fraction": _mean(added_values),
        "mean_dropped_fraction": _mean(dropped_values),
    }


def _pareto(results):
    output = []
    for candidate in results:
        dominated = any(
            other["mean_dense_mass_covered"] >= candidate["mean_dense_mass_covered"]
            and other["mean_work_fraction"] <= candidate["mean_work_fraction"]
            and (other["mean_dense_mass_covered"] > candidate["mean_dense_mass_covered"]
                 or other["mean_work_fraction"] < candidate["mean_work_fraction"])
            for other in results
        )
        if not dominated:
            output.append(candidate)
    return sorted(output, key=lambda item: (
        item["mean_work_fraction"], -item["mean_dense_mass_covered"]))


class SpatialDirectionSearchProbe:
    def __init__(self, path, args):
        self.path = Path(path)
        self.args = args
        self.records = []
        self.layouts = {}
        self.cross_calls = 0

    @torch.no_grad()
    def record(self, q, k, v, attention_id, step, branch, grid,
               softmax_scale=None, q_scale=None, dtype=torch.bfloat16):
        if q.shape[0] != 1 or q.shape[1] != k.shape[1]:
            self.cross_calls += 1
            return None
        grid = tuple(int(value) for value in grid)
        layout = self.layouts.setdefault(
            grid, build_spatial_layout(
                grid, self.args.block_size, self.args.spatial_microtile))
        output_dtype = q.dtype
        q, k, v = (layout.reorder(value.to(dtype)) for value in (q, k, v))
        if q_scale is not None:
            q = q * q_scale
        output, lse = _flex_kernel(True)(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            scale=softmax_scale, return_lse=True)
        exact_mass = _block_attention_mass(
            q, k, lse, self.args.block_size, scale=softmax_scale)
        sampled_mass = _sampled_block_mass(
            q, k, self.args.block_size, samples=self.args.samples,
            scale=softmax_scale)
        self.records.append({
            "attention_id": int(attention_id),
            "step": int(step),
            "branch": int(branch),
            "exact_mass": exact_mass.to(torch.float16).cpu(),
            "sampled_mass": sampled_mass.to(torch.float16).cpu(),
        })
        return layout.restore(
            output.transpose(1, 2).contiguous()).to(output_dtype)

    def save(self):
        if not self.records:
            raise RuntimeError("spatial direction search recorded no self-attention")
        layout = next(iter(self.layouts.values()))
        neighbors = layout.neighbors_cpu.long()
        common = {
            "keep": self.args.keep,
            "target": self.args.mass_target,
            "update_interval": self.args.update_interval,
            "dense_threshold": self.args.dense_threshold,
        }
        results = [
            _evaluate(self.records, neighbors, mode="static", **common),
        ]
        for persistence in self.args.persistence:
            results.append(_evaluate(
                self.records, neighbors, mode="global",
                persistence=persistence, **common))
        for values in itertools.product(
                self.args.persistence, self.args.min_ratio,
                self.args.candidate_bonus, self.args.budget_scale,
                self.args.exploration, (False, True)):
            persistence, ratio, bonus, budget, exploration, joint = values
            results.append(_evaluate(
                self.records, neighbors, mode="directional",
                persistence=persistence, min_ratio=ratio,
                candidate_bonus=bonus, budget_scale=budget,
                exploration=exploration, joint=joint, **common))

        ranked = sorted(results, key=lambda item: (
            -(item["mean_dense_mass_covered"] - .12 * item["mean_work_fraction"]),
            item["mean_work_fraction"],
        ))
        report = {
            "schema_version": 1,
            "configuration": {
                "grid": list(layout.grid),
                "microtile_shape": list(layout.microtile_shape),
                "route_blocks": layout.blocks,
                "samples": self.args.samples,
                "keep": self.args.keep,
                "mass_target": self.args.mass_target,
                "update_interval": self.args.update_interval,
                "dense_threshold": self.args.dense_threshold,
                "parameter_combinations": len(results),
            },
            "calls": {
                "self_attention": len(self.records),
                "cross_attention": self.cross_calls,
            },
            "baselines": [result for result in results
                          if result["mode"] != "directional"],
            "pareto_frontier": _pareto(results),
            "ranked_top_20": ranked[:20],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(report, indent=2) + "\n")


def install(probe):
    from profiles._runner import install_probe

    def attention_hook(original_attention, model_module, args, kwargs):
        q = kwargs.get("q", args[0] if args else None)
        k = kwargs.get("k", args[1] if len(args) > 1 else None)
        v = kwargs.get("v", args[2] if len(args) > 2 else None)
        if q is None or k is None or v is None or q.shape[1] != k.shape[1]:
            probe.cross_calls += 1
            return original_attention(*args, **kwargs)
        return probe.record(
            q, k, v,
            getattr(model_module, "_CURRENT_ATTN_ID", -1),
            getattr(model_module, "_CURRENT_DENOISE_STEP", -1),
            getattr(model_module, "_CURRENT_CFG_BRANCH", -1),
            getattr(model_module, "_CURRENT_GRID_SIZE", None),
            softmax_scale=kwargs.get("softmax_scale"),
            q_scale=kwargs.get("q_scale"),
            dtype=kwargs.get("dtype", torch.bfloat16))

    return install_probe(attention_hook, lambda timestep: None)


def main():
    from profiles._runner import add_model_args, run_probe

    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--spatial-microtile", type=int, default=32)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--keep", type=float, default=.625)
    parser.add_argument("--mass-target", type=float, default=.95)
    parser.add_argument("--update-interval", type=int, default=2)
    parser.add_argument("--dense-threshold", type=float, default=.58)
    parser.add_argument("--persistence", type=float, nargs="+", default=(.25, .5, 1.))
    parser.add_argument("--min-ratio", type=float, nargs="+", default=(0., .5, 1.))
    parser.add_argument("--candidate-bonus", type=float, nargs="+", default=(0., .25, .5))
    parser.add_argument("--budget-scale", type=float, nargs="+", default=(.9, 1., 1.05))
    parser.add_argument("--exploration", type=float, nargs="+", default=(0., .02))
    parser.add_argument("--result", type=Path,
                        default=Path("results/spatial_direction_search.json"))
    args = parser.parse_args()
    run_probe(SpatialDirectionSearchProbe(args.result, args), args, install)


if __name__ == "__main__":
    main()
