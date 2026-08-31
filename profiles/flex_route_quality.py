"""Compare lightweight sampled Flex routes with exact routes on real Wan Q/K."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from attention_backends.flex import (
    _block_attention_mass,
    _flex_kernel,
    _sampled_block_mass,
    _top_mass_route,
)


def _mean(values):
    return sum(values) / len(values) if values else None


def _quantiles(values):
    ordered = sorted(values)
    return {
        f"p{int(q * 100)}": ordered[min(len(ordered) - 1, int(q * len(ordered)))]
        for q in (.1, .5, .9)
    } if ordered else {}


class FlexRouteQualityProbe:
    def __init__(self, path, block_size=128, keep=.5, mass_target=.9,
                 samples=(1, 2, 4, 8, 16)):
        self.path = Path(path)
        self.block_size = block_size
        self.keep = keep
        self.mass_target = mass_target
        self.samples = tuple(samples)
        self.values = defaultdict(list)
        self.previous_exact = {}
        self.initial_exact = {}
        self.predicted = {}
        self.persistence_factors = (.5, 1., 2., 4.)
        self.calls = 0
        self.cross_calls = 0

    @torch.no_grad()
    def record(self, q, k, v, attention_id, step, branch,
               softmax_scale=None, q_scale=None, dtype=torch.bfloat16):
        if q.shape[0] != 1 or q.shape[1] != k.shape[1]:
            self.cross_calls += 1
            return None
        output_dtype = q.dtype
        q = q.to(dtype)
        k = k.to(dtype)
        v = v.to(dtype)
        if q_scale is not None:
            q = q * q_scale
        output, lse = _flex_kernel(True)(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            scale=softmax_scale, return_lse=True)
        exact_mass = _block_attention_mass(
            q, k, lse, self.block_size, scale=softmax_scale)
        exact_route = _top_mass_route(
            exact_mass, self.mass_target, self.keep)
        exact_total = exact_mass.sum().clamp_min(1e-12)
        exact_selected = (exact_mass * exact_route).sum().clamp_min(1e-12)

        key = (int(attention_id), int(branch))
        initial = self.initial_exact.setdefault(key, exact_route)
        if int(step) > 0:
            self.values["static_initial_route_recall"].append(float(
                (initial & exact_route).sum()
                / exact_route.sum().clamp_min(1)))
            self.values["static_initial_dense_mass_covered"].append(float(
                (exact_mass * initial).sum() / exact_total))
        previous = self.previous_exact.get(key)
        if previous is not None:
            intersection = previous & exact_route
            self.values["reuse_route_recall"].append(float(
                intersection.sum() / exact_route.sum().clamp_min(1)))
            self.values["reuse_dense_mass_covered"].append(float(
                (exact_mass * previous).sum() / exact_total))
        self.previous_exact[key] = exact_route

        for sample_count in self.samples:
            sampled_mass = _sampled_block_mass(
                q, k, self.block_size, samples=sample_count,
                scale=softmax_scale)
            sampled_route = _top_mass_route(
                sampled_mass, self.mass_target, self.keep)
            intersection = sampled_route & exact_route
            union = sampled_route | exact_route
            prefix = f"sample_{sample_count}"
            self.values[f"{prefix}_route_recall"].append(float(
                intersection.sum() / exact_route.sum().clamp_min(1)))
            self.values[f"{prefix}_jaccard"].append(float(
                intersection.sum() / union.sum().clamp_min(1)))
            self.values[f"{prefix}_dense_mass_covered"].append(float(
                (exact_mass * sampled_route).sum() / exact_total))
            self.values[f"{prefix}_selected_mass_recall"].append(float(
                (exact_mass * sampled_route).sum() / exact_selected))
            for factor in self.persistence_factors:
                prediction_key = (key, sample_count, factor)
                predicted_previous = self.predicted.get(prediction_key)
                if predicted_previous is None:
                    predicted_route = exact_route
                else:
                    row_scale = sampled_mass.mean(-1, keepdim=True)
                    blended_mass = (
                        sampled_mass
                        + factor * row_scale * predicted_previous.float())
                    predicted_route = _top_mass_route(
                        blended_mass, self.mass_target, self.keep)
                    blend_prefix = f"{prefix}_persistence_{factor:g}"
                    self.values[f"{blend_prefix}_route_recall"].append(float(
                        (predicted_route & exact_route).sum()
                        / exact_route.sum().clamp_min(1)))
                    self.values[f"{blend_prefix}_dense_mass_covered"].append(float(
                        (exact_mass * predicted_route).sum() / exact_total))
                    self.values[f"{blend_prefix}_keep_fraction"].append(float(
                        predicted_route.float().mean()))
                self.predicted[prediction_key] = predicted_route
        self.values["exact_keep_fraction"].append(float(
            exact_route.float().mean()))
        self.calls += 1
        return output.transpose(1, 2).contiguous().to(output_dtype)

    def save(self):
        report = {
            "schema_version": 1,
            "configuration": {
                "block_size": self.block_size,
                "keep": self.keep,
                "mass_target": self.mass_target,
                "samples": list(self.samples),
                "persistence_factors": list(self.persistence_factors),
            },
            "calls": {"self_attention": self.calls,
                      "cross_attention": self.cross_calls},
            "metrics": {
                name: {"mean": _mean(values), "quantiles": _quantiles(values)}
                for name, values in sorted(self.values.items())
            },
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
        output = probe.record(
            q, k, v,
            getattr(model_module, "_CURRENT_ATTN_ID", -1),
            getattr(model_module, "_CURRENT_DENOISE_STEP", -1),
            getattr(model_module, "_CURRENT_CFG_BRANCH", -1),
            softmax_scale=kwargs.get("softmax_scale"),
            q_scale=kwargs.get("q_scale"),
            dtype=kwargs.get("dtype", torch.bfloat16),
        )
        return output

    return install_probe(attention_hook, lambda timestep: None)


def main():
    from profiles._runner import add_model_args, run_probe

    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--keep", type=float, default=.5)
    parser.add_argument("--mass-target", type=float, default=.9)
    parser.add_argument("--samples", type=int, nargs="+", default=(1, 2, 4, 8, 16))
    parser.add_argument("--result", type=Path,
                        default=Path("results/flex_route_quality.json"))
    args = parser.parse_args()
    probe = FlexRouteQualityProbe(
        args.result, args.block_size, args.keep, args.mass_target, args.samples)
    run_probe(probe, args, install)


if __name__ == "__main__":
    main()
