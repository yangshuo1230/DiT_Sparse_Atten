"""Stage profile for exact and sampled FlexAttention route updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from attention_backends.flex import (
    _block_attention_mass,
    _build_block_mask,
    _flex_kernel,
    _flex_output,
    _sampled_block_mass,
    _top_mass_route,
)
from profiles.backend.timing import timed
from profiles.provenance import collect


def _route_quality(candidate, exact_route, exact_mass):
    intersection = candidate & exact_route
    union = candidate | exact_route
    exact_selected_mass = (exact_mass * exact_route).sum(-1).clamp_min(1e-12)
    return {
        "keep_fraction": float(candidate.float().mean().item()),
        "exact_route_recall": float(
            intersection.sum().float() / exact_route.sum().clamp_min(1)),
        "jaccard": float(intersection.sum().float() / union.sum().clamp_min(1)),
        "exact_selected_mass_recall": float(
            ((exact_mass * candidate).sum(-1) / exact_selected_mass).mean()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=57600)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--keep", type=float, default=.5)
    parser.add_argument("--mass-target", type=float, default=.9)
    parser.add_argument("--samples", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path,
                        default=Path("results/flex_route_profile.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tokens % args.block_size:
        raise SystemExit("This profile currently requires tokens divisible by block-size")
    generator = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (
        torch.randn(
            (1, args.tokens, args.heads, args.head_dim),
            device="cuda", dtype=torch.bfloat16, generator=generator)
        for _ in range(3)
    )
    scale = args.head_dim**-0.5
    # A raw FlexAttention call can fall back to math attention on accelerator
    # builds and materialize QK. Match inference by compiling the HOP once.
    flex = _flex_kernel(True)
    _, lse = flex(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        scale=scale, return_lse=True)
    exact_mass = _block_attention_mass(
        q, k, lse, args.block_size, scale=scale)
    exact_route = _top_mass_route(exact_mass, args.mass_target, args.keep)
    block_mask = _build_block_mask(exact_route, args.tokens, args.block_size)
    _flex_output(q, k, v, block_mask, args.block_size, True, scale=scale)

    stages = {
        "dense_sdpa_seconds": timed(
            lambda: torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                scale=scale), args.repeats, args.warmup),
        "dense_flex_lse_seconds": timed(
            lambda: flex(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                scale=scale, return_lse=True), args.repeats, args.warmup),
        "exact_mass_seconds": timed(
            lambda: _block_attention_mass(
                q, k, lse, args.block_size, scale=scale),
            args.repeats, args.warmup),
        "exact_route_selection_seconds": timed(
            lambda: _top_mass_route(exact_mass, args.mass_target, args.keep),
            args.repeats, args.warmup),
        "block_mask_build_seconds": timed(
            lambda: _build_block_mask(exact_route, args.tokens, args.block_size),
            args.repeats, args.warmup),
        "sparse_flex_output_seconds": timed(
            lambda: _flex_output(
                q, k, v, block_mask, args.block_size, True, scale=scale),
            args.repeats, args.warmup),
    }
    sampled = {}
    for samples in args.samples:
        mass = _sampled_block_mass(
            q, k, args.block_size, samples=samples, scale=scale)
        route = _top_mass_route(mass, args.mass_target, args.keep)
        mass_seconds = timed(
            lambda samples=samples: _sampled_block_mass(
                q, k, args.block_size, samples=samples, scale=scale),
            args.repeats, args.warmup)
        selection_seconds = timed(
            lambda mass=mass: _top_mass_route(
                mass, args.mass_target, args.keep),
            args.repeats, args.warmup)
        mask_seconds = timed(
            lambda route=route: _build_block_mask(
                route, args.tokens, args.block_size),
            args.repeats, args.warmup)
        sampled[str(samples)] = {
            "mass_seconds": mass_seconds,
            "route_selection_seconds": selection_seconds,
            "block_mask_build_seconds": mask_seconds,
            "update_seconds": mass_seconds + selection_seconds + mask_seconds,
            "update_plus_sparse_output_seconds": (
                mass_seconds + selection_seconds + mask_seconds
                + stages["sparse_flex_output_seconds"]),
            "quality": _route_quality(route, exact_route, exact_mass),
        }

    stages["exact_refresh_seconds"] = sum(stages[key] for key in (
        "dense_flex_lse_seconds", "exact_mass_seconds",
        "exact_route_selection_seconds", "block_mask_build_seconds"))
    report = {
        "schema_version": 1,
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "stages": stages,
        "sampled_updates": sampled,
        "run": collect(args, Path("/root/Wan2.1")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
