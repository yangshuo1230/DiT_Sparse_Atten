"""Long-step probe for the current spatial sparse inference backend."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

from attention_backends.sparse import (
    MatrixSparseBackend,
    SparseConfig,
    _expand,
    _routing_context,
    _spatial_layout,
)
from profiles._runner import add_model_args


class RecordingBackend:
    """Delegate to MatrixSparseBackend while retaining bounded route metrics."""

    name = "sparse-recording"

    def __init__(self, config, records):
        self.inner = MatrixSparseBackend(config)
        self.config = config
        self.records = records

    def __call__(self, q, k, v, softmax_scale=None, **kwargs):
        if q.shape[0] != 1 or q.shape[1] != k.shape[1]:
            return self.inner(q, k, v, softmax_scale=softmax_scale, **kwargs)

        attention_id, step, branch, grid = _routing_context()
        layout = _spatial_layout(grid, self.config.tile, q.shape[1])
        if layout is None:
            return self.inner(q, k, v, softmax_scale=softmax_scale, **kwargs)

        key = (attention_id, branch, q.shape[2], layout)
        before = self.inner.state.get(key)
        if step == 0 or before is None:
            executed_fraction = 1.0
            predicted_fraction = 1.0
        else:
            predicted = _expand(
                before, self.config.policy, self.config.centroid_threshold,
                self.config.huge_factor, layout)
            predicted_fraction = float(predicted.float().mean().item())
            executed_fraction = predicted_fraction

        output = self.inner(q, k, v, softmax_scale=softmax_scale, **kwargs)
        after = self.inner.state.get(key)
        next_fraction = float(after["mask"].float().mean().item()) if after else 1.0
        dropped_fraction = 0.0
        retained_mass_fraction = 1.0
        if step > 0 and before is not None and after is not None:
            dropped_fraction = float(
                (predicted.cpu() & ~after["mask"]).float().mean().item())
            executed_mass = after["mass"]
            retained_mass_fraction = float(
                torch.where(after["mask"], executed_mass, 0).sum().item()
                / executed_mass.sum().clamp_min(1e-12).item())
        self.records.append({
            "step": int(step),
            "branch": int(branch),
            "attention_id": int(attention_id),
            "heads": int(q.shape[2]),
            "tokens": int(q.shape[1]),
            "grid": list(grid),
            "spatial_tile": [layout.tile_h, layout.tile_w],
            "tile_tokens": layout.tile_tokens,
            "executed_tile_fraction": executed_fraction,
            "predicted_tile_fraction": predicted_fraction,
            "next_mask_fraction": next_fraction,
            "dropped_previous_fraction": dropped_fraction,
            "retained_executed_mass_fraction": retained_mass_fraction,
        })
        return output


def _summary(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["step"]].append(record)
    rows = []
    for step in sorted(grouped):
        values = grouped[step]
        rows.append({
            "step": step,
            "calls": len(values),
            "executed_tile_fraction_mean": sum(
                x["executed_tile_fraction"] for x in values) / len(values),
            "next_mask_fraction_mean": sum(
                x["next_mask_fraction"] for x in values) / len(values),
            "dropped_previous_fraction_mean": sum(
                x["dropped_previous_fraction"] for x in values) / len(values),
            "retained_executed_mass_fraction_mean": sum(
                x["retained_executed_mass_fraction"] for x in values) / len(values),
            "executed_tile_fraction_min": min(
                x["executed_tile_fraction"] for x in values),
            "executed_tile_fraction_max": max(
                x["executed_tile_fraction"] for x in values),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument("--output", type=Path,
                        default=Path("results/spatial_sparse_long.json"))
    parser.add_argument("--tile", type=int, default=64)
    parser.add_argument("--keep", type=float, default=.625)
    parser.add_argument("--mass-target", type=float, default=.90)
    parser.add_argument("--drop-factor", type=float, default=.1)
    parser.add_argument("--query-chunk", type=int, default=256)
    parser.add_argument("--policy", choices=("reuse", "directional", "all"),
                        default="directional")
    args = parser.parse_args()

    wan_repo = args.wan_repo.resolve()
    sys.path.insert(0, str(wan_repo))
    import wan.modules.attention as attention_module
    import wan.modules.model as model_module
    records = []
    backend = RecordingBackend(SparseConfig(
        tile=args.tile, keep=args.keep, mass_target=args.mass_target,
        drop_factor=args.drop_factor, query_chunk=args.query_chunk,
        policy=args.policy), records)
    attention_module.flash_attention = backend
    model_module.flash_attention = backend

    args.video.parent.mkdir(parents=True, exist_ok=True)
    sys.argv = [
        str(wan_repo / "generate.py"), "--task", "t2v-14B",
        "--size", args.size, "--frame_num", str(args.frames),
        "--sample_steps", str(args.steps), "--base_seed", str(args.seed),
        "--ckpt_dir", str(args.model_dir.resolve()), "--prompt", args.prompt,
        "--save_file", str(args.video.resolve()), "--offload_model", "False",
    ]
    import runpy
    runpy.run_path(str(wan_repo / "generate.py"), run_name="__main__")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "config": {
            "size": args.size, "frames": args.frames, "steps": args.steps,
            "seed": args.seed, "tile_target": args.tile, "keep": args.keep,
            "mass_target": args.mass_target, "drop_factor": args.drop_factor,
            "query_chunk": args.query_chunk, "policy": args.policy,
        },
        "summary_by_step": _summary(records),
        "record_count": len(records),
        "records": records,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output), "record_count": len(records),
        "summary_by_step": payload["summary_by_step"],
    }, indent=2))


if __name__ == "__main__":
    main()
