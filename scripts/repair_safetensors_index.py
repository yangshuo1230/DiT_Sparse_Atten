"""Rebuild a Diffusers safetensors index from the shards actually on disk."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--pattern", default="diffusion_pytorch_model-*-of-*.safetensors")
    parser.add_argument("--index", default="diffusion_pytorch_model.safetensors.index.json")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    shards = sorted(args.model_dir.glob(args.pattern))
    if not shards:
        raise SystemExit(f"No shards matching {args.pattern!r} under {args.model_dir}")
    weight_map = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118 - safe_open is not iterable.
                if key in weight_map:
                    raise SystemExit(
                        f"Duplicate tensor {key!r} in {weight_map[key]} and {shard.name}")
                weight_map[key] = shard.name
    payload = {
        "metadata": {"total_size": sum(shard.stat().st_size for shard in shards)},
        "weight_map": dict(sorted(weight_map.items())),
    }
    target = args.model_dir / args.index
    print(json.dumps({
        "target": str(target),
        "shards": [shard.name for shard in shards],
        "tensors": len(weight_map),
        "write": args.write,
    }, indent=2))
    if not args.write:
        return
    backup = target.with_suffix(target.suffix + ".bak")
    if target.exists():
        shutil.copy2(target, backup)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(target)
    print(f"Wrote {target}; previous index preserved at {backup}")


if __name__ == "__main__":
    main()
