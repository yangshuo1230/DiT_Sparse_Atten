"""Command-line frontend for the unified sparse-attention benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from profiles.backend.comparison import run_comparison


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=(3600, 14400, 57600))
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--tile", type=int, default=64)
    parser.add_argument("--keep", type=float, default=.625)
    parser.add_argument("--query-chunk", type=int, default=240)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path,
                        default=Path("results/sparse_attention_profile.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    report = {
        "schema_version": 1,
        "device": "cuda",
        "configuration": vars(args) | {"output": str(args.output)},
        "cases": [run_comparison(**{
            name: value for name, value in vars(args).items()
            if name not in ("tokens", "output")
        } | {"tokens": tokens}) for tokens in args.tokens],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
