"""Minimal Wan2.1-T2V inference frontend with pluggable attention backends."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


DEFAULT_PROMPT = "A small white dog running on a beach at sunset"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("dense", "sparse"), default="dense")
    parser.add_argument("--wan-repo", type=Path, default=Path("/root/Wan2.1"))
    parser.add_argument("--model-dir", type=Path, default=Path("/root/.cache/wan2.1-14b"))
    parser.add_argument("--output", type=Path, default=Path("output.mp4"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--size", default="832*480")
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offload-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--timing", type=Path)
    parser.add_argument(
        "--tile", type=int, default=64,
        help=("target tokens per spatial HxW tile; the backend chooses exact "
              "divisors of Wan's current grid"))
    parser.add_argument("--keep", type=float, default=0.625)
    parser.add_argument("--mass-target", type=float, default=0.90)
    parser.add_argument(
        "--drop-factor", type=float, default=0.1,
        help="directional policy mass ratio below which a routed tile is dropped")
    parser.add_argument("--query-chunk", type=int, default=256)
    parser.add_argument("--policy", choices=("reuse", "directional", "all"), default="reuse")
    return parser.parse_args()


def install_backend(args):
    if args.backend == "dense":
        from attention_backends.dense import install
        return install()

    from attention_backends.sparse import SparseConfig, install_sparse
    return install_sparse(SparseConfig(
        tile=args.tile,
        keep=args.keep,
        mass_target=args.mass_target,
        drop_factor=args.drop_factor,
        query_chunk=args.query_chunk,
        policy=args.policy,
    ))


def main():
    args = parse_args()
    wan_repo = args.wan_repo.resolve()
    if not (wan_repo / "generate.py").is_file():
        raise SystemExit(f"Wan generate.py not found under {wan_repo}")
    if not args.model_dir.is_dir():
        raise SystemExit(f"Model directory not found: {args.model_dir}")
    sys.path.insert(0, str(wan_repo))
    install_backend(args)
    if args.timing:
        os.environ["WAN_TIMING_PATH"] = str(args.timing.resolve())
    else:
        os.environ.pop("WAN_TIMING_PATH", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.argv = [
        str(wan_repo / "generate.py"),
        "--task", "t2v-14B",
        "--size", args.size,
        "--frame_num", str(args.frames),
        "--sample_steps", str(args.steps),
        "--base_seed", str(args.seed),
        "--ckpt_dir", str(args.model_dir.resolve()),
        "--prompt", args.prompt,
        "--save_file", str(args.output.resolve()),
        "--offload_model", str(args.offload_model),
    ]
    runpy.run_path(str(wan_repo / "generate.py"), run_name="__main__")


if __name__ == "__main__":
    main()
