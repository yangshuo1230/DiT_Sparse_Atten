"""Shared real-Wan runner used only by profiling programs."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def add_model_args(parser):
    parser.add_argument("--wan-repo", type=Path, default=Path("/root/Wan2.1"))
    parser.add_argument("--model-dir", type=Path, default=Path("/root/.cache/wan2.1-14b"))
    parser.add_argument("--size", default="832*480")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompt", default="A small white dog running on a beach at sunset")
    parser.add_argument("--video", type=Path, default=Path("results/profile.mp4"))


def run_probe(probe, args, installer):
    wan_repo = args.wan_repo.resolve()
    sys.path.insert(0, str(wan_repo))
    from attention_backends.dense import install as install_dense

    install_dense()
    installer(probe)
    args.video.parent.mkdir(parents=True, exist_ok=True)
    sys.argv = [
        str(wan_repo / "generate.py"),
        "--task", "t2v-14B",
        "--size", args.size,
        "--frame_num", str(args.frames),
        "--sample_steps", str(args.steps),
        "--base_seed", str(args.seed),
        "--ckpt_dir", str(args.model_dir.resolve()),
        "--prompt", args.prompt,
        "--save_file", str(args.video.resolve()),
        "--offload_model", "False",
    ]
    runpy.run_path(str(wan_repo / "generate.py"), run_name="__main__")
    probe.save()
