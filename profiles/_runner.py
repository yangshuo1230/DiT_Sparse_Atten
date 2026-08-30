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


def install_probe(attention_hook, forward_hook):
    """Patch all Wan attention call sites and wrap model forward.

    Profilers previously duplicated this code independently. Keeping it in one
    place ensures both the attention module and WanModel module are patched,
    preserves the original attention callable for hooks, and makes future
    call-site changes a single edit.
    """
    import wan.modules.attention as attention_module
    import wan.modules.model as model_module

    original_attention = attention_module.flash_attention
    original_forward = model_module.WanModel.forward

    def wrapped_attention(*args, **kwargs):
        return attention_hook(original_attention, model_module, args, kwargs)

    def wrapped_forward(instance, *args, **kwargs):
        timestep = kwargs.get("t", args[1] if len(args) > 1 else None)
        if timestep is not None:
            forward_hook(timestep)
        return original_forward(instance, *args, **kwargs)

    attention_module.flash_attention = wrapped_attention
    model_module.flash_attention = wrapped_attention
    model_module.WanModel.forward = wrapped_forward
    return original_attention


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
