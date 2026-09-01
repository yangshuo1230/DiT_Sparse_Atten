"""Minimal Wan2.1-T2V inference frontend with pluggable attention backends."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

DEFAULT_PROMPT = "A small white dog running on a beach at sunset"
DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("dense", "sparse", "flex_reuse"), default="dense")
    parser.add_argument(
        "--model-dir", type=Path,
        default=Path("/root/models/wan2.1-t2v-14b-diffusers"))
    parser.add_argument("--output", type=Path, default=Path("output.mp4"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
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
    parser.add_argument(
        "--triton-sparse", action=argparse.BooleanOptionalAction, default=False,
        help="use the experimental fused Triton sparse-attention output kernel")
    parser.add_argument(
        "--route-prefetch", action=argparse.BooleanOptionalAction, default=True,
        help="prefetch the next sparse route on a separate CUDA stream")
    parser.add_argument(
        "--flex-block", type=int, default=128,
        help="fixed token block used by the flex_reuse backend")
    parser.add_argument(
        "--flex-compile", action=argparse.BooleanOptionalAction, default=True,
        help="compile the FlexAttention output kernel")
    parser.add_argument(
        "--flex-update-interval", type=int, default=0,
        help=("exactly recompute the Flex route every N denoising steps; "
              "0 keeps the step-0 route static"))
    parser.add_argument(
        "--flex-sampled-update-interval", type=int, default=0,
        help=("update the Flex route from sampled Q/K blocks every N steps; "
              "0 disables sampled updates"))
    parser.add_argument(
        "--flex-route-samples", type=int, default=2,
        help="Q and K samples per 128-token block for lightweight updates")
    parser.add_argument(
        "--flex-route-persistence", type=float, default=0.5,
        help="previous-route prior strength for sampled updates")
    parser.add_argument(
        "--flex-sampled-prefetch", action=argparse.BooleanOptionalAction,
        default=True,
        help="prepare the next sampled route on a separate CUDA stream")
    parser.add_argument(
        "--flex-bootstrap-prefetch", action=argparse.BooleanOptionalAction,
        default=True,
        help="prepare a sampled bootstrap route on a separate CUDA stream")
    parser.add_argument(
        "--flex-bootstrap", choices=("exact", "sampled"), default="exact",
        help="exact route bootstrap or dense-output plus sampled route bootstrap")
    parser.add_argument(
        "--flex-dense-route-threshold", type=float,
        help="use dense SDPA when a layer/branch route keep reaches this fraction")
    parser.add_argument(
        "--flex-spatial-reorder", action=argparse.BooleanOptionalAction,
        default=False,
        help="Morton-pack spatial microtiles into Flex route blocks")
    parser.add_argument(
        "--flex-spatial-microtile", type=int, default=32,
        help="maximum tokens in an exact spatial microtile")
    parser.add_argument(
        "--flex-directional-update", action=argparse.BooleanOptionalAction,
        default=False,
        help="expand only spatial route frontiers before fixed-budget pruning")
    parser.add_argument("--flex-direction-min-ratio", type=float, default=0.0)
    parser.add_argument("--flex-direction-bonus", type=float, default=0.0)
    parser.add_argument("--flex-route-budget-scale", type=float, default=1.0)
    parser.add_argument("--flex-route-exploration", type=float, default=0.0)
    parser.add_argument(
        "--flex-direction-q", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--flex-direction-k", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--flex-direction-joint", action=argparse.BooleanOptionalAction,
        default=True)
    return parser.parse_args()


def _git_state(path):
    try:
        commit = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True, capture_output=True, text=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return None


def _json_value(value):
    if isinstance(value, Path):
        return str(value.resolve())
    return value


def _write_timing(path, args, backend, error=None):
    import diffusers

    from attention_backends.context import runtime_profile

    profile = runtime_profile()
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    payload = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed" if error else "completed",
        "error": ({"type": type(error).__name__, "message": str(error)}
                  if error else None),
        "config": {
            key: _json_value(value)
            for key, value in vars(args).items()
            if key != "timing"
        },
        "environment": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": device,
            "diffusers_version": diffusers.__version__,
            "study_git": _git_state(Path(__file__).resolve().parent),
            "model_path": str(args.model_dir.resolve()),
            "svg_operators": getattr(backend, "svg_operator_status", None),
        },
        "routing": {
            "backend": getattr(backend, "name", type(backend).__name__),
            "flex_update_semantics": (
                f"sampled_refresh_every_{args.flex_sampled_update_interval}_steps"
                 if args.flex_sampled_update_interval else
                 "static_step_0_route" if args.flex_update_interval == 0 else
                 f"exact_refresh_every_{args.flex_update_interval}_steps"),
            "backend_profile": (
                backend.profile_summary()
                if hasattr(backend, "profile_summary") else None),
        },
        "timing": profile,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def install_backend(args):
    if args.backend == "dense":
        from attention_backends.dense import DenseBackend
        return DenseBackend()

    if args.backend == "flex_reuse":
        from attention_backends.flex import FlexReuseBackend, FlexReuseConfig
        return FlexReuseBackend(FlexReuseConfig(
            block_size=args.flex_block,
            keep=args.keep,
            mass_target=args.mass_target,
            compile_kernel=args.flex_compile,
            update_interval=args.flex_update_interval,
            sampled_update_interval=args.flex_sampled_update_interval,
            route_samples=args.flex_route_samples,
            route_persistence=args.flex_route_persistence,
            prefetch_sampled_update=args.flex_sampled_prefetch,
            prefetch_sampled_bootstrap=args.flex_bootstrap_prefetch,
            total_steps=args.steps,
            bootstrap_mode=args.flex_bootstrap,
            dense_route_threshold=args.flex_dense_route_threshold,
            spatial_reorder=args.flex_spatial_reorder,
            spatial_microtile_tokens=args.flex_spatial_microtile,
            directional_update=args.flex_directional_update,
            direction_min_ratio=args.flex_direction_min_ratio,
            direction_candidate_bonus=args.flex_direction_bonus,
            route_budget_scale=args.flex_route_budget_scale,
            route_exploration_fraction=args.flex_route_exploration,
            direction_expand_q=args.flex_direction_q,
            direction_expand_k=args.flex_direction_k,
            direction_expand_joint=args.flex_direction_joint,
        ))

    from attention_backends.sparse import MatrixSparseBackend, SparseConfig
    return MatrixSparseBackend(SparseConfig(
        tile=args.tile,
        keep=args.keep,
        mass_target=args.mass_target,
        drop_factor=args.drop_factor,
        query_chunk=args.query_chunk,
        policy=args.policy,
        use_triton=args.triton_sparse,
        prefetch_route=args.route_prefetch,
    ))


def main():
    args = parse_args()
    if not args.model_dir.is_dir():
        raise SystemExit(f"Model directory not found: {args.model_dir}")

    from attention_backends.context import _install_transformer_engine_compatibility
    _install_transformer_engine_compatibility()
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video

    from attention_backends.wan_diffusers import install_svg_wan_pipeline

    backend = install_backend(args)
    os.environ.pop("WAN_TIMING_PATH", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        width, height = (int(value) for value in args.size.split("*", 1))
    except (AttributeError, ValueError) as error:
        raise SystemExit("--size must use WIDTH*HEIGHT, for example 1280*720") from error

    model_path = str(args.model_dir.resolve())
    vae = AutoencoderKLWan.from_pretrained(
        model_path, subfolder="vae", torch_dtype=torch.float32,
        local_files_only=True)
    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction", use_flow_sigmas=True,
        num_train_timesteps=1000, flow_shift=5.0)
    pipe = WanPipeline.from_pretrained(
        model_path, vae=vae, torch_dtype=torch.bfloat16,
        local_files_only=True)
    pipe.scheduler = scheduler
    backend.svg_operator_status = install_svg_wan_pipeline(
        pipe.transformer, backend)
    if args.offload_model:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    error = None
    try:
        frames = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=height,
            width=width,
            num_frames=args.frames,
            guidance_scale=5.0,
            num_inference_steps=args.steps,
            generator=generator,
        ).frames[0]
        export_to_video(frames, str(args.output.resolve()), fps=16)
    except BaseException as caught:
        error = caught
        raise
    finally:
        if args.timing:
            _write_timing(args.timing.resolve(), args, backend, error)


if __name__ == "__main__":
    main()
