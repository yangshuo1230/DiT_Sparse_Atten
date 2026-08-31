"""Runtime Wan context injection and lightweight model-forward profiling.

The study used to require local edits to Wan's ``model.py`` to expose the
current layer, denoising step, CFG branch, and token grid.  This module installs
the same information at runtime, before the model is constructed, so an
unmodified supported Wan2.1 checkout is sufficient.
"""

from __future__ import annotations

import functools
import importlib.util
import math
import os.path as osp
import sys
import time
import types
import warnings
from dataclasses import dataclass, field

import torch


@dataclass
class _RuntimeState:
    attention_id: int = -1
    denoising_step: int = -1
    cfg_branch: int = -1
    grid_size: tuple[int, int, int] | None = None
    last_timestep: float | None = None
    forward_calls: list[dict] = field(default_factory=list)


_STATE = _RuntimeState()


def reset_runtime_state():
    """Reset per-generation routing and timing state."""
    global _STATE
    _STATE = _RuntimeState()


def routing_context(include_grid=False):
    values = (
        _STATE.attention_id,
        _STATE.denoising_step,
        _STATE.cfg_branch,
    )
    return values + (_STATE.grid_size,) if include_grid else values


def _publish_compatibility_context(model_module):
    """Keep existing profilers compatible without requiring a Wan patch."""
    model_module._CURRENT_ATTN_ID = _STATE.attention_id
    model_module._CURRENT_DENOISE_STEP = _STATE.denoising_step
    model_module._CURRENT_CFG_BRANCH = _STATE.cfg_branch
    model_module._CURRENT_GRID_SIZE = _STATE.grid_size


def _timestep_value(args, kwargs):
    timestep = kwargs.get("t", args[1] if len(args) > 1 else None)
    if timestep is None:
        return None
    if torch.is_tensor(timestep):
        return float(timestep.detach().flatten()[0].item())
    return float(timestep)


def _grid_value(args, kwargs):
    grid_sizes = kwargs.get("grid_sizes", args[2] if len(args) > 2 else None)
    if grid_sizes is None:
        return None
    grid = grid_sizes[0]
    if torch.is_tensor(grid):
        grid = grid.detach().cpu().tolist()
    return tuple(int(value) for value in grid)


def install_wan_context():
    """Inject routing context into an imported, otherwise unmodified Wan model.

    Installation is idempotent per ``wan.modules.model`` module.  It must run
    before model construction so every attention module receives a stable ID.
    """
    import wan.modules.model as model_module

    if getattr(model_module, "_WAN_ATTENTION_STUDY_CONTEXT", False):
        return

    _install_video_compatibility()
    _install_transformer_engine_compatibility()

    reset_runtime_state()
    next_attention_id = 0
    original_attention_init = model_module.WanSelfAttention.__init__
    original_self_forward = model_module.WanSelfAttention.forward
    original_model_forward = model_module.WanModel.forward

    @functools.wraps(original_attention_init)
    def attention_init(instance, *args, **kwargs):
        nonlocal next_attention_id
        original_attention_init(instance, *args, **kwargs)
        instance._wan_attention_study_id = next_attention_id
        next_attention_id += 1

    @functools.wraps(original_self_forward)
    def self_forward(instance, *args, **kwargs):
        _STATE.attention_id = instance._wan_attention_study_id
        grid = _grid_value(args, kwargs)
        if grid is not None:
            _STATE.grid_size = grid
        _publish_compatibility_context(model_module)
        return original_self_forward(instance, *args, **kwargs)

    @functools.wraps(original_model_forward)
    def model_forward(instance, *args, **kwargs):
        timestep = _timestep_value(args, kwargs)
        if timestep is not None:
            if (_STATE.last_timestep is None or
                    not math.isclose(timestep, _STATE.last_timestep)):
                _STATE.denoising_step += 1
                _STATE.cfg_branch = 0
                _STATE.last_timestep = timestep
            else:
                _STATE.cfg_branch += 1
        _publish_compatibility_context(model_module)

        try:
            device = next(instance.parameters()).device
        except (AttributeError, StopIteration):
            device = torch.device("cpu")
        start_event = end_event = None
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        wall_started = time.perf_counter()
        try:
            return original_model_forward(instance, *args, **kwargs)
        finally:
            if end_event is not None:
                end_event.record()
            _STATE.forward_calls.append({
                "step": _STATE.denoising_step,
                "branch": _STATE.cfg_branch,
                "timestep": timestep,
                "wall_seconds_unsynchronized": time.perf_counter() - wall_started,
                "start_event": start_event,
                "end_event": end_event,
            })

    model_module.WanSelfAttention.__init__ = attention_init
    model_module.WanSelfAttention.forward = self_forward
    model_module.WanModel.forward = model_forward
    model_module._WAN_ATTENTION_STUDY_CONTEXT = True
    _publish_compatibility_context(model_module)


def _install_video_compatibility():
    """Install portable uint8 video conversion without editing Wan sources."""
    try:
        import wan.utils.utils as utils_module
    except ImportError:
        return
    if getattr(utils_module, "_WAN_ATTENTION_STUDY_VIDEO", False):
        return

    def cache_video(
        tensor,
        save_file=None,
        fps=30,
        suffix=".mp4",
        nrow=8,
        normalize=True,
        value_range=(-1, 1),
        retry=5,
    ):
        cache_file = (
            osp.join("/tmp", utils_module.rand_name(suffix=suffix))
            if save_file is None else save_file
        )
        error = None
        for _ in range(retry):
            writer = None
            try:
                value = tensor.clamp(min(value_range), max(value_range))
                value = torch.stack([
                    utils_module.torchvision.utils.make_grid(
                        frame, nrow=nrow, normalize=normalize,
                        value_range=value_range)
                    for frame in value.unbind(2)
                ], dim=1).permute(1, 2, 3, 0)
                value = (
                    value * 255
                ).clamp(0, 255).round().to(torch.uint8).cpu()
                writer = utils_module.imageio.get_writer(
                    cache_file, fps=fps, codec="libx264", quality=8)
                for frame in value.numpy():
                    writer.append_data(frame)
                writer.close()
                return cache_file
            except Exception as caught:  # noqa: BLE001 - match Wan's retry contract.
                error = caught
                if writer is not None:
                    writer.close()
        print(f"cache_video failed, error: {error}", flush=True)
        return None

    utils_module.cache_video = cache_video
    utils_module._WAN_ATTENTION_STUDY_VIDEO = True


def _install_transformer_engine_compatibility():
    """Hide a present but unloadable Transformer Engine from optional PEFT use."""
    specification = importlib.util.find_spec("transformer_engine")
    if specification is None:
        return
    try:
        __import__("transformer_engine")
        return
    except (ImportError, OSError) as error:
        for name in list(sys.modules):
            if name == "transformer_engine" or name.startswith("transformer_engine."):
                sys.modules.pop(name, None)
        stub = types.ModuleType("transformer_engine")
        stub.__spec__ = specification
        sys.modules["transformer_engine"] = stub
        warnings.warn(
            f"Transformer Engine is installed but cannot load ({error}); "
            "disabling optional PEFT Transformer Engine dispatch.",
            RuntimeWarning,
            stacklevel=2,
        )


def runtime_profile():
    """Synchronize recorded events and return serializable per-forward timing."""
    if any(call["end_event"] is not None for call in _STATE.forward_calls):
        torch.cuda.synchronize()
    calls = []
    for call in _STATE.forward_calls:
        elapsed = None
        if call["start_event"] is not None:
            elapsed = call["start_event"].elapsed_time(call["end_event"]) / 1000
        calls.append({
            "step": call["step"],
            "branch": call["branch"],
            "timestep": call["timestep"],
            "cuda_seconds": elapsed,
            "wall_seconds_unsynchronized": call["wall_seconds_unsynchronized"],
        })
    cuda_values = [call["cuda_seconds"] for call in calls
                   if call["cuda_seconds"] is not None]
    return {
        "denoising_steps_observed": len({call["step"] for call in calls}),
        "model_forward_calls": len(calls),
        "model_forward_cuda_seconds": sum(cuda_values) if cuda_values else None,
        "calls": calls,
    }
