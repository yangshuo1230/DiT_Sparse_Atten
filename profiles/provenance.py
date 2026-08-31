"""Reproducibility metadata shared by real-model profiling programs."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


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


def _value(value):
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _value(item) for key, item in value.items()}
    return str(value)


def collect(args, wan_repo):
    study_root = Path(__file__).resolve().parents[1]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command_config": {
            key: _value(value) for key, value in vars(args).items()
        },
        "environment": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": (torch.cuda.get_device_name(0)
                       if torch.cuda.is_available() else None),
            "study_git": _git_state(study_root),
            "wan_git": _git_state(wan_repo),
        },
    }
