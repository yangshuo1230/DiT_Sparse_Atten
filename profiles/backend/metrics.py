"""Numerical metrics for sparse-attention benchmark outputs."""

from __future__ import annotations

import torch


def relative_error(actual, reference):
    difference = (actual.float() - reference.float()).norm()
    denominator = reference.float().norm().clamp_min(1e-12)
    return float((difference / denominator).item())


def max_absolute_error(actual, reference):
    return float((actual.float() - reference.float()).abs().max().item())


def cosine_similarity(actual, reference):
    actual = actual.float().flatten()
    reference = reference.float().flatten()
    return float(torch.nn.functional.cosine_similarity(
        actual, reference, dim=0).item())


def summarize(actual, reference):
    return {
        "relative_l2": relative_error(actual, reference),
        "max_absolute_error": max_absolute_error(actual, reference),
        "cosine_similarity": cosine_similarity(actual, reference),
    }
