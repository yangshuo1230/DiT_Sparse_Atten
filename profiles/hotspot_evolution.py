"""Measure and suppress attention tiles that appear outside the previous route.

This is a deliberately small, oracle-assisted experiment.  Every denoising
step computes exact dense tile mass, constructs the per-query-tile route that
covers ``mass_target`` attention mass, and compares it with the previous
step's route after an optional matrix-neighbourhood dilation.

``observe`` leaves model output unchanged.  ``suppress`` returns attention
computed only on the predicted previous-step route.  Dense QK is still
computed for measurement, so this profile measures quality/counterfactual
behaviour, not speed.  With ``--reference-video``, the generated video is also
compared against a same-seed dense reference using PSNR and SSIM.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from attention_backends.sparse import (
    _row_mask,
    _spatial_layout,
    _spatial_neighbors,
)


def _mean(values):
    return sum(values) / len(values) if values else None


def _quantiles(values):
    ordered = sorted(values)
    return {
        f"p{int(q * 100)}": ordered[min(len(ordered) - 1, int(q * len(ordered)))]
        for q in (.1, .5, .9)
    } if ordered else {}


def _dilate_spatial_mask(mask, layout, radius):
    """Apply the backend's coupled Q/K spatial neighbourhood repeatedly."""
    expanded = mask.clone()
    for _ in range(max(0, radius)):
        expanded = _spatial_neighbors(expanded.reshape(
            mask.shape[0],
            layout.frames, layout.block_h, layout.block_w,
            layout.frames, layout.block_h, layout.block_w,
        )).reshape_as(mask)
    return expanded


def _token_mask(tile_mask, tile, query_start, query_end, tokens, device):
    """Expand a head-specific matrix-tile mask for one query chunk."""
    query_ids = torch.arange(query_start, query_end, device=device) // tile
    key_ids = torch.arange(tokens, device=device) // tile
    return tile_mask.to(device)[:, query_ids[:, None], key_ids[None, :]]


@torch.no_grad()
def _spatial_tile_mass(q, k, tile, query_chunk, softmax_scale=None,
                       q_scale=None):
    """Compute exact dense mass after Q/K have been reordered tile-major."""
    tokens, heads, head_dim = q.shape[1:]
    tiles = tokens // tile
    mass = torch.zeros((heads, tiles, tiles), device=q.device)
    scale = softmax_scale if softmax_scale is not None else 1 / math.sqrt(head_dim)
    q_value = q.float()
    if q_scale is not None:
        q_value = q_value * q_scale
    k_value = k.float()
    query_chunk = max(tile, query_chunk // tile * tile)
    for start in range(0, tokens, query_chunk):
        end = min(start + query_chunk, tokens)
        scores = torch.einsum(
            "bqhd,bkhd->bhqk", q_value[:, start:end], k_value) * scale
        probabilities = scores.softmax(-1)[0]
        query_tiles = (end - start) // tile
        mass[:, start // tile:start // tile + query_tiles] = probabilities.reshape(
            heads, query_tiles, tile, tiles, tile).sum((2, 4))
    return (mass / tokens).cpu()


@torch.no_grad()
def _masked_attention(q, k, v, tile_mask, tile, query_chunk,
                      softmax_scale=None, q_scale=None):
    """Reference attention restricted to a [head, q_tile, k_tile] mask."""
    if q.shape[0] != 1:
        raise ValueError("Hotspot suppression supports batch-1 self-attention")
    tokens, heads, head_dim = q.shape[1:]
    scale = softmax_scale if softmax_scale is not None else 1 / math.sqrt(head_dim)
    q_value = q.float()
    if q_scale is not None:
        q_value = q_value * q_scale
    k_value = k.float()
    v_value = v.float()
    output = torch.empty_like(q_value)
    query_chunk = max(tile, query_chunk // tile * tile)
    for start in range(0, tokens, query_chunk):
        end = min(start + query_chunk, tokens)
        scores = torch.einsum(
            "bqhd,bkhd->bhqk", q_value[:, start:end], k_value) * scale
        allowed = _token_mask(
            tile_mask, tile, start, end, tokens, scores.device)
        # The oracle row route always has at least one K tile.  Keep this
        # assertion explicit so an invalid experimental mask cannot produce
        # silent NaNs.
        if not allowed.any(-1).all():
            raise RuntimeError("suppression mask contains an empty query row")
        probabilities = scores.masked_fill(~allowed.unsqueeze(0), -torch.inf).softmax(-1)
        output[:, start:end] = torch.einsum(
            "bhqk,bkhd->bqhd", probabilities, v_value)
    return output.to(q.dtype)


def _video_metrics(reference_path, candidate_path):
    """Return frame-aligned RGB PSNR/SSIM without adding a hard dependency."""
    import cv2
    from skimage.metrics import structural_similarity

    reference = cv2.VideoCapture(str(reference_path))
    candidate = cv2.VideoCapture(str(candidate_path))
    psnr_values = []
    ssim_values = []
    while True:
        ok_reference, ref = reference.read()
        ok_candidate, got = candidate.read()
        if not ok_reference or not ok_candidate:
            if ok_reference != ok_candidate:
                raise RuntimeError("reference and candidate have different frame counts")
            break
        if ref.shape != got.shape:
            raise RuntimeError(
                f"video frame shapes differ: {ref.shape} versus {got.shape}")
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB).astype("float32") / 255
        got = cv2.cvtColor(got, cv2.COLOR_BGR2RGB).astype("float32") / 255
        mse = float(((ref - got) ** 2).mean())
        psnr_values.append(float("inf") if mse == 0 else -10 * math.log10(mse))
        ssim_values.append(float(structural_similarity(
            ref, got, data_range=1.0, channel_axis=2)))
    reference.release()
    candidate.release()
    if not psnr_values:
        raise RuntimeError("no video frames were decoded")
    return {
        "frames": len(psnr_values),
        "mean_psnr": _mean(psnr_values),
        "mean_ssim": _mean(ssim_values),
        "psnr_quantiles": _quantiles(psnr_values),
        "ssim_quantiles": _quantiles(ssim_values),
    }


class HotspotEvolutionProbe:
    """Dense oracle probe with an optional no-sudden-hotspot intervention."""

    def __init__(self, path, tile=64, query_chunk=256, mass_target=.90,
                 expansion_radius=1, mode="observe", reference_video=None,
                 candidate_video=None):
        self.path = Path(path)
        self.tile = tile
        self.query_chunk = query_chunk
        self.mass_target = mass_target
        self.expansion_radius = expansion_radius
        self.mode = mode
        self.reference_video = Path(reference_video) if reference_video else None
        self.candidate_video = Path(candidate_video) if candidate_video else None
        self.step = -1
        self.timestep = None
        self.branch = -1
        self.calls = 0
        self.cross_calls = 0
        self.previous = {}
        self.values = defaultdict(lambda: defaultdict(list))
        self.output_errors = defaultdict(list)
        self._last_summary = None

    def begin_forward(self, timestep):
        value = float(timestep.detach().flatten()[0].item())
        if self.timestep is None or not math.isclose(value, self.timestep):
            self.step += 1
            self.timestep = value
            self.branch = 0
        else:
            self.branch += 1

    @torch.no_grad()
    def route_for(self, q, k, v, attention_id, grid, softmax_scale=None,
                  q_scale=None):
        if q.shape[1] != k.shape[1]:
            self.cross_calls += 1
            return None
        layout = _spatial_layout(grid, self.tile, q.shape[1])
        if layout is None:
            self.cross_calls += 1
            return None
        layer = int(attention_id) // 2
        q_major = layout.to_tile_major(q)
        k_major = layout.to_tile_major(k)
        v_major = layout.to_tile_major(v)
        mass = _spatial_tile_mass(
            q_major, k_major, layout.tile_tokens, self.query_chunk,
            softmax_scale=softmax_scale, q_scale=q_scale)
        oracle = _row_mask(mass, self.mass_target, 1.0)
        oracle_covered = (mass * oracle).sum((1, 2))
        key = (self.branch, layer)
        predicted = None
        if key in self.previous:
            predicted = _dilate_spatial_mask(
                self.previous[key], layout, self.expansion_radius)
            sudden = oracle & ~predicted
            oracle_tiles = oracle.sum((1, 2)).clamp_min(1)
            sudden_tiles = sudden.sum((1, 2))
            sudden_mass = (mass * sudden).sum((1, 2))
            missed_total_mass = (mass * ~predicted).sum((1, 2))
            recall = (oracle & predicted).sum((1, 2)) / oracle_tiles
            predicted_fraction = predicted.float().mean((1, 2))
            for head in range(mass.shape[0]):
                current = self.values[self.step]
                current["sudden_oracle_tile_fraction"].append(
                    float(sudden_tiles[head] / oracle_tiles[head]))
                current["sudden_dense_mass_fraction"].append(float(sudden_mass[head]))
                current["all_mass_outside_prediction"].append(
                    float(missed_total_mass[head]))
                current["oracle_route_recall"].append(float(recall[head]))
                current["predicted_tile_fraction"].append(
                    float(predicted_fraction[head]))
                current["oracle_mass_covered"].append(float(oracle_covered[head]))
        self.previous[key] = oracle
        self.calls += 1
        self.actual_spatial_tile = [layout.tile_h, layout.tile_w]
        return predicted, layout, q_major, k_major, v_major

    def record_output_error(self, dense, suppressed):
        difference = (dense.float() - suppressed.float()).flatten()
        reference = dense.float().flatten()
        relative_l2 = float(
            difference.square().sum().sqrt() /
            reference.square().sum().sqrt().clamp_min(1e-12))
        cosine = float(F.cosine_similarity(
            dense.float().flatten(), suppressed.float().flatten(), dim=0))
        self.output_errors[self.step].append({
            "relative_l2": relative_l2,
            "cosine_similarity": cosine,
            "max_absolute_error": float(difference.abs().max()),
        })

    def summary(self):
        per_transition = []
        for step in sorted(self.values):
            metrics = self.values[step]
            entry = {"from_step": step - 1, "to_step": step}
            for name, values in metrics.items():
                entry[f"mean_{name}"] = _mean(values)
                entry[f"{name}_quantiles"] = _quantiles(values)
            errors = self.output_errors.get(step, [])
            if errors:
                for name in errors[0]:
                    values = [row[name] for row in errors]
                    entry[f"mean_attention_output_{name}"] = _mean(values)
                    entry[f"attention_output_{name}_quantiles"] = _quantiles(values)
            per_transition.append(entry)
        result = {
            "schema_version": 1,
            "hypotheses": {
                "h1": "Few oracle-route tiles appear outside the dilated previous oracle route.",
                "h2": "Suppressing all such tiles has little effect on attention outputs and final video.",
            },
            "config": {
                "mode": self.mode,
                "target_spatial_tile_tokens": self.tile,
                "actual_spatial_tile_shape": getattr(
                    self, "actual_spatial_tile", None),
                "spatial_neighbourhood_radius": self.expansion_radius,
                "mass_target": self.mass_target,
                "query_chunk": self.query_chunk,
                "sudden_hotspot_definition": (
                    "current per-query-tile oracle route member outside the backend's "
                    "coupled Q/K spatial dilation of the previous-step oracle route"),
                "intervention": (
                    "attention is restricted to the dilated previous oracle route"
                    if self.mode == "suppress" else "none"),
                "speed_note": "Dense QK is retained for oracle measurement; timings are not speed results.",
            },
            "calls": {
                "self_attention": self.calls,
                "cross_attention_skipped": self.cross_calls,
            },
            "per_transition": per_transition,
            "aggregate": {
                name: _mean([
                    value for metrics in self.values.values()
                    for value in metrics[name]])
                for name in (
                    "sudden_oracle_tile_fraction",
                    "sudden_dense_mass_fraction",
                    "all_mass_outside_prediction",
                    "oracle_route_recall",
                    "predicted_tile_fraction",
                )
            } if self.values else {},
            "interpretation_guardrails": [
                "This experiment isolates cross-step surprise; it does not benchmark speed.",
                "The predictor receives the previous exact oracle route and is therefore an upper bound on a deployed sparse predictor.",
                "One prompt and seed are insufficient for a general quality conclusion.",
                "Low average surprise is not sufficient if p90 or worst-case layer/head values are large.",
            ],
        }
        if (self.reference_video and self.candidate_video and
                self.reference_video.is_file() and self.candidate_video.is_file()):
            result["video_comparison"] = _video_metrics(
                self.reference_video, self.candidate_video)
        self._last_summary = result
        return result

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.summary(), indent=2) + "\n")


def install(probe):
    import wan.modules.attention as attention_module
    import wan.modules.model as model_module

    original_attention = attention_module.flash_attention
    original_forward = model_module.WanModel.forward

    def wrapped_attention(*args, **kwargs):
        q = kwargs.get("q", args[0] if args else None)
        k = kwargs.get("k", args[1] if len(args) > 1 else None)
        v = kwargs.get("v", args[2] if len(args) > 2 else None)
        dense_output = original_attention(*args, **kwargs)
        if q is None or k is None or v is None:
            return dense_output
        route = probe.route_for(
            q, k, v, getattr(model_module, "_CURRENT_ATTN_ID", -1),
            getattr(model_module, "_CURRENT_GRID_SIZE", None),
            softmax_scale=kwargs.get("softmax_scale"),
            q_scale=kwargs.get("q_scale"))
        if probe.mode != "suppress" or route is None or route[0] is None:
            return dense_output
        predicted, layout, q_major, k_major, v_major = route
        suppressed = _masked_attention(
            q_major, k_major, v_major, predicted, layout.tile_tokens,
            probe.query_chunk,
            softmax_scale=kwargs.get("softmax_scale"),
            q_scale=kwargs.get("q_scale"))
        suppressed = layout.from_tile_major(suppressed)
        probe.record_output_error(dense_output, suppressed)
        return suppressed

    def wrapped_forward(instance, *args, **kwargs):
        timestep = kwargs.get("t", args[1] if len(args) > 1 else None)
        if timestep is not None:
            probe.begin_forward(timestep)
        return original_forward(instance, *args, **kwargs)

    attention_module.flash_attention = wrapped_attention
    model_module.flash_attention = wrapped_attention
    model_module.WanModel.forward = wrapped_forward
    return original_attention


def main():
    from profiles._runner import add_model_args, run_probe

    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument("--tile", type=int, default=64)
    parser.add_argument("--query-chunk", type=int, default=256)
    parser.add_argument("--mass-target", type=float, default=.90)
    parser.add_argument("--expansion-radius", type=int, default=1)
    parser.add_argument("--mode", choices=("observe", "suppress"), default="observe")
    parser.add_argument("--reference-video", type=Path)
    parser.add_argument(
        "--result", type=Path,
        default=Path("results/hotspot_evolution.json"))
    args = parser.parse_args()
    probe = HotspotEvolutionProbe(
        args.result, tile=args.tile, query_chunk=args.query_chunk,
        mass_target=args.mass_target, expansion_radius=args.expansion_radius,
        mode=args.mode, reference_video=args.reference_video,
        candidate_video=args.video)
    run_probe(probe, args)
    # run_probe saves immediately after generation. Save once more so the
    # just-written candidate video can be decoded for final quality metrics.
    if args.reference_video:
        probe.save()


if __name__ == "__main__":
    main()
