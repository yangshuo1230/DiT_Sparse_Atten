"""Render the bounded representative 2x2 spatial attention snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path("results/real_single_query_spatial_probe.json"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/spatial2x2_attention_heatmap.png"))
    args = parser.parse_args()

    data = json.loads(args.input.read_text())
    snapshot = data["representative_spatial2x2_snapshot"]
    frames, rows, cols = snapshot["key_block_grid"]
    individual = np.asarray(snapshot["individual_query_block_mass"]).reshape(
        4, frames, rows, cols)
    aggregate = np.asarray(snapshot["aggregate_query_block_mass"]).reshape(
        frames, rows, cols)
    selected = np.asarray(snapshot["selected_key_blocks"]).reshape(
        frames, rows, cols)
    positive = individual[individual > 0]
    norm = LogNorm(
        vmin=max(float(np.percentile(positive, 1)), 1e-8),
        vmax=float(np.percentile(individual, 99.8)))

    figure, axes = plt.subplots(
        5, frames, figsize=(7 * frames, 17), constrained_layout=True,
        squeeze=False)
    query_coordinates = snapshot["query_coordinates_tyx"]
    image = None
    for query in range(4):
        qt, qy, qx = query_coordinates[query]
        for frame in range(frames):
            axis = axes[query, frame]
            image = axis.imshow(
                individual[query, frame], cmap="magma", norm=norm,
                origin="upper", interpolation="nearest", aspect="equal")
            if frame == qt:
                axis.add_patch(Rectangle(
                    (qx // 2 - .5, qy // 2 - .5), 1, 1,
                    fill=False, edgecolor="#31e6ff", linewidth=1.8))
            axis.set_title(
                f"q{query}=(t{qt}, y{qy}, x{qx}) -> K t{frame}", fontsize=11)
            axis.set_xlabel("K spatial block x (2 tokens)")
            axis.set_ylabel("K spatial block y (2 tokens)")

    aggregate_norm = LogNorm(
        vmin=max(float(np.percentile(aggregate[aggregate > 0], 1)), 1e-8),
        vmax=float(np.percentile(aggregate, 99.8)))
    for frame in range(frames):
        axis = axes[4, frame]
        axis.imshow(
            aggregate[frame], cmap="magma", norm=aggregate_norm,
            origin="upper", interpolation="nearest", aspect="equal")
        axis.contour(
            selected[frame].astype(float), levels=[.5], colors="#53ff77",
            linewidths=.65, origin="upper")
        axis.set_title(
            f"4-q aggregate -> K t{frame}; green = selected for 90% mass",
            fontsize=11)
        axis.set_xlabel("K spatial block x (2 tokens)")
        axis.set_ylabel("K spatial block y (2 tokens)")

    if image is not None:
        figure.colorbar(
            image, ax=axes[:4, :].ravel().tolist(), shrink=.72,
            label="Attention mass in each 2x2 K block (log scale)")
    figure.suptitle(
        "Wan2.1 single-frame spatial 2x2 attention blocks\n"
        f"step={snapshot['step']}, branch={snapshot['cfg_branch']}, "
        f"layer={snapshot['layer']}, head={snapshot['head']}, "
        f"keep={snapshot['route_keep_fraction']:.1%}, "
        f"covered={snapshot['route_mass_covered']:.2%}, "
        f"same-frame mass={snapshot['same_frame_mass']:.1%}",
        fontsize=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, facecolor="white")
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
