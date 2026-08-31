"""Compare two frame-aligned videos with RGB PSNR and SSIM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from profiles.hotspot_evolution import _video_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--size")
    parser.add_argument("--denoising-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--prompt")
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "comparison": {
            "reference": str(args.reference.resolve()),
            "candidate": str(args.candidate.resolve()),
            "method": ("frame-aligned decoded RGB in [0,1]; "
                       "skimage SSIM and RGB MSE PSNR"),
            "size": args.size,
            "denoising_steps": args.denoising_steps,
            "seed": args.seed,
            "prompt": args.prompt,
        },
        "metrics": _video_metrics(args.reference, args.candidate),
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
