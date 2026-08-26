"""Sweep sparse drop_factor and summarize mass/compute trade-offs.

Each value launches an independent ``infer.py`` process.  The sparse backend
emits JSONL route statistics when ``WAN_SPARSE_STATS_PATH`` is set.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--drop-factors", default="0,0.025,0.05,0.1,0.15,0.2,0.3,0.5")
    p.add_argument("--output-dir", type=Path, default=Path("strategy_search/results/drop_factor"))
    p.add_argument("--summary", type=Path, default=Path("strategy_search/results/drop_factor_summary.json"))
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--infer", type=Path, default=Path("infer.py"))
    p.add_argument("--keep-outputs", action="store_true")
    p.add_argument("infer_args", nargs=argparse.REMAINDER,
                   help="extra args passed to infer.py; do not include --backend/--drop-factor")
    return p.parse_args()


def main():
    args = parse_args()
    factors = [float(x) for x in args.drop_factors.split(",") if x.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for factor in factors:
        tag = f"drop_{factor:g}"
        stats_path = args.output_dir / f"{tag}.jsonl"
        video_path = args.output_dir / f"{tag}.mp4"
        stats_path.unlink(missing_ok=True)
        cmd = [args.python, str(args.infer), "--backend", "sparse",
               "--drop-factor", str(factor), "--output", str(video_path)]
        # argparse.REMAINDER keeps the conventional separator; it is for this
        # wrapper only and must not be forwarded to infer.py.
        extra_args = list(args.infer_args)
        if extra_args[:1] == ["--"]:
            extra_args = extra_args[1:]
        cmd.extend(extra_args)
        env = os.environ.copy()
        env["WAN_SPARSE_STATS_PATH"] = str(stats_path)
        print("$", " ".join(cmd), flush=True)
        completed = subprocess.run(cmd, env=env)
        if completed.returncode:
            raise SystemExit(f"drop_factor={factor} failed with exit code {completed.returncode}")
        rows = [json.loads(line) for line in stats_path.read_text().splitlines() if line.strip()]
        sparse = [row for row in rows if row.get("phase") == "sparse"]
        by_step = {}
        for row in sparse:
            step = str(row["step"])
            by_step.setdefault(step, []).append(row)
        step_summary = []
        for step, values in sorted(by_step.items(), key=lambda item: int(item[0])):
            mean = lambda key: sum(float(v[key]) for v in values) / len(values)
            step_summary.append({"step": int(step), "calls": len(values),
                                 "executed_tile_fraction": mean("executed_tile_fraction"),
                                 "route_mass_fraction": mean("route_mass_fraction"),
                                 "next_tile_fraction": mean("next_tile_fraction"),
                                 "next_route_mass_fraction": mean("next_route_mass_fraction")})
        records.append({"drop_factor": factor, "steps": step_summary,
                        "stats_path": str(stats_path), "video_path": str(video_path)})
        if not args.keep_outputs:
            video_path.unlink(missing_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
