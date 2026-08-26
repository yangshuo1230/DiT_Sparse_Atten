# Wan attention study

Minimal Wan2.1-T2V inference frontend with two attention backends and isolated
profiling programs.

## Layout

```text
infer.py                 single inference frontend
attention_backends/
  dense.py               PyTorch SDPA backend
  sparse.py              full QxK matrix sparse reference backend
profiles/
  gemm.py                attention-shaped GEMM benchmark
  decision.py            top-mass route decision benchmark
  kernel.py              dense versus sparse reference kernel benchmark
  matrix_sparsity.py      real full-matrix sparsity probe
  hotspot_evolution.py    cross-step new-hotspot and suppression experiment
  single_query.py         real single-query and spatial 2x2 probe
  heatmap.py              spatial 2x2 heatmap renderer
results/                  current evidence and reports
```

The sparse backend is a correctness/reference implementation. It uses Python
gather/scatter operations and is not a production acceleration kernel. The
single-query spatial 2x2 route remains a separate profiling experiment; sparse
inference instead routes full QxK matrix blocks whose axes are per-frame HxW
spatial tiles.

## Inference

```bash
python3 infer.py --backend dense --output output-dense.mp4

python3 infer.py --backend sparse --tile 64 --policy reuse \
  --output output-sparse.mp4

# Experimental Triton output kernel (keeps the PyTorch route-statistics path).
python3 infer.py --backend sparse --tile 64 --policy reuse \
  --triton-sparse --output output-sparse-triton.mp4
```

The fused Triton path is used for `reuse`, where no per-step centroid
statistics are required. `directional` currently uses the PyTorch reference
path so it does not compute attention twice; its Triton statistics fusion is a
separate optimization task.

For sparse inference, `--tile` is a target spatial tile area rather than a
linear token count. The backend reads Wan's `(F, H, W)` token grid and chooses
an `H-tile x W-tile` shape with sides that exactly divide `H` and `W`. Each
frame is tiled independently, so spatial boundary padding is not used.
Existing result files were produced with the earlier linear-token tiles and
must be regenerated before comparing sparsity or performance with this layout.
The `directional` policy tracks independent Q/K `(y, x)` centroids inside each
spatial tile. Every eligible source can add one Q-only step, one K-only step,
and one joint Q/K step; each direction moves by at most one spatial block, with
no global cap across sources. Out-of-grid moves are discarded rather than
wrapped.
Blocks below `--drop-factor` times their head's average selected mass are
dropped (`0.1` by default), while the strongest original K block is retained
if a Q row becomes empty. Top-mass selection is used only to bootstrap the
first dense route; later masks are not reselected and contract only through
this explicit drop rule. The `all` policy adds at most eight neighbors by
applying the same spatial `(dy, dx)` to the Q and K block coordinates; it does
not form their 3x3 Cartesian product.

Common options:

```text
--model-dir /root/.cache/wan2.1-14b
--wan-repo /root/Wan2.1
--size 832*480
--frames 17
--steps 5
--seed 0
```

## Profiles

```bash
python3 -m profiles.gemm
python3 -m profiles.decision
python3 -m profiles.kernel --tokens 3648
python3 -m profiles.matrix_sparsity --frames 5 --steps 5
python3 -m profiles.hotspot_evolution --frames 5 --steps 5 --mode observe
python3 -m profiles.single_query --frames 5 --steps 5
python3 -m profiles.heatmap
```

The real-model profilers use dense SDPA and save only bounded aggregate data.
They do not write full QK or attention tensors.

To test whether important tiles appear suddenly outside the previous route,
first generate a same-seed dense reference and then run the counterfactual
suppression profile:

```bash
python3 infer.py --backend dense --frames 5 --steps 5 \
  --output results/hotspot_dense.mp4
python3 -m profiles.hotspot_evolution --frames 5 --steps 5 --tile 64 \
  --mode suppress --reference-video results/hotspot_dense.mp4 \
  --video results/hotspot_suppress.mp4 \
  --result results/hotspot_evolution_suppress.json
```

The experiment defines a sudden hotspot as a tile in the current exact 90%
mass route but outside a one-tile coupled Q/K spatial dilation of the previous
exact route, using the same spatial layout as the sparse backend. The
suppression run returns attention restricted to that predicted route and
reports attention-output error plus final-video PSNR/SSIM.  It intentionally
retains dense QK for oracle measurement, so its runtime is not a speed result.
The full protocol and suggested acceptance thresholds are in
`profiles/HOTSPOT_EVOLUTION.md`.

Current conclusions are in `results/FINAL_REPORT_ZH.md` and
`results/SINGLE_QUERY_SPARSITY_ZH.md`.
