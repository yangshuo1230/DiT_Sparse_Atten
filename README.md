# Wan attention study

Minimal Wan2.1-T2V inference frontend with pluggable attention backends and isolated
profiling programs.

## Layout

```text
infer.py                 single inference frontend
attention_backends/
  dense.py               PyTorch SDPA backend
  context.py             runtime Wan layer/step/CFG/grid context injection
  flex.py                reusable-route FlexAttention backend
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

python3 infer.py --backend flex_reuse --flex-block 128 \
  --flex-update-interval 0 \
  --output output-flex-reuse.mp4

# Validated 57,600-token hybrid with one lightweight update in five steps.
python3 infer.py --backend flex_reuse --size 1280*720 --frames 61 --steps 5 \
  --flex-bootstrap sampled --flex-bootstrap-prefetch \
  --flex-sampled-update-interval 2 --no-flex-sampled-prefetch \
  --flex-route-samples 16 --flex-route-persistence 0.5 \
  --mass-target 0.95 --keep 0.625 \
  --flex-dense-route-threshold 0.58 \
  --timing results/flex-timing.json --output output-flex.mp4

# Faster spatial packing variant (1.0556x model-forward speedup).
python3 infer.py --backend flex_reuse --size 1280*720 --frames 61 --steps 5 \
  --flex-bootstrap sampled --flex-bootstrap-prefetch \
  --flex-spatial-reorder --flex-sampled-update-interval 2 \
  --no-flex-sampled-prefetch --flex-route-samples 16 \
  --flex-route-persistence 0.25 --mass-target 0.95 --keep 0.625 \
  --flex-dense-route-threshold 0.58 --output output-spatial.mp4

# Quality-biased spatial frontier/directional variant.
python3 infer.py --backend flex_reuse --size 1280*720 --frames 61 --steps 5 \
  --flex-bootstrap sampled --flex-bootstrap-prefetch \
  --flex-spatial-reorder --flex-directional-update \
  --flex-direction-min-ratio 0 --flex-direction-bonus 0.25 \
  --flex-route-budget-scale 1.05 --flex-route-exploration 0.02 \
  --flex-sampled-update-interval 2 --no-flex-sampled-prefetch \
  --flex-route-samples 16 --flex-route-persistence 0.25 \
  --mass-target 0.95 --keep 0.625 --flex-dense-route-threshold 0.58 \
  --output output-spatial-direction.mp4

# Experimental Triton output and route-statistics kernels.
python3 infer.py --backend sparse --tile 64 --policy directional \
  --triton-sparse --output output-sparse-triton.mp4
```

`flex_reuse` can periodically refresh its exact top-mass route. The default
`--flex-update-interval 0` keeps the step-0 mask static. A positive value
performs an exact dense refresh every N denoising steps; `1` therefore refreshes
every step and executes no sparse reuse steps. Exact refresh includes dense
FlexAttention, a second complete QK pass, route selection, and `BlockMask`
construction. It is a quality reference, not a lightweight route update.

The lightweight path is enabled with `--flex-sampled-update-interval N`. It
samples Q/K tokens inside each 128-token block, estimates block mass without
dense attention, adds a previous-route persistence prior, and rebuilds the
mask. Synchronous updates use the current step immediately. Optional
`--flex-sampled-prefetch` prepares a route for the next step on a side stream;
it is faster but adds prediction lag and was not used by the validated quality
configuration below.

`--flex-bootstrap sampled` still returns complete dense SDPA output at step 0;
only route measurement is sampled. With `--flex-bootstrap-prefetch`, that route
is prepared alongside the remaining transformer work. Per-layer/CFG routes at
or above `--flex-dense-route-threshold` remain on dense SDPA, since Flex is not
faster for high keep fractions.

`--flex-spatial-reorder` partitions Wan's `(F,H,W)` grid into exact spatial
microtiles whose area divides 128, orders them by a Morton/Z traversal, and
packs them into the existing 128-token route blocks without changing sequence
length or block count. At 720p it uses `1x16` microtiles, eight per route block;
Q/K/V are jointly reordered after RoPE and outputs are restored afterward.

`--flex-directional-update` maintains a separate frontier mask. Only selected
QxK cells with an open spatial neighbour generate Q-only, K-only, or joint
candidates. Candidate and old blocks are rescored, then pruned to
`--flex-route-budget-scale` times the previous per-row budget. A small global
exploration fraction avoids making spatial locality a hard assumption.

The fused Triton path is used for both `reuse` and `directional`. The output
kernel writes each query's online-softmax max and sum; a second Triton kernel
recomputes QK once and uses that state to emit tile mass and Q/K centroids.
Compared with the PyTorch reference path, the synthetic 57,600-token
directional benchmark improves from about 8.41 seconds to 4.04 seconds
(about 2.08x) at 62.5% keep. Correctness is covered by
`tests/test_sparse_triton_stats.py`.

### FlexAttention closure

`flex_reuse` supports two bootstrap modes:

1. `exact` runs dense FlexAttention with LSE, followed by an exact Triton QK
   mass pass. This is the oracle/reference route.
2. `sampled` runs complete dense SDPA output and estimates only the next route
   from sampled Q/K blocks. It never materializes the full attention matrix.
3. Routes become compressed FlexAttention `BlockMask` objects. Later steps
   reuse them, optionally update them, or dispatch high-keep routes to dense.
4. Cross-attention and unsupported modes remain on dense SDPA.

The Flex block size remains 128. Without spatial reordering these are contiguous
linear token groups. With spatial reordering, each block packs complete nearby
image microtiles. Sequences are padded to a block boundary only inside
FlexAttention and trimmed afterward. `3120` tokens therefore become `3200`;
`57600` is already divisible by `128`.

Current version boundaries:

- exact refresh and sampled lightweight updates are separate explicit modes;
- automatic dense/Flex dispatch is available, but fixed K-count buckets and
  learned/adaptive sampling are not yet implemented;
- divisible sequences build BlockMask directly from packed KV indices; a
  partial final block retains the mature element-mask builder;
- CUDA route metadata uses stable linear compaction for both KV and transposed
  Q traversal, avoiding the framework builder's dense conversion and row-wise
  sorts; immutable empty partial-block metadata is shared across routes;
- spatial Q/K/V placement is fused into one contiguous-feature Triton gather,
  following the fused placement pattern used by Sparse-VideoGen;
- sparse FlexAttention runs declare the route's one-K-block-per-row invariant
  through `ROWS_GUARANTEED_SAFE`, removing empty-row guards from the generated
  kernel without changing selected blocks or softmax semantics;
- exact bootstrap still pays for a second full QK pass, while sampled bootstrap
  trades route accuracy for much lower startup cost;
- the exact-mass path currently requires CUDA/Triton and 128-token route
  blocks; its two 128x64 subtiles use atomic addition to form each block mass;
- only batch-1 global self-attention uses FlexAttention; unsupported modes and
  all cross-attention calls fall back to dense SDPA;
- the final denoising step releases route and BlockMask state before VAE decode;
  without this, 57K metadata can prevent selection of a VAE convolution engine;
- this remains a research prototype; the current matched validation covers one
  prompt and seed and is not a general quality claim.

Use `--no-flex-compile` only for debugging. The normal path compiles
FlexAttention once and reuses the compiled kernel and per-layer BlockMasks.

Measured validation on the current PPU environment:

- 3,120 tokens, two denoising steps: exact bootstrap `6.07 s`, reused Flex step
  about `2.10 s`; the previous explicit-probability bootstrap took `129.9 s`.
- 57,600 tokens, 50% keep: exact bootstrap `178.2 s`, reused Flex step about
  `63.5 s`, versus dense `81.4 s/step`. The reused kernel is about `1.28x`
  faster, while a two-step run remains slower overall because of bootstrap.

The 57K validation completed both denoising steps; its subsequent VAE decode
failed in the device convolution engine, after timing had already been saved.
These numbers establish the attention path only and are not yet a production
end-to-end speedup claim. The compact record is in
`results/flex_lse_57k_summary.json`.

The newer sampled/hybrid implementation closes that startup gap. On matched
1280x720, 61-frame, 5-step Wan2.1-T2V-14B inference:

- dense model forwards: `406.583 s`;
- sampled-bootstrap Hybrid with one step-2 update: `386.919 s`, or `1.0508x`;
- steady Hybrid steps: about `72.70 s` versus dense `81.26 s`;
- 15 of 80 layer/CFG routes dispatched to dense; the other 65 used Flex;
- matched output quality: `PSNR 28.38 dB`, `SSIM 0.8517` over 61 frames;
- both dense and Hybrid completed VAE decode and wrote video.

The exact command, stage profile, limitations, and artifact names are in
`results/FLEX_ITERATION_REPORT_ZH.md`.

The spatial follow-up produced two useful Pareto points on the same matched
run:

- spatial packing + global sampled update: `385.161 s` (`1.0556x`), PSNR
  `27.65 dB`, SSIM `0.8507`;
- spatial frontier/directional update: `395.330 s` (`1.0285x`), PSNR
  `27.70 dB`, SSIM `0.8573`.

Direction expansion improves both quality metrics relative to the same spatial
baseline, but costs about 10.17 seconds. The maintained 57K frontier is only
20.72% of matrix cells; nevertheless the current update still computes global
16-sample mass, so frontier-only candidate generation does not remove the main
update cost.

### Reproducibility

The frontend injects the current attention ID, denoising step, CFG branch, and
token grid into Wan at runtime through `attention_backends/context.py`. It does
not require edits to the Wan checkout. The adapter also installs the portable
`clamp/round/to(uint8)` video conversion needed by some accelerator builds.
Timing is collected by the same runtime adapter; `--timing result.json` writes
the complete command configuration,
study/Wan Git revisions and dirty state, device and PyTorch version, plus
per-model-forward CUDA timings even if a later VAE decode fails.

The currently validated Wan base revision is
`9737cba` (`Update README with community projects using Wan2.1 (#582)`). A
minimal setup is:

```bash
git clone https://github.com/Wan-Video/Wan2.1.git /path/to/Wan2.1
git -C /path/to/Wan2.1 checkout 9737cba
pip install -r /path/to/Wan2.1/requirements.txt
pip install -r requirements-study.txt

python3 infer.py --wan-repo /path/to/Wan2.1 \
  --model-dir /path/to/Wan2.1-T2V-14B \
  --backend dense --steps 2 --timing results/dense_timing.json
```

Model weights are intentionally not stored in this repository. Flex exact-mass
profiling additionally requires a CUDA/accelerator PyTorch build with
FlexAttention and Triton support.

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
python3 -m profiles.frontend.cli
python3 -m profiles.gemm
python3 -m profiles.decision
python3 -m profiles.kernel --tokens 3648
python3 -m profiles.matrix_sparsity --frames 5 --steps 5
python3 -m profiles.hotspot_evolution --frames 5 --steps 5 --mode observe
python3 -m profiles.single_query --frames 5 --steps 5
python3 -m profiles.heatmap
python3 -m profiles.spatial_direction_search --frames 5 --steps 5
python3 -m profiles.video_quality reference.mp4 candidate.mp4 \
  --output results/video_quality.json
```

The unified benchmark is split into a CLI/report frontend
(`profiles/frontend/cli.py`) and measurement backends:
`profiles/backend/comparison.py` for dense/PyTorch-sparse/Triton-sparse timing
and quality, `profiles/backend/metrics.py` for numerical errors, and
`profiles/backend/timing.py` for CUDA timing and memory. Its current report is
`results/sparse_attention_profile.json`; the condensed interpretation is
`results/sparse_attention_profile_summary.json`.

The real-model profilers use dense SDPA and save only bounded aggregate data.
They do not write full QK or attention tensors. Profilers share one attention
and forward installation helper (`profiles._runner.install_probe`), so all Wan
attention call sites and the timestep hook stay synchronized; the shared
behavior is covered by `tests/test_profiler_runner.py`.

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
