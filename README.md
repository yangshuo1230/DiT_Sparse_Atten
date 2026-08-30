# Wan attention study

Minimal Wan2.1-T2V inference frontend with pluggable attention backends and isolated
profiling programs.

## Layout

```text
infer.py                 single inference frontend
attention_backends/
  dense.py               PyTorch SDPA backend
  flex.py                minimal static-route FlexAttention backend
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
  --output output-flex-reuse.mp4

# Experimental Triton output and route-statistics kernels.
python3 infer.py --backend sparse --tile 64 --policy directional \
  --triton-sparse --output output-sparse-triton.mp4
```

`flex_reuse` can now periodically refresh its exact top-mass route instead of
keeping the first-step mask forever. `--flex-update-interval 1` updates every
denoising step, while larger values amortize the `BlockMask` construction
cost. On the synthetic 57,600-token benchmark, later-step averages are about
`1.90 s` at interval 1, `1.10 s` at interval 2, and `0.81 s` at interval 5
with 50% keep. Larger intervals trade route freshness for speed; the historical
fully-static behavior remains available with a very large interval.

The fused Triton path is used for both `reuse` and `directional`. The output
kernel writes each query's online-softmax max and sum; a second Triton kernel
recomputes QK once and uses that state to emit tile mass and Q/K centroids.
Compared with the PyTorch reference path, the synthetic 57,600-token
directional benchmark improves from about 8.41 seconds to 4.04 seconds
(about 2.08x) at 62.5% keep. Correctness is covered by
`tests/test_sparse_triton_stats.py`.

### Minimal FlexAttention closure

`flex_reuse` is a deliberately limited kernel-validation backend:

1. The first denoising step runs dense FlexAttention once to obtain the normal
   output and each query's log-sum-exp (LSE).
2. A second exact Triton QK pass uses that LSE to reduce probabilities directly
   into 128x128 block mass without materializing an attention matrix.
3. The per-head, per-query-block top-mass route is converted to a PyTorch
   FlexAttention `BlockMask`.
4. Every later step for the same layer and CFG branch reuses that unchanged
   mask. Cross-attention remains on dense SDPA.

The block size is fixed to contiguous linear token groups (`128` by default),
not the per-frame spatial HxW tiles used by the reference sparse backend.
Sequences are padded to a block boundary only inside FlexAttention and trimmed
afterward. `3120` tokens therefore become `3200`; `57600` is already divisible
by `128`.

Current version boundaries:

- no route update, expansion, drop policy, centroid, sampled statistics, or
  adaptive contraction after the dense bootstrap;
- no automatic dense/Flex layer dispatch and no fixed K-count buckets yet;
- `create_block_mask` construction and first-use compilation are part of the
  run, but have not yet been optimized or comprehensively benchmarked;
- bootstrap route mass is exact (up to normal kernel floating-point error) and
  uses no sampling, but it still pays for a second full QK pass;
- the exact-mass path currently requires CUDA/Triton and 128-token route
  blocks; its two 128x64 subtiles use atomic addition to form each block mass;
- only batch-1 global self-attention uses FlexAttention; unsupported modes and
  all cross-attention calls fall back to dense SDPA;
- this is a performance/correctness prototype. Its static route can accumulate
  quality error across denoising steps and should not be treated as the final
  sparse inference policy.

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
