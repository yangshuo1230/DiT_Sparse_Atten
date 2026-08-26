# Hotspot evolution experiment

目的：检验相邻 denoising step 之间是否基本不存在无法由上一 step 空间邻域覆盖的
重要 attention tile，并用反事实生成检验忽略这些 tile 是否明显影响输出质量。

## 定义

每个 step 都用完整 dense attention 建立逐 `(layer, head, CFG branch, q_tile)` 的
90% mass oracle route。把上一 step oracle route 按 sparse backend 的 coupled Q/K
空间八邻域扩展一格，得到 `predicted_route`。

```text
sudden_hotspot = current_oracle & ~predicted_route
```

该定义比实际 sparse predictor 更有利，因为它使用上一 step 的精确 oracle route；
因此实验是在检验“局部演化假设的上限”，不是直接证明当前 predictor 已经足够好。

## 两阶段协议

### 1. Observe

```bash
python3 -m profiles.hotspot_evolution \
  --frames 5 --steps 5 --tile 64 --mode observe \
  --result results/hotspot_evolution_observe.json \
  --video results/hotspot_observe.mp4
```

模型仍返回 dense attention。主要观察：

- `sudden_oracle_tile_fraction`：当前 oracle route 中突然 tile 的比例；
- `sudden_dense_mass_fraction`：突然 tile 占完整 dense attention 的 mass；
- `all_mass_outside_prediction`：预测 route 外的全部 dense mass；
- `oracle_route_recall`：预测 route 对当前 oracle tile 的 recall；
- `predicted_tile_fraction`：获得上述覆盖率需要执行的 tile 比例。

不能只看均值，同时检查每个指标的 p90 和逐 transition 结果。

### 2. Suppress

先生成相同 prompt/seed 的 dense reference，再让所有 self-attention 只使用
`predicted_route`：

```bash
python3 infer.py --backend dense --frames 5 --steps 5 \
  --output results/hotspot_dense.mp4

python3 -m profiles.hotspot_evolution \
  --frames 5 --steps 5 --tile 64 --mode suppress \
  --reference-video results/hotspot_dense.mp4 \
  --video results/hotspot_suppress.mp4 \
  --result results/hotspot_evolution_suppress.json
```

该模式仍计算 dense QK 以获得 oracle，但返回 masked attention；因此只用于质量反事实，
不能用于计时。结果额外包含每次 attention 的 relative L2、cosine similarity、最大误差，
以及最终视频的逐帧 PSNR/SSIM。

## 建议判据

在至少 3 个 prompt × 3 个 seed 上同时满足以下条件，才把假设视为得到初步支持：

- `sudden_dense_mass_fraction`：总体均值 `< 0.5%`，p90 `< 1%`；
- `all_mass_outside_prediction`：总体均值 `< 2%`，p90 `< 5%`；
- `oracle_route_recall`：总体均值 `> 97%`，p10 `> 90%`；
- attention output relative L2：均值 `< 5%`，p90 `< 10%`；
- 视频：SSIM `> 0.83`、PSNR `> 25 dB`，并补充 LPIPS/VBench 或人工盲评；
- `predicted_tile_fraction` 必须仍明显低于 dense，否则高覆盖率没有加速价值。

这些阈值是用于筛选方案的工程门槛，不是统计学定理。若平均值合格但少数
layer/head 的尾部很差，应为这些层保留探索或周期性 dense refresh，而不是宣布
“不存在突然热点”。
