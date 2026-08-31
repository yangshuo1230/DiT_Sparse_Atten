# Wan2.1-14B 完整 Attention 方阵 Tile 稀疏实验报告

> 2026-08-31 更新：本文第 1～10 节记录早期 matrix-sparse reference 路线，
> 其中“下一工程重点”已经完成一次 FlexAttention 迭代。新的 sampled-bootstrap
> Hybrid 在 57,600 tokens、5 steps 上取得 `1.0508×` model-forward 加速，matched
> 视频为 PSNR `28.38 dB`、SSIM `0.8517`。实现、命令、算子分解和限制见
> `FLEX_ITERATION_REPORT_ZH.md`。旧 Python/Triton matrix-sparse 性能数字仍作为
> 历史对照，不代表当前最快实现。

> 空间化后续迭代进一步验证：Morton-packed spatial route 在不增加token/block数量
> 的情况下达到 `1.0556×`；维护20.72% frontier并做方向扩展可将SSIM从`0.8507`
> 提升到`0.8573`，但加速降为`1.0285×`。方向扩展是质量/速度Pareto选项，不是
> 对空间全局更新的无条件替代。完整数据同样见 `FLEX_ITERATION_REPORT_ZH.md`。

> 同期 hotspot suppression 反事实实验并未支持“局部演化足以保证质量”：单
> prompt/seed 下 sudden dense mass 均值约 `0.763%`、预测外总 mass 约 `5.14%`，
> 最终视频 PSNR `17.36 dB`、SSIM `0.622`，均未达到既定门槛。因此当前 Hybrid
> 保留 persistence、周期 sampled update 和 dense/Flex 自动分派，不能把简单空间
> 邻域扩展当作已验证方案。

## 1. 修正后的研究对象

当前实验研究每个denoising step、transformer layer、attention head和CFG branch的完整attention矩阵：

```text
A = softmax(QK^T) ∈ R^(Lq × Lk)
```

softmax对每个query行独立计算。之后才同时沿Q轴和K轴切分二维矩阵tile：

```text
T[i,j] = A[query tile i, key tile j]
mass(T[i,j]) = sum(A[q,k]) / Lq
```

当前有效结果不再把多个query平均成一个K轴mask。所有mask均为`[layer, head, q_tile, k_tile]`。完整QK/attention矩阵从不写盘，probe按query chunk流式计算，仅保存聚合结果。

共同配置：Wan2.1-T2V-14B、单卡约96GB、SDPA dense基线、seed 0、至少5 denoising steps、VBench `custom_input/imaging_quality`。

## 2. 16×16矩阵Tile稀疏性

真实probe使用832×480、5帧、5 steps、16×16 attention-elements tile、query chunk 256。共统计400次self-attention，cross-attention单独跳过。每次均覆盖全部query和全部key。

| Step | 全局top-mass keep | 可执行逐Q-tile keep | 覆盖mass |
|---:|---:|---:|---:|
| 0 | 56.09% | 56.66% | 90.21% |
| 1 | 51.21% | 51.87% | 90.24% |
| 2 | 45.01% | 45.84% | 90.26% |
| 3 | 42.40% | 43.27% | 90.28% |
| 4 | 36.72% | 37.66% | 90.32% |

- 全局二维top-mass五步平均keep：46.29%，即平均可跳过53.71%的矩阵tiles。
- 可执行逐query-tile路由平均keep：47.06%，即每个Q tile独立选择K tiles后可跳过52.94%。
- 1600个layer-head组合的全局keep范围：0.62%～88.68%，中位数48.01%。
- 逐Q-tile keep范围：0.81%～88.98%，中位数48.62%。

全局mask略稀疏，但无法保证每个query tile有可计算的key；实际sparse kernel使用逐Q-tile mask。证据见`real_matrix_locality_probe.json`。

## 3. 二维矩阵局部性

二维8邻域定义在attention方阵tile坐标`(q_tile,k_tile)`上，不是图像K空间邻域。质心也分别计算tile内部的query轴偏移和key轴偏移。

### 3.1 16×16逐Q-tile路由

| 策略 | 下一step recall | 下一step mass | 预测矩阵tile比例 |
|---|---:|---:|---:|
| Reuse-only | 87.17% | 88.40% | 49.41% |
| Directional `c=.15,m=.5,h=2` | 89.22% | 90.07% | 53.19% |
| Directional `c=0,m=0,h=1.5` | 91.82% | 92.25% | 59.23% |
| 全8邻域 | 97.37% | 96.89% | 73.49% |

最省计算且达到90% mass的是`c=.15,m=.5,h=2`，比全扩展少20.30个百分点tile。

### 3.2 64×64大型策略probe

粒度变粗后，逐Q-tile keep五步为64.37%、58.83%、53.31%、51.25%、45.40%。Reuse-only已经达到：

- recall：91.03%
- 下一step mass：90.34%
- 预测矩阵tile比例：56.94%

因此720p实验使用64×64 tile和reuse-only，不做额外邻域扩展。证据见`real_matrix_locality_probe_tile64.json`。

## 4. 二维Matrix Sparse Kernel

实现语义：

1. 首个step使用完整dense attention，并流式建立精确二维tile mass、质心和逐Q-tile mask。
2. 第`i+1`步只计算第`i`步mask预测的`(q_tile,k_tile)` blocks。
3. 每个query的softmax仅在其当前选中的key tokens上独立计算。
4. 当前step只从实际计算过的blocks累计新mass/质心，生成下一step mask；不会暗中重算完整QK。
5. layer、head、CFG branch和Q tile全部独立。

对人工不同head/不同Q-tile mask，kernel输出与逐block朴素attention最大误差为`1.2e-7`。

## 5. 832×480、17帧Matched实验

固定seed、5 steps、16×16矩阵tile。Dense为5.028 s/step，VBench 42.206。

| 策略 | 时间/step | 相对dense | VBench | SSIM | PSNR | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| Reuse, cap 62.5% | 142.69 s | 0.0352× | 56.901 | 0.798 | 20.95 | 0.305 |
| Directional, cap 62.5% | 169.45 s | 0.0297× | 55.068 | 0.808 | 22.23 | 0.261 |
| 全8邻域, cap 62.5% | 170.76 s | 0.0294× | 50.483 | 0.828 | 21.25 | 0.224 |
| Directional, cap 75% | 171.09 s | 0.0294× | 61.103 | 0.806 | 22.05 | 0.268 |

没有配置同时满足`SSIM>0.83, PSNR>25, LPIPS<0.2`。提高cap并未单调改善与dense的相似度，因为route误差会沿denoising轨迹累积。结果见`real_matrix_sparse_sweep.json`。

## 6. 二维Mask Packing

16×16全局二维mask的确定性样本统计：

| 策略 | 单bbox利用率 | Row runs | Components | Component-bbox利用率 |
|---|---:|---:|---:|---:|
| Reuse-only | 48.71% | 4236.67 | 934.51 | 42.62% |
| Directional `c=0,m=0,h=1` | 59.93% | 2739.33 | 804.92 | 53.82% |
| 全8邻域 | 72.77% | 845.34 | 21.56 | 69.89% |

二维mask比旧K轴mask更分散。Component bbox对reuse/directional甚至低于单bbox利用率，因为大量小component的bbox总面积和launch数都很高。当前mask更适合真正的BlockMask/CSR式kernel，而不是component bbox打包。

## 7. Matrix Tile Profiling

3600 tokens、40 heads、head dim128、query chunk256、独立二维head mask：

| Tile | Keep | 理论matmul加速 | 手写kernel实测 |
|---:|---:|---:|---:|
| 16×16 | 50% | 2.0× | 0.0036× |
| 16×16 | 62.5% | 1.6× | 0.0037× |
| 32×32 | 50% | 2.0× | 0.0074× |
| 64×64 | 50% | 2.0× | 0.0138× |
| 64×64 | 62.5% | 1.6× | 0.0132× |

手写kernel被Python变长索引、padding、scatter和小matmul调度主导。设备上的PyTorch FlexAttention可以执行head-specific二维BlockMask，3600 tokens约55% mask时核心约0.05 s，但它不返回本实验下一step更新所需的tile mass/质心。结果见`profile_matrix_tiles.json`。

## 8. 720p、61帧完整矩阵实验

官方Wan对61个输出帧使用16个latent frames，DiT长度为57,600 tokens。64×64 tile产生每轴900 tiles，即每head 810,000个二维矩阵tiles。

| 策略 | 时间/step | 相对dense | VBench | SSIM | PSNR | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 81.29 s | 1.000× | 61.255 | — | — | — |
| 64×64 matrix reuse | 853.37 s | 0.0953× | 59.782 | 0.801 | 23.65 | 0.172 |

Reuse-only sparse steps保留56.94%矩阵tiles；首步dense的五步平均matmul work fraction为65.55%，理论上限1.525×。实际慢约10.5倍。LPIPS达到0.2阈值，但SSIM和PSNR未达标。结果见`real_large_720p_61f_matrix.json`和`large_matrix_estimate.json`。

## 9. 最终结论

1. 之前的K轴key-routing定义确实不等于完整attention方阵稀疏性，相关结果仅保留为历史数据。
2. 修正后的16×16完整矩阵tile五步平均可跳过53.71%，后期可跳过63.28%。
3. 可执行逐Q-tile路由与全局top-mass稀疏率接近，说明二维稀疏性不是query平均产生的假象。
4. 二维mask具有明显step间局部性；16×16定向策略用53.19% tile覆盖90.07%下一step mass。
5. 64×64大型reuse-only用56.94% tile覆盖90.34%下一step mass，理论五步上限1.525×。
6. 当前手写二维block kernel在小型和大型设置都显著慢于dense，并且质量未完全达标。
7. 下一工程重点是可同时输出attention结果和tile统计的fused/FlexAttention类kernel，而不是继续优化Python gather。

## 10. 当前有效结果

- 16×16完整矩阵probe：`real_matrix_locality_probe.json`
- 64×64完整矩阵probe：`real_matrix_locality_probe_tile64.json`
- 小型matched sweep：`real_matrix_sparse_sweep.json`
- Matrix kernel profiling：`profile_matrix_tiles.json`, `profile_matrix_tiles.csv`
- 大型理论估算：`large_matrix_estimate.json`
- 720p/61帧完整矩阵结果：`real_large_720p_61f_matrix.json`

当前代码统一使用`infer.py --backend dense|sparse`推理。矩阵稀疏、单query、kernel、GEMM和route decision profiling均位于`profiles/`，不再通过历史shell脚本修改推理环境。

旧的`real_spatial_*`、`real_sparse_*`和不含`matrix`的large结果研究的是K轴routing或共享head历史实现，不属于当前attention方阵结论。
