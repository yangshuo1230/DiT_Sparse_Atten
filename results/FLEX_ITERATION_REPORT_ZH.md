# FlexAttention 轻量路由迭代报告

## 结论

当前实现已经在真实 Wan2.1-T2V-14B、1280×720、61 帧、5 denoising steps
上同时获得速度和基本 matched 质量优势：

| 配置 | model forward 总时间 | 相对 dense |
|---|---:|---:|
| Dense SDPA | 406.583 s | 1.0000× |
| Sampled-bootstrap Hybrid | 386.919 s | **1.0508×** |

Hybrid 节省 19.664 s。最终视频共 61 帧，与相同 prompt/seed 的 dense 视频相比：

| 指标 | 均值 | p10 | p50 | p90 |
|---|---:|---:|---:|---:|
| PSNR | **28.378 dB** | 27.736 | 28.349 | 29.394 |
| SSIM | **0.8517** | 0.8295 | 0.8508 | 0.8707 |

结果超过本项目使用的 `PSNR > 25 dB`、`SSIM > 0.83` 工程门槛。当前环境未安装
LPIPS 和 VBench，因此这两个指标尚未补测；同时只有一个 prompt 和 seed，不能外推为
通用质量结论。

## 空间化与方向边缘扩展

后续迭代把 RoPE 后的 Q/K/V 按图像空间重排。720p token grid 为
`16×45×80`，选择 `1×16` exact microtile，按 Morton/Z-order 排列，每8个
microtiles组成一个128-token Flex block。序列仍是57,600 tokens和450 blocks，
没有空间 padding；Q/K/V联合重排，attention输出再逆排。

在相同 prompt、seed、5 steps 下：

| 配置 | Model forward | Speedup | PSNR | SSIM |
|---|---:|---:|---:|---:|
| Dense | 406.583 s | 1.0000× | — | — |
| Linear sampled Hybrid | 386.919 s | 1.0508× | 28.378 | 0.8517 |
| Spatial + global sampled update | **385.161 s** | **1.0556×** | 27.652 | 0.8507 |
| Spatial + frontier directional | 395.330 s | 1.0285× | 27.698 | **0.8573** |

因此保留两档推荐：

- 速度优先：空间重排 + global sampled update；
- SSIM优先：空间重排 + frontier directional update。

方向版本为每条 route 独立保存 frontier mask。只有 frontier cells 产生 Q-only、
K-only、joint candidates；参数搜索覆盖328组 persistence、方向阈值、candidate bonus、
预算比例、全局探索比例和joint开关。最终质量点为：

```text
persistence       = 0.25
direction ratio   = 0
candidate bonus   = 0.25
budget scale      = 1.05
global exploration= 0.02
joint expansion   = true
```

57K平均 frontier fraction 为20.72%，平均route keep为53.03%，20/80条
layer/CFG routes走dense。方向扩展相对同一空间global基线提高PSNR约0.046 dB、
SSIM约0.0066，但增加约10.17 s。

将方向候选生成改为 `frontier.nonzero()` 稀疏索引后，57K总时间仅从395.645 s降到
395.330 s。原因是当前更新仍先计算完整的16-sample block-mass矩阵，并仍需完整
BlockMask排序/重建；因此“边缘mask”降低了扩展候选范围，但尚未消除主要更新成本。
下一步若继续优化，必须让 sampled-mass 本身只计算 active+frontier candidates，或用
融合Triton kernel直接输出候选分数。

## 最终方案

```text
step 0: 完整 dense SDPA 输出
          + 16-sample/block 路由估计（侧流）
          ↓
按 route keep 分派：keep >= 58% → dense，否则 FlexAttention
          ↓
step 1: 复用 step-0 mask
step 2: 对 Flex 路由做一次同步 sampled Q/K 轻量更新
step 3-4: 复用更新后的 mask
          ↓
最终 step 逐层释放 route/BlockMask，再进入 VAE
```

完整命令：

```bash
python3 infer.py --backend flex_reuse \
  --size '1280*720' --frames 61 --steps 5 --seed 0 \
  --keep 0.625 --mass-target 0.95 \
  --flex-bootstrap sampled --flex-bootstrap-prefetch \
  --flex-sampled-update-interval 2 --no-flex-sampled-prefetch \
  --flex-route-samples 16 --flex-route-persistence 0.5 \
  --flex-dense-route-threshold 0.58 \
  --output results/final_flex.mp4 \
  --timing results/final_flex_timing.json
```

本次实际运行中，80 条 `(layer, CFG branch)` 路由有 15 条走 dense、65 条走
Flex。step 2 更新 65 条路由，更新后的全体平均 keep 为 50.59%。

质量复算命令：

```bash
python3 -m profiles.video_quality \
  results/final_dense_57k_5step.mp4 \
  results/final_flex_hybrid058_update2_release_57k_5step.mp4 \
  --size '1280*720' --denoising-steps 5 --seed 0 \
  --prompt 'A small white dog running on a beach at sunset' \
  --output results/final_flex_hybrid058_update2_quality.json
```

## 分 step 性能

每个 denoising step 包含两个 CFG forward：

| Step | Dense | Hybrid | 说明 |
|---:|---:|---:|---|
| 0 | 81.521 s | 87.584 s | 完整 dense 输出 + sampled bootstrap |
| 1 | 81.269 s | 76.007 s | 包含首次 Flex 编译 |
| 2 | 81.262 s | 77.932 s | 同步轻量 mask 更新 |
| 3 | 81.271 s | 72.699 s | 稳态复用 |
| 4 | 81.259 s | 72.698 s | 稳态复用并释放路由 |

稳态约 `1.118×`，启动和更新成本摊销后五步为 `1.0508×`。

## 算子分解与瓶颈

57,600 tokens、40 heads、head dim 128 的单 attention profile：

| 阶段 | 时间 |
|---|---:|
| Dense SDPA | 0.678 s |
| Dense Flex + LSE | 1.107 s |
| Exact second-QK mass | 0.746 s |
| Exact refresh 合计 | 1.859 s |
| 16-sample mass | 0.074 s |
| Route selection | 0.0014 s |
| BlockMask build | 0.0042 s |
| Flex output（62.5% synthetic keep） | 0.704 s |

因此已经得到明确结论：

1. 原 exact refresh 的绝对瓶颈是第二遍完整 QK；它比完整 dense SDPA 本身还慢，不能作为频繁更新方案。
2. sampled mass 将路由测量降到约 74 ms，但 62.5% keep 时 Flex 输出本身仍慢于 dense。
3. 自动 dense/Flex 分派是获得净优势的必要条件；只有实际 keep 低于交叉点的路由才值得进入 Flex。
4. BlockMask 构建只有约 4 ms，不是当前计算瓶颈，但其双向索引在 80 条 57K 路由上会形成显著显存常驻。
5. 更新时必须先释放旧 mask，最终 step 必须释放全部状态；否则会 OOM 或使 VAE 卷积找不到可执行引擎。

## 可复现证据

- Dense timing：`final_dense_57k_5step_timing.json`
- Hybrid timing：`final_flex_hybrid058_update2_release_57k_5step_timing.json`
- Matched quality：`final_flex_hybrid058_update2_quality.json`（包含输入视频、prompt、seed 和计算方法）
- 57K stage profile：`flex_route_profile_target95_keep625.json`
- Real-Q/K route profile：`flex_route_quality_target95_5f.json`
- Spatial parameter search：`spatial_direction_search_route_5f.json`
- Spatial speed timing/quality：`final_spatial_global_57k_5step_timing.json`, `final_spatial_global_57k_5step_quality.json`
- Spatial directional timing/quality：`final_spatial_direction_edge_57k_5step_timing.json`, `final_spatial_direction_edge_57k_5step_quality.json`

Timing JSON 使用 schema v2，包含完整参数、PyTorch/设备信息、study/Wan Git revision
和 dirty 状态、每个 CFG forward CUDA 时间、路由 phase counts。运行时上下文由本仓库
`attention_backends/context.py` 注入；已使用干净的 Wan2.1 `9737cba` worktree 验证导入和
layer/step/CFG/grid 注入。加速器需要的 portable uint8 视频转换也由该适配器安装，
不依赖 `/root/Wan2.1` 的本地修改。

## 后续工作

- 用至少 3 prompts × 3 seeds 重复最终配置，并补 LPIPS/VBench/盲评。
- 共享 BlockMask 中与 route 无关的空 transpose 元数据，进一步降低 57K 常驻显存。
- 用真实 Q/K 搜索分层 sample 数、mass target 和 dense 阈值，而不是所有层统一参数。
- 探索 Triton sampled-mass 融合和固定 K-count bucket；当前 16-sample PyTorch matmul 已不再是首要瓶颈。
