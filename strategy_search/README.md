# Strategy search

用于搜索 Wan sparse attention 的最佳策略与参数组合。

后续实验脚本可以放在这里，统一比较：

- `policy`：`reuse`、`directional`、`all`
- 空间 tile 大小（`tile`）
- 保留比例（`keep`）
- 目标质量（`mass_target`）
- 低质量 block 丢弃阈值（`drop_factor`）

建议每次搜索同时记录运行时间、显存占用、选中 block 比例，以及生成质量指标，避免只按单一指标选择策略。

## Drop-factor sweep

```bash
python3 strategy_search/sweep_drop_factor.py \
  --drop-factors 0,0.025,0.05,0.1,0.15,0.2,0.3,0.5 \
  -- --wan-repo /root/Wan2.1 \
     --model-dir /root/.cache/wan2.1-14b \
     --size 832*480 --frames 17 --steps 5 \
     --tile 64 --policy directional
```

每个 `drop_factor` 会生成一个 JSONL 文件，并在
`strategy_search/results/drop_factor_summary.json` 汇总。首个 dense step 的
`route_mass_fraction` 是真实 dense attention mass 覆盖率；后续 sparse step
该字段为 `null`，避免把 sparse softmax 的重新归一化结果误当成 dense coverage。
`executed_tile_fraction`
是当前 step 实际执行的矩阵 tile 比例；`next_*` 是丢弃低质量 tile 后下一
step 的预测值。
