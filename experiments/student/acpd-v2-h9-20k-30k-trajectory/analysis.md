# H9 25K 轨迹分析

## 结果

每个 benchmark 使用 500 episodes，共 2,000 episodes。

| Checkpoint | Spatial | Object | Goal | LIBERO-10 | Pooled |
|---|---:|---:|---:|---:|---:|
| H9 25K | 63.20% | 69.40% | 62.00% | 28.40% | 55.75% |
| H9 30K | 64.00% | 73.00% | 61.80% | 33.20% | 58.00% |

25K 相对 30K 的 pooled 差值为 `-2.25` 点，配对 bootstrap 95% CI 为
`[-4.70, +0.20]`。

## 结论

结果不满足预注册的“25K 高于 30K且置信区间下界大于 0”判据，不支持 H9 在
25K 已经早于 30K 达峰。该结果只比较 H9 内部轨迹；匹配 SFT 25K 完成前，不能判断
ACPD-v2 在 25K 是否仍优于同进度 SFT。

## 证据

- 验证汇总：`/opt/liutong/openpi-5090-evals/slurm-log/sum-bv-h9-25k_132084.out`
- 成对分析：`results/h9_25k_vs_30k_paired_analysis.json`
