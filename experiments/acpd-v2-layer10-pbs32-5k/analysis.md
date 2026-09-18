# H9 分析

## 结果

| 方法 | Spatial | Object | Goal | LIBERO-10 | Pooled |
|---|---:|---:|---:|---:|---:|
| H8 ACL-only | 2.00% | 15.40% | 7.60% | 0.40% | 6.35% |
| H9 ACPD-v2 | 5.20% | 24.20% | 15.60% | 0.40% | 11.35% |

H9 相对 H8 的 pooled 差值为 `+5.00` 个百分点。对相同 2,000 个 episode
进行 10,000 次 task-stratified paired bootstrap，95% CI 为
`[+3.60, +6.45]` 个百分点。

## 结论

H9 同时超过预注册的 `+1.5` 点门槛，且置信区间下界大于 0，因此支持
layer-10 exact contribution 的 5K 部署效果。该结论来自单 seed、单 student
视角，仍需重复 seed 确认。

原始验证结果位于
`/opt/liutong/openpi-5090-evals/acpd-v2-task-success-5k/layer10-exact-contribution-pbs32/4999`，
配对分析位于 `results/h9_vs_h8_paired_analysis.json`。
