# H13：ACPD-v2 注入消融结果

## 5K 验证

| 模型 | Spatial | Object | Goal | LIBERO-10 | Pooled |
|---|---:|---:|---:|---:|---:|
| H13 loss-only | 19.40% | 35.60% | 25.00% | 2.40% | 20.60% |
| H9 显式注入 | 25.40% | 32.40% | 30.00% | 4.80% | 23.15% |

H9 相对 H13 的 pooled 差值为 `+2.55` 个百分点。10,000 次 task-stratified
paired bootstrap 的 95% CI 为 `[+0.40, +4.70]` 个百分点，达到预注册的
`delta >= 1.5` 且 CI 下界大于 `0` 的判据。因此，这组 5K 验证支持显式
contribution 注入有效。Object 单项下降 `3.20` 点，其余三套均提高。

原始配对分析：`results/h9_vs_h13_paired_analysis.json`。

