# ACPD-v2-TDCA 5K 结果

TDCA 只解除预测贡献注入路径的动作损失梯度截断，其余设置与 H9-Fixed
BS64 5K 相同。两组均按相同任务和初始回合验证，每套 500 回合。

| 方法 | Spatial | Object | Goal | LIBERO-10 | Pooled |
|---|---:|---:|---:|---:|---:|
| H9-Fixed | 25.40% | 32.40% | 30.00% | 4.80% | 23.15% |
| TDCA | 18.20% | 35.00% | 28.40% | 3.80% | 21.35% |

TDCA 减 H9-Fixed 的 pooled 差值为 `-1.80` 个百分点，task-stratified
paired bootstrap 95% CI 为 `[-3.90, +0.35]` 个百分点。区间包含零，
不支持 TDCA 提高 5K pooled 成功率，也不足以断言其总体效果更差。
Spatial 单套差值为 `-7.20` 点，95% CI `[-12.00, -2.20]`；
这是分项探索结果，不替代预注册的 pooled 判据。

原始验证：`/opt/liutong/openpi-5090-evals/acpd-v2-tdca-5k/4999/summary.txt`。
配对结果：`results/tdca_vs_h9_5k_paired_analysis.json`。
