# H12 结果：匹配 BS64 SFT

H12 比较 H9-scale-b ACPD-v2 与标准 backview-only SFT。两者使用相同的
pi0.5 初始化、LoRA、seed 42、FSDP4、physical global BS64、无梯度累积、
5K optimizer steps 和 30K cosine schedule；区别是 H9 使用冻结 teacher、ACL、
exact contribution loss 和最终 hidden 注入。

| 方法 | Spatial | Object | Goal | LIBERO-10 | Pooled |
|---|---:|---:|---:|---:|---:|
| H12：匹配 SFT | 9.40% | 18.40% | 24.80% | 2.80% | 13.85% |
| H9-scale-b：ACPD-v2 | 25.40% | 32.40% | 30.00% | 4.80% | 23.15% |
| H9 - H12 | +16.00 | +14.00 | +5.20 | +2.00 | +9.30 |

10,000 次 task-stratified paired bootstrap 得到 pooled 差值 95% CI
`[+7.30, +11.35]` 个百分点。H9 达到预注册的 `+1.5` 点和正置信区间下界，
因此 H12 支持：在匹配 BS64 和训练预算下，ACPD-v2 整体训练方案优于标准 SFT。

该结果不能单独区分 ACL、contribution loss 和显式注入的贡献；H13 负责隔离
显式注入。原始配对结果位于 `results/h9_vs_h12_paired_analysis.json`。

## 30K 结果

| 方法 | Spatial | Object | Goal | LIBERO-10 | Pooled |
|---|---:|---:|---:|---:|---:|
| H12：匹配 SFT | 70.00% | 69.80% | 68.20% | 33.80% | 60.45% |
| H9-scale-b：ACPD-v2 | 64.00% | 73.00% | 61.80% | 33.20% | 58.00% |
| H9 - H12 | -6.00 | +3.20 | -6.40 | -0.60 | -2.45 |

10,000 次 task-stratified paired bootstrap 得到 pooled 差值 95% CI
`[-5.00, 0.00]` 个百分点。Spatial 和 Goal 的差值分别为 `-6.00` 和
`-6.40` 点，置信区间均低于零；Object 的 `+3.20` 点和 LIBERO-10 的
`-0.60` 点均未排除零。

因此，5K 的 ACPD-v2 优势没有保持到 30K。20K/25K 验证用于判断优势消失前
是否存在优于双方 30K 的中间 checkpoint。原始配对结果位于
`results/h9_vs_h12_30k_paired_analysis.json`。

## 25K 结果

| 方法 | Spatial | Object | Goal | LIBERO-10 | Pooled |
|---|---:|---:|---:|---:|---:|
| H12：匹配 SFT | 64.00% | 64.00% | 65.60% | 29.60% | 55.80% |
| H9-scale-b：ACPD-v2 | 63.20% | 69.40% | 62.00% | 28.40% | 55.75% |
| H9 - H12 | -0.80 | +5.40 | -3.60 | -1.20 | -0.05 |

Pooled 差值为 `-0.05` 点，task-stratified paired bootstrap 95% CI 为
`[-2.50, +2.45]` 点。Object 的正差值没有转化为整体优势。25K 时两种方法整体
持平，说明 5K 的 ACPD-v2 优势在 25K 已不可检测。原始结果位于
`results/h9_vs_h12_25k_paired_analysis.json`。
