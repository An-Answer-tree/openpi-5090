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
