# H18 结果分析

## 实际运行

Job `136130` 使用 H9-Fixed BS64 30K checkpoint，冻结原模型，只训练
`hidden-only`、`hidden-layer10`、`hidden-contribution` 三个动作修正 head。
训练500步、BS8；随后在固定的194个 held-out episode上评估128个batch。
该 held-out 划分只约束新 head，H9主干此前已经见过这些episode。

证据：
`/opt/liutong/openpi-5090-research/acpd-v2-h18-action-readout/results/136130/summary.json`。

## 结果

| 模型 | episode mean flow MSE |
|---|---:|
| 冻结 H9 | 0.087586 |
| hidden-only | 0.086893 |
| hidden-layer10 | 0.086849 |
| hidden-contribution | 0.086874 |

`hidden-contribution - frozen H9` 为 `-0.000712`，相对下降 `0.813%`，配对
95% CI 为 `[-1.804%, +0.031%]`（按冻结 H9 的 episode MSE 归一化），上界跨过0。

相对两个新增 head 对照：

| 比较 | contribution 相对 MSE变化 | 配对95% CI |
|---|---:|---:|
| hidden-only | `-0.000019`（下降0.022%） | `[-0.000706, 0.000589]` |
| hidden-layer10 | `+0.000025`（变差0.029%） | `[-0.000522, 0.000536]` |

预设要求是相对冻结 H9 和两个对照都下降至少1%，且区间上界小于0。结果未通过，
不进入基于该 head 的正式仿真或35K续训。

## 结论

小型动作修正头本身可以略微降低 flow MSE，但 contribution 输入没有显示出相对
普通 layer10 输入的独立收益。该实验不支持“把预测 contribution 接到动作修正头就能
提高后期动作”的假设，也不支持立即增加 prediction head 深度。

这个结果不等于 contribution 不能被预测：H7/H9 的 contribution cosine 仍说明
teacher contribution 可恢复。当前更可能的问题是预测目标与动作使用之间的接口，或
H9 30K 已接近该离线动作损失的可达水平。由于没有闭环仿真，这个结果不能直接写成
成功率下降或最终方法失败。
