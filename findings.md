# Research Findings

## 研究问题

如何利用 teacher 的 `agentview+wrist` 特权视觉信息，提高只看弱视角的 pi0.5 student？

## 当前认识

| 问题 | 证据 | 结论 |
|---|---|---|
| 原始 Cue 是否有效 | Full ACPD 仅比 ACL-only 高 `0.15` 点，95% CI `[-1.15, 1.40]` | 没有检测到 Cue 的额外收益。 |
| 特权视觉贡献能否恢复 | layer 10 overall gap `0.3992`，比次优层高 `0.0485` | backview 表征可以预测 teacher 的真实 attention contribution。 |
| ACPD-v2 是否有正向任务信号 | BS32 5K 相对 ACL-only 提升 `5.00` 点，95% CI `[3.60, 6.45]` | 单 seed、5K 筛选支持 ACPD-v2。 |
| ACPD-v2 是否优于公平 SFT | BS64 5K 为 `23.15%` 对 `13.85%`；30K 为 `58.00%` 对 `60.45%` | ACPD-v2 有早期优势，但没有保持到 30K。 |
| ACPD-v2 在 10K 是否仍有优势 | ACPD-v2 `37.45%`，SFT `23.25%`；差值 `+14.20` 点，95% CI `[+11.90, +16.60]` | 10K 仍有明确早期优势。 |
| ACPD-v2 在 25K 是否仍有优势 | ACPD-v2 `55.75%`，SFT `55.80%`；差值 `-0.05` 点，95% CI `[-2.50, +2.45]` | 25K 已无可检测优势。 |
| 匹配 SFT 在 20K 的水平 | BS64 backview SFT 20K pooled success 为 `47.20%` | H9 20K 完成后可直接判断早期优势在 20K 是否仍存在。 |
| ACPD-v2 延长训练是否有效 | H9-scale-b 从 5K `23.15%` 提高到 30K `58.00%`，但匹配 SFT 30K 为 `60.45%` | 延长训练提高了绝对成功率，尚未证明最终优于 SFT。 |
| H9 是否在 25K 早于 30K 达峰 | 25K 为 `55.75%`，30K 为 `58.00%`；差值 `-2.25` 点，95% CI `[-4.70, +0.20]` | 不支持 25K 已过峰值。 |
| H9 在 30K 后是否继续提高 | 35K 为 `61.05%`，相对 30K 提升 `3.05` 点，95% CI `[+0.60, +5.50]` | 检测到 30K--35K 的继续提高；40K--60K 尚待验证。 |
| H9 35K 是否优于 SFT 30K | `61.05%` 对 `60.45%`；差值 `+0.60` 点，95% CI `[-1.90, +3.05]` | 35K 是当前最高 backview 观测值，但尚不能证明优于 SFT。 |
| 显式 residual 注入是否有效 | H9 为 `23.15%`，H13 loss-only 为 `20.60%`；差值 `+2.55` 点，配对 95% CI `[+0.40, +4.70]` | 达到预注册判据，支持显式注入。 |
| ACPD-v2 是否降低训练 flow MSE | 299 个对齐点与 SFT 的相关系数为 `0.9985`，全程平均 MSE 几乎相同 | 5K 成功率增益不是更低训练 MSE 的结果；内部表示改变是待验证的机制解释。 |
| 辅助目标在后期是否仍有数值权重 | 首末窗口中 contribution/ACL 占总目标约 `61.39/16.69%` 与 `60.29/16.52%` | 辅助目标未自然消失；loss 占比不能判断其对成功率的因果作用。 |
| 后期辅助梯度是否更冲突 | 正式 BS32 中组合冲突率在 5K/30K 均为 `0%`，cosine 中位数为 `0.6279/0.5920` | 不支持后期辅助梯度冲突解释，不优先运行梯度投影。 |
| 单视角 baseline 是否受视角影响 | left/right/top/backview 30K pooled 分别为 `78.65/77.00/71.55/60.45%` | 视角差异大；ACPD 必须使用相同 student 视角的 SFT 对照。 |

## 方法判断

ACPD-v2 使用 teacher 真实 Q/K/V、完整 attention softmax、输出投影和 residual gate，
分别提取 agentview 与 wrist 对 action token 的贡献。Student 在 layer 10 预测两个
贡献向量，训练时使用 contribution loss，并在最终 hidden 上进行门控注入。

H7/H7.1 证明该目标可以从 backview 恢复；H9 与 H12 在 5K、10K 检测到 ACPD-v2
早期优势，但 30K 时 ACPD-v2 为 `58.00%`，匹配 SFT 为 `60.45%`，差值
`-2.45` 点，95% CI `[-5.00, 0.00]`。H13 表明 5K 时显式注入带来 `2.55`
点 pooled 增益。训练 loss 只用于健康检查，最终判断使用 benchmark 成功率。

匹配 loss 分析显示，SFT 与 ACPD-v2 的监督 flow MSE 轨迹几乎重合，但 ACPD-v2
在 5K 已有明显成功率优势。Contribution cosine 在 5K 达到 `0.7157`，之后缓慢
升至 30K 的 `0.7999`；gate 在约 12K 达峰后下降。25K 成功率为 `55.75%`，
低于 30K 的 `58.00%`。匹配 SFT 在 25K 为 `55.80%`，与 ACPD-v2 的
`55.75%` 无差异，因此早期增益在 25K 已消失；训练 loss、cosine 和 gate 都不能
替代任务成功率验证。

H15a 正式 BS32 诊断没有观察到 30K 组合冲突率上升：5K 与 30K 均为 `0%`，组合
cosine 中位数仍为正。该结果否定了优先测试 conflict-aware 梯度投影的依据。H9 35K
相对自身 30K 有显著提高，但尚未显著优于 SFT 30K；当前继续验证 15K、20K 与
40K--60K，分别定位早期优势消失区间和后期最佳点。

## 工程约束

- 最终论文对照除 teacher 外统一使用 4 GPU、physical global BS64。
- 蒸馏 checkpoint 使用零起始目录编号；需要保留每 5K 时设置 `keep_period=1`。
- 精选 checkpoint 只在完整写入后建立硬链接，不移动或删除原 checkpoint。
- 活动任务的 checkpoint、数据集、脚本和输出路径不得修改。
- teacher 特征依赖 noisy action 和 flow time，不能预计算而不改变方法。
- 早期 supervised loss 不能代替任务成功率筛选 ACPD 组件。

## 待回答问题

- ACPD-v2 相对匹配 SFT 的早期优势在 20K 是否仍然存在？
- 公平 BS64 ACL-only 训练到 30K 后，ACPD-v2 的增益是否仍然成立？
