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
| ACPD-v2 延长训练是否有效 | H9-scale-b 从 5K `23.15%` 提高到 30K `58.00%`，但匹配 SFT 30K 为 `60.45%` | 延长训练提高了绝对成功率，尚未证明最终优于 SFT。 |
| 显式 residual 注入是否有效 | H9 为 `23.15%`，H13 loss-only 为 `20.60%`；差值 `+2.55` 点，配对 95% CI `[+0.40, +4.70]` | 达到预注册判据，支持显式注入。 |
| ACPD-v2 是否降低训练 flow MSE | 299 个对齐点与 SFT 的相关系数为 `0.9985`，全程平均 MSE 几乎相同 | 5K 成功率增益不是更低训练 MSE 的结果；内部表示改变是待验证的机制解释。 |
| 固定蒸馏权重是否适合全程训练 | 30K 时加权 contribution 与 ACL 的标量和约为 supervised 的 `3.29` 倍，而5K后的 contribution cosine 改善有限 | 固定权重可能使后期辅助目标过强；该证据支持测试非零下限退火，但不等同于梯度冲突证明。 |

## 方法判断

ACPD-v2 使用 teacher 真实 Q/K/V、完整 attention softmax、输出投影和 residual gate，
分别提取 agentview 与 wrist 对 action token 的贡献。Student 在 layer 10 预测两个
贡献向量，训练时使用 contribution loss，并在最终 hidden 上进行门控注入。

H7/H7.1 证明该目标可以从 backview 恢复；H9 与 H12 在 5K 检测到 ACPD-v2
早期优势，但 30K 时 ACPD-v2 为 `58.00%`，匹配 SFT 为 `60.45%`，差值
`-2.45` 点，95% CI `[-5.00, 0.00]`。H13 表明 5K 时显式注入带来 `2.55`
点 pooled 增益。训练 loss 只用于健康检查，最终判断使用 benchmark 成功率。

匹配 loss 分析显示，SFT 与 ACPD-v2 的监督 flow MSE 轨迹几乎重合，但 ACPD-v2
在 5K 已有明显成功率优势。Contribution cosine 在 5K 达到 `0.7157`，之后缓慢
升至 30K 的 `0.7999`；gate 在约 12K 达峰后下降。这支持继续检查中间 checkpoint，
但训练 loss、cosine 和 gate 都不能替代任务成功率验证。

下一步 H15 不关闭后期蒸馏，而是把 contribution 权重从 `0.2` 退火到 `0.02`、
ACL 权重从 `0.5` 退火到 `0.1`，并在 15K--30K 保持该非零下限。目标是让真实动作
监督成为后期主目标，同时持续更新 privileged contribution predictor，提高最终上限，
而不是只获得早期训练加速。

## 工程约束

- 最终论文对照除 teacher 外统一使用 4 GPU、physical global BS64。
- 蒸馏 checkpoint 使用零起始目录编号；需要保留每 5K 时设置 `keep_period=1`。
- 精选 checkpoint 只在完整写入后建立硬链接，不移动或删除原 checkpoint。
- 活动任务的 checkpoint、数据集、脚本和输出路径不得修改。
- teacher 特征依赖 noisy action 和 flow time，不能预计算而不改变方法。
- 早期 supervised loss 不能代替任务成功率筛选 ACPD 组件。

## 待回答问题

- BS64 rightview baseline 的 30K 表现如何？
- H9-scale-b 25K 是否优于 30K，或仍低于匹配 SFT？
- H15 的非零下限持续蒸馏能否在 30K 超过 SFT 的 `60.45%`？
- 公平 BS64 ACL-only 训练到 30K 后，ACPD-v2 的增益是否仍然成立？
