# Research Findings

## 研究问题

如何利用 teacher 的 `agentview+wrist` 特权视觉信息，提高只看弱视角的 pi0.5 student？

## 当前认识

| 问题 | 证据 | 结论 |
|---|---|---|
| 原始 Cue 是否有效 | Full ACPD 仅比 ACL-only 高 `0.15` 点，95% CI `[-1.15, 1.40]` | 没有检测到 Cue 的额外收益。 |
| 特权视觉贡献能否恢复 | layer 10 overall gap `0.3992`，比次优层高 `0.0485` | backview 表征可以预测 teacher 的真实 attention contribution。 |
| ACPD-v2 是否有正向任务信号 | BS32 5K 相对 ACL-only 提升 `5.00` 点，95% CI `[3.60, 6.45]` | 单 seed、5K 筛选支持 ACPD-v2。 |
| ACPD-v2 是否优于公平 SFT | BS64 5K 为 `23.15%` 对 `13.85%`，差值 `+9.30` 点，95% CI `[7.30, 11.35]` | 当前公平对照支持 ACPD-v2 整体方案。 |
| 显式 residual 注入是否必要 | H13 loss-only 已完成 5K，验证排队 | 尚无结论。 |

## 方法判断

ACPD-v2 使用 teacher 真实 Q/K/V、完整 attention softmax、输出投影和 residual gate，
分别提取 agentview 与 wrist 对 action token 的贡献。Student 在 layer 10 预测两个
贡献向量，训练时使用 contribution loss，并在最终 hidden 上进行门控注入。

H7/H7.1 证明该目标可以从 backview 恢复；H9 与 H12 证明完整 ACPD-v2 在相同
BS64、step 和初始化下优于 SFT。H13 用完全匹配的 loss-only 训练判断任务增益来自
辅助监督还是显式注入。训练 loss 只用于健康检查，最终判断使用 benchmark 成功率。

## 工程约束

- 最终论文对照除 teacher 外统一使用 4 GPU、physical global BS64。
- 蒸馏 checkpoint 使用零起始目录编号；需要保留每 5K 时设置 `keep_period=1`。
- 精选 checkpoint 只在完整写入后建立硬链接，不移动或删除原 checkpoint。
- 活动任务的 checkpoint、数据集、脚本和输出路径不得修改。
- teacher 特征依赖 noisy action 和 flow time，不能预计算而不改变方法。
- 早期 supervised loss 不能代替任务成功率筛选 ACPD 组件。

## 待回答问题

- H13 loss-only 与 H9-scale-b 在 5K benchmark 上是否存在显著差异？
- BS64 backview、topview、leftview、rightview baseline 的 30K 表现如何？
- H9-scale-b 在 10K-30K 的最佳停止点是什么？
- 公平 BS64 ACL-only 训练到 30K 后，ACPD-v2 的增益是否仍然成立？
