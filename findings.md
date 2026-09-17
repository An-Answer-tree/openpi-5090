# Research Findings

## 研究问题

如何利用 teacher 的 `agentview+wrist` 特权视觉信息，提高只看 backview 的
pi0.5 student？

## 当前结论

| 问题 | 证据 | 结论 |
|---|---|---|
| 原始 ACPD 是否需要 layer 6+12 | H3，单层 6 的 5K loss 最低，层间差异小于 1% | 不需要双层；loss 不能代表成功率。 |
| Cue 或 ACL 是否加快早期收敛 | H4，2K supervised loss 差异均小于 1% | 没有可靠证据。 |
| Full ACPD 是否优于 Flow-only | H5，`6.50%` 对 `4.45%`，差值 `+2.05` 点，95% CI `[0.90, 3.25]` | 单 seed、5K 筛选为正。 |
| H5 增益来自哪里 | H8，ACL-only `6.35%`；Full 比 ACL-only 仅 `+0.15` 点，95% CI `[-1.15, 1.40]` | 当前证据支持 ACL，不支持旧 Cue 的额外收益。 |
| exact contribution 能否从 backview 恢复 | H7/H7.1，layer 10 gap `0.3992`、hard gap `0.3779`、EV `0.3556` | 可以；layer 10 是 6--12 层中的最佳层。 |

## 方法判断

旧 Cue 同时使用 teacher visual 和 action memory，容易依靠共享 noisy action 与
flow time，而不是真正转移特权视觉信息。ACPD-v2 改为固定目标：teacher 每个
视角通过真实 Q/K/V、完整 attention softmax、输出投影和 AdaRMS gate 对 action
token 产生的残差贡献。agentview 与 wrist 分开预测，注入时相加。

H7/H7.1 只证明目标可恢复，不证明能提高任务成功率。H9 正在检验最终 hidden
注入；H11 只改变部署位置，在 layer 10 attention 后、FFN 前注入，使后续网络
继续处理预测 contribution。H11 的主要判据是相对同为 4 GPU、BS64 的 H9-scale-b
在 step 4,999 提升至少 `1.5` 个 pooled 百分点，且配对 bootstrap 95% CI 下界
大于 0。训练 loss、cosine 和 gate 只用于健康检查。

## 工程约束

- 训练一个 student view；不要在一个任务中放四组 teacher-student。
- teacher 特征依赖 noisy action 和 flow time，不能预计算而不改变方法。
- 4 GPU、physical BS64、accumulation 1 已运行，峰值 `31,464 MiB/卡`，显存余量很小。
- H11 必须保持单次 student forward，不移动或复制 dataset、teacher、checkpoint。
- 早期 supervised loss 不能筛选 ACPD 组件；最终决策使用同状态 benchmark 成功率。

## 当前任务

| 实验 | Job | 状态 | 结论 |
|---|---:|---|---|
| H9 BS32 5K，最终 hidden 注入 | 129710 | 运行中 | 尚无结论。 |
| H9-scale-b BS64 30K，最终 hidden 注入 | 129728 | 运行中 | 尚无结论。 |
| H11 BS64 smoke，同层注入 | 129807 | 等待资源 | 尚无结论。 |
| H11 BS64 30K，同层注入 | 129808 | 等待 smoke 成功 | 尚无结论。 |

H10 的 loss 权重校准保留，但在 H9/H11 选定注入结构后再做，避免同时改变结构和
权重。
