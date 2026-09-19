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
| exact contribution 能否提高任务成功率 | H9，`11.35%` 对 H8 的 `6.35%`，差值 `+5.00` 点，95% CI `[+3.60, +6.45]` | 单 seed、5K 筛选支持 ACPD-v2。 |
| H9 的训练图是否稳定且目标可学 | H9 step 0--4,900：supervised loss `0.0892→0.0311`，exact contribution cosine `0.0057→0.6980`，单卡峰值 `17,406 MiB` | 训练实现稳定，layer-10 exact contribution 能被 student 学习。 |
| H9 是否已证明直接注入机制 | gate 末步仅 `0.0048`，融合预测 stop-gradient；H9 只比较最终任务成功率 | 尚未完全证明；增益可能包含辅助损失的表征约束，需要 H13 loss-only 对照。 |
| 同层注入是否优于最终 hidden 注入 | H11 `21.20%` 对 H9-scale-b `23.15%`；差值 `-1.95` 点，95% CI `[-4.10, +0.20]` | 不支持 H11；保留 H9 的最终 hidden 注入作为当前结构。 |
| ACPD-v2 是否优于匹配 BS64 SFT | H9-scale-b `23.15%` 对 H12 `13.85%`；差值 `+9.30` 点，95% CI `[+7.30, +11.35]` | 5K 单 seed 下支持 ACPD-v2 整体方案，不是单纯 BS64 效应。 |

## 方法判断

旧 Cue 同时使用 teacher visual 和 action memory，容易依靠共享 noisy action 与
flow time，而不是真正转移特权视觉信息。ACPD-v2 改为固定目标：teacher 每个
视角通过真实 Q/K/V、完整 attention softmax、输出投影和 AdaRMS gate 对 action
token 产生的残差贡献。agentview 与 wrist 分开预测，注入时相加。

H7/H7.1 证明目标可恢复，H9 进一步证明最终 hidden 注入在 5K 筛选中优于
ACL-only。H9 使用 layer 10 block 输出预测，并在最终 hidden 注入。H11 将 query 和注入共同前移：使用 layer 10
attention 输出预测，在该层 FFN 前注入，使后续网络继续处理 contribution。H11
的主要判据是相对同为 4 GPU、BS64 的 H9-scale-b
在 step 4,999 提升至少 `1.5` 个 pooled 百分点，且配对 bootstrap 95% CI 下界
大于 0。训练 loss、cosine 和 gate 只用于健康检查。

H9 训练过程本身没有数值或资源异常：supervised/student task loss 和 exact
contribution loss 均下降，预测 cosine 上升，LoRA 与 predictor 梯度持续非零。由于
最终 gate 很小，H9 的正向成功率只能说明 ACPD-v2 整体训练目标和部署组合有效，不能
单独证明推理时残差注入是唯一原因。

## 工程约束

- 训练一个 student view；不要在一个任务中放四组 teacher-student。
- teacher 特征依赖 noisy action 和 flow time，不能预计算而不改变方法。
- 4 GPU、physical BS64、accumulation 1 已运行，峰值 `31,464 MiB/卡`，显存余量很小。
- 蒸馏保存目录使用零起始 step；每 5K 保存且需要全部保留时必须设置
  `keep_period=1`。
- H11 必须保持单次 student forward，不移动或复制 dataset、teacher、checkpoint。
- 早期 supervised loss 不能筛选 ACPD 组件；最终决策使用同状态 benchmark 成功率。

## 当前任务

| 实验 | Job | 状态 | 结论 |
|---|---:|---|---|
| H9 BS32 5K，最终 hidden 注入 | 129710/129711 | 完成 | `11.35%`；相对 H8 `+5.00` 点，95% CI `[+3.60, +6.45]`。 |
| H9-scale-b BS64 30K，最终 hidden 注入 | 129728/130703 | 原任务约 25.4K；25K 已保护；完成后恢复 25K 与 30K | 5K 历史验证为 `23.15%`；30K 尚无结论。 |
| H9-scale-b 20K 轨迹恢复 | 130704 | 排队；保留 5K/10K/15K/20K | 与原任务训练设置相同，仅恢复阶段 checkpoint；尚无新结论。 |
| H11 BS64，同层注入 | 129808/130490 | 完成 | `21.20%`；相对 H9-scale-b `-1.95` 点，95% CI `[-4.10, +0.20]`，不支持前移注入。 |
| H12 backview SFT BS64 5K | 130285/130491 | 完成 | `13.85%`；H9-scale-b 相对提升 `9.30` 点，95% CI `[+7.30, +11.35]`。 |
| H13 loss-only BS64 5K | 130599/130600 | 训练中，验证依赖训练 | 保留 contribution loss 和 ACL，只关闭训练与推理注入；尚无结论。 |
| H14 top/left/right SFT BS64 30K | 130669/130670/130671 | 排队 | 匹配多视角 baseline；尚无结论。 |

H10 的 loss 权重校准保留，但在 H9/H11 选定注入结构后再做，避免同时改变结构和
权重。
