# H15a 梯度冲突分析

## 正式 BS32 诊断

正式实验在 H9 5K 与 30K checkpoint 上使用相同的 200 个 BS32 batches。下表比较
共享 LoRA 参数中的加权辅助梯度与 flow 梯度。

| 指标 | 5K | 30K | 30K - 5K | 95% CI |
|---|---:|---:|---:|---:|
| 组合辅助梯度冲突率 | 0.00% | 0.00% | 0.00 点 | [0.00, 0.00] |
| 组合辅助梯度 cosine 中位数 | 0.6279 | 0.5920 | -0.0360 | [-0.0560, -0.0116] |
| 组合辅助梯度范数比中位数 | 1.6259 | 1.5011 | -0.1248 | - |
| contribution 冲突率 | 2.00% | 2.50% | +0.50 点 | [-2.50, +3.50] |
| contribution cosine 中位数 | 0.1699 | 0.0957 | -0.0743 | [-0.1002, -0.0583] |
| ACL 冲突率 | 0.00% | 0.00% | 0.00 点 | [0.00, 0.00] |
| ACL cosine 中位数 | 0.9208 | 0.9012 | -0.0196 | [-0.0288, -0.0129] |

正式结果不满足预注册判据。30K 的组合辅助梯度仍与 flow 梯度同向，且相对范数没有
增加。因此，当前证据不支持将 ACPD-v2 后期上限不足归因于辅助梯度冲突，也不支持
优先运行 conflict-aware 梯度投影。下一项实验按预注册分支测试 H16 动态视角 gate。

## H15a-fast

本次探索性诊断在 H9 5K 与 30K checkpoint 上使用相同的 100 个 BS1 samples。下表
比较共享 LoRA 参数中的加权辅助梯度与 flow 梯度。

| 指标 | 5K | 30K | 30K - 5K | 95% CI |
|---|---:|---:|---:|---:|
| 组合辅助梯度冲突率 | 6.00% | 3.00% | -3.00 点 | [-9.00, +3.00] |
| 组合辅助梯度 cosine 中位数 | 0.4228 | 0.3795 | -0.0433 | [-0.1012, -0.0085] |
| 组合辅助梯度范数比中位数 | 2.6919 | 2.8109 | +0.1190 | - |
| contribution 冲突率 | 24.00% | 30.00% | +6.00 点 | [-5.00, +17.00] |
| contribution cosine 中位数 | 0.0572 | 0.0332 | -0.0240 | [-0.0672, +0.0046] |
| ACL 冲突率 | 0.00% | 1.00% | +1.00 点 | [0.00, +3.00] |
| ACL cosine 中位数 | 0.9414 | 0.9402 | -0.0012 | [-0.0194, +0.0163] |

## 探索性结论

结果不满足预注册的冲突判据：30K 的组合冲突率没有比 5K 高至少 10 个百分点，且
30K 组合 cosine 中位数仍为正。因此，当前证据不支持将 ACPD-v2 后期上限不足主要
归因于组合辅助梯度冲突，暂不优先测试只投影负梯度的 conflict-aware ACPD。

Contribution 单项接近正交且范数较大，但 ACL 梯度与 flow 高度同向，使组合梯度总体
保持同向。下一项小实验应针对当前结构缺陷：H9 使用一个全局标量同时缩放 agentview
和 wrist contribution，不能按样本或 action token 选择视角。

本结果使用 BS1，只用于选择探索方向；正式结论以上述 BS32 结果为准。

## 证据

- 5K 原始结果：`/opt/liutong/openpi-5090-research/acpd-v2-gradient-conflict/h9-5k-bs1-n100/`
- 30K 原始结果：`/opt/liutong/openpi-5090-research/acpd-v2-gradient-conflict/h9-30k-bs1-n100/`
- 成对分析：`results/h15a_fast_paired_analysis.json`
- 正式 5K 原始结果：`/opt/liutong/openpi-5090-research/acpd-v2-gradient-conflict/h9-5k/`
- 正式 30K 原始结果：`/opt/liutong/openpi-5090-research/acpd-v2-gradient-conflict/h9-30k/`
- 正式成对分析：`results/h15a_formal_paired_analysis.json`
