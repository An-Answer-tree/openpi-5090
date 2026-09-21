# H15a：ACPD-v2 梯度冲突诊断

## 问题

ACPD-v2 在 5K 相对匹配 SFT 高 `9.30` 点，但在 30K 低 `2.45` 点。H15a 检验：
contribution 与 ACL 的梯度是否在训练后期干扰动作监督梯度。

## 假设

若固定辅助目标导致后期性能回落，则 30K checkpoint 的辅助梯度应比 5K 更频繁地
与 flow 梯度反向，或相对范数明显过大。标量 loss 比例不能回答该问题，必须直接测量
共享 LoRA 参数上的梯度方向和范数。

## 设置

- Checkpoint：H9-scale-b 的完整 `4999` 与 `29999`。
- 数据：两组使用相同 seed、相同前 200 个 shuffled batch、相同噪声和 flow time。
- 资源：2×RTX 5090，FSDP，physical global BS32，不更新参数。
- 权重：flow `1.0`、contribution `0.2`、ACL `0.5`，与 H9 一致。
- 测量参数：共享 LoRA 参数；predictor 和 gate 不进入冲突统计。
- 每个 batch 分别计算 `g_flow`、`g_contribution`、`g_acl`，并组合
  `g_priv = 0.2 g_contribution + 0.5 g_acl`。
- 输出：逐 batch JSONL 和汇总 JSON；不保存新 checkpoint。

## 指标与判断

分别报告：

- `cos(g_flow, g_contribution)`；
- `cos(g_flow, g_acl)`；
- `cos(g_flow, g_priv)`；
- 三组 cosine 小于 0 的 batch 比例；
- 三组辅助梯度相对 flow 梯度的范数比例。

若 30K 的组合梯度冲突率比 5K 高至少 10 个百分点、组合 cosine 中位数小于 0，
或组合梯度范数长期大于 flow 梯度且伴随冲突，则支持后续测试 conflict-aware ACPD：
只投影掉与 flow 冲突的辅助梯度分量，并限制其范数，不关闭 teacher 信号。

若没有观察到上述现象，则不运行 conflict-aware 训练，下一步改测 agentview/wrist
分离且按 action token 条件化的动态 gate。该实验是机制诊断，不用训练 loss 推断任务成功率。
