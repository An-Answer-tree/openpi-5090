# H15a-fast：单卡梯度冲突探索

## 目的

正式 H15a 等待 2 卡资源时，先用单卡小实验判断 5K 到 30K 的辅助梯度关系是否出现
明显变化。该实验是探索性诊断，不替代正式 BS32 结果。

## 设置

- Checkpoint：H9-scale-b 的完整 `4999` 与 `29999`。
- 数据：两组使用 seed 42、相同的前 100 个 shuffled samples、相同噪声和 flow time。
- 资源：1×RTX 5090，BS1，无参数更新。
- 权重：flow `1.0`、contribution `0.2`、ACL `0.5`，与 H9 一致。
- 测量：共享 LoRA 参数上的 cosine、冲突率和相对范数。

## 判断

若 30K 的组合辅助梯度冲突率比 5K 高至少 10 个百分点，且组合 cosine 中位数小于
0，则优先准备 conflict-aware ACPD 小实验。否则不把梯度冲突作为主解释，下一步优先
测试 agentview/wrist 分离、按 action token 条件化的动态 gate。

由于 BS1 的梯度方差和正式 BS32 不同，本实验只能决定下一步探索方向。最终机制结论
仍以正式 H15a 为准。
