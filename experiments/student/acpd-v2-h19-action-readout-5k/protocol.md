# H19-AR5K：动作读出头的真实 5K 验证

## 目的

验证 H18 冻结模型中的小型动作读出头，在真实 ACPD-v2 训练和推理链路中是否能改善早期模型，而不是只改善一个离线特征回归指标。

## 训练

| 项目 | 设置 |
|---|---|
| 学生视角 | backview |
| 教师视角 | agentview + wrist |
| 数据集 | `libero_multiview_tuned_6view_lerobot` fixed_dataset |
| 学生初始化 | pi0.5 base LoRA 初始权重 |
| 教师 | agentview+wrist SFT 30K，`/opt/liutong/openpi_checkpoints/fixed_dataset/sft/agentview_wrist/29999` |
| ACPD 层 | layer 10 exact contribution |
| 读出头输入 | `[final_hidden, predicted_agentview_contribution, predicted_wrist_contribution]` 拼接 |
| 读出头位置 | action projection 后，对 7 个真实动作维度增加 correction；训练和推理相同 |
| 损失 | flow 1.0 + contribution 0.2 + ACL 0.5 |
| 优化 | LoRA，4×5090 FSDP，global BS64，梯度累积 1，seed 42 |
| 学习率 | warmup 1K，peak `2.5e-5`，cosine 到 5K，末端 `2.5e-6` |
| 训练步数 | 5,000 |
| checkpoint | step 4,999，`keep_period=5,000` |

动作读出头的输出层零初始化，因此训练开始时不会改变原 H9 ACPD-v2 的动作输出；它只能通过训练逐步学习有效 correction。

## 对照与评测

主对照是已有 H9 ACPD-v2 layer10 5K checkpoint；同时记录已有 SFT 5K 和 H13 loss-only 5K 作为背景。所有模型使用相同 fixed_dataset、相同四个 LIBERO benchmark suite、每 suite 500 回合、相同评测配置。

## 判定

- 若 H19-AR5K 在 pooled 2K success rate 高于 H9 5K，且不是仅由单一 suite 偶然造成，才进入 30K 正式验证。
- 若不高于 H9 5K，结论是当前读出头在 5K 早期没有实证收益；不把 H18 的离线 MSE 小幅下降解释为策略提升。
- 本实验只判断 5K 早期收益，不能单独证明 30K 上限；若有正向结果，再做同配置 30K。

## 证据路径

- 训练 checkpoint：`/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/h19_action_readout_5k`
- 日志：`/opt/liutong/openpi-5090-research/acpd-v2-h19-action-readout/slurm-log`
- 评测结果：`/opt/liutong/openpi-5090-research/acpd-v2-h19-action-readout/eval`
