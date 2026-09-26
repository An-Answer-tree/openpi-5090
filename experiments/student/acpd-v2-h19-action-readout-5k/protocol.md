# H19-AR：动作读出头的 30K 轨迹验证

## 目的

验证小型动作读出头从训练开始就参与 ACPD-v2 的训练和推理，是否能改善同一步数的 H9-Fixed 模型。5K、10K 是中间决策点，30K 是最终确认点。

## 训练

| 项目 | 设置 |
|---|---|
| 学生视角 | backview |
| 教师视角 | agentview + wrist |
| 数据集 | `libero_multiview_tuned_6view_lerobot` fixed_dataset |
| 学生初始化 | pi0.5 base LoRA 初始权重，从 0 步训练 |
| 教师 | agentview+wrist SFT 30K，`/opt/liutong/openpi_checkpoints/fixed_dataset/sft/agentview_wrist/29999` |
| ACPD 层 | layer 10 exact contribution |
| 读出头输入 | `[final_hidden, predicted_agentview_contribution, predicted_wrist_contribution]` 拼接 |
| 读出头位置 | action projection 后增加 correction；训练和推理相同 |
| 损失 | flow 1.0 + contribution 0.2 + ACL 0.5 |
| 优化 | LoRA，4×5090 FSDP，global BS64，梯度累积 1，seed 42 |
| 学习率 | warmup 1K，peak `2.5e-5`，cosine 到 5K，末端 `2.5e-6` |
| 训练步数 | 30,000 |
| checkpoint | steps 4,999、9,999、14,999、19,999、24,999、29,999，`keep_period=1` |

动作读出头的输出层零初始化，因此训练开始时不会改变原 H9 ACPD-v2 的动作输出；它只能通过训练逐步学习有效 correction。

## 对照与中止规则

主对照是已有 H9-Fixed layer10 在相同步数的 checkpoint；同时记录 SFT 和 H13 作为背景。所有模型使用相同 fixed_dataset、相同四个 LIBERO benchmark suite、相同评测配置。

- 5K：四个 suite 各 250 回合，作为早期筛选，不因单一 suite 波动直接停止。
- 10K：四个 suite 各 500 回合；若 pooled success 不高于 H9 10K，且配对 95% CI 不支持正向差异，则停止 H19，不继续 30K。
- 只有 10K 通过筛选，才保留并评测 30K；30K 使用四个 suite 各 500 回合。
- 5K、10K 的评测使用固定 episode seeds；最终结论以同配置配对 bootstrap 为准。

## 判定

- 若 H19-AR 在 pooled success 高于同一步 H9，且至少两个 suite 同方向，记录为正向筛选结果。
- 若 10K 未通过预注册筛选，结论是当前读出头没有足够的早期任务收益；不把 H18 的离线 MSE 小幅下降解释为策略提升。
- 30K 的成功率只能在 10K 通过后解释为最终确认结果。

## 证据路径

- 训练 checkpoint：`/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/action_readout_30k`
- 日志：`/opt/liutong/openpi-5090-research/acpd-v2-h19-action-readout/slurm-log`
- 评测结果：`/opt/liutong/openpi-5090-research/acpd-v2-h19-action-readout/eval`
