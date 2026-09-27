# H19-AR：动作读出头的 30K 轨迹验证

## 目的

验证小型动作读出头从训练开始就参与 ACPD-v2 的训练和推理，是否能改善同一步数的 H9-Fixed 模型。5K、10K 用于观察轨迹，30K 用于判断最终效果。

协议修订（2026-09-27，训练尚未启动）：取消 10K 提前停训规则。早期成功率不能替代 30K 上限判断；同时将学习率衰减终点更正为实际提交脚本中的 30K。

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
| 学习率 | warmup 1K，peak `2.5e-5`，cosine 到 30K，末端 `2.5e-6` |
| 训练步数 | 30,000 |
| checkpoint | steps 4,999、9,999、14,999、19,999、24,999、29,999，`keep_period=1` |

动作读出头的输出层零初始化，因此训练开始时不会改变原 H9 ACPD-v2 的动作输出；它只能通过训练逐步学习有效 correction。

## 对照与评测

主对照是已有 H9-Fixed layer10 在相同步数的 checkpoint；同时记录 SFT 和 H13 作为背景。所有模型使用相同 fixed_dataset、相同四个 LIBERO benchmark suite、相同评测配置。

- 5K：四个 suite 各 250 回合，观察早期表现。
- 10K：四个 suite 各 500 回合，观察中期表现。
- 无论 5K、10K 结果如何，训练至 30K；30K 使用四个 suite 各 500 回合评测。
- 各步评测使用固定 episode seeds；最终结论以同配置配对 bootstrap 为准。

## 判定

- 5K、10K 只描述收敛轨迹，不作为 30K 训练或评测的门槛。
- 30K 以同一步 H9-Fixed 为主对照，以匹配 SFT 为次对照；报告四套和 pooled 成功率及配对区间。
- 若 30K 优于 H9，还需读出头容量匹配的对照，才能将增益归因于特权贡献，而非新增参数本身。

## 证据路径

- 训练 checkpoint：`/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/action_readout_30k`
- 日志：`/opt/liutong/openpi-5090-research/acpd-v2-h19-action-readout/slurm-log`
- 评测结果：`/opt/liutong/openpi-5090-research/acpd-v2-h19-action-readout/eval`
