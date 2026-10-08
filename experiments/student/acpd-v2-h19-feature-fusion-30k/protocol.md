# H19：预测视觉贡献的特征融合

## 目的

检验学生在动作输出前融合预测的 agentview 与 wrist 贡献，能否在 30K
超过同配置 H9-Fixed 与 backview SFT。

## 方法

学生从第 10 层预测两路 1024 维教师 attention contribution。将两路预测与
最后一层 action hidden 拼接，经过宽度 128 的网络生成 1024 维特征修正，
加到原 hidden 后交给原有动作输出层。新网络末层小幅随机初始化。
不使用旧 scalar gate，也不增加 32 维动作修正头。预测器继续接受贡献蒸馏损失；
flow loss 更新融合网络，但不经该路径直接更新预测器。

融合网络的新增参数为 `fc1: 3×1024×128+128=393,344` 和
`fc2: 128×1024+1024=132,096`，合计 `525,440`；不包含已有的 contribution
predictor。若 H19 的成功率提高，后续必须使用同样参数量但不输入 predicted
contribution 的对照，才能将收益归因于特权信息，而不是模型容量增加。

代码已加入容量匹配对照开关 `--no-feature-fusion-use-contribution`：保持同一个
`3×1024→128→1024` 融合网络和参数量，但将两路 contribution 输入置零。该对照尚未
提交，待 H19 结果出来后再决定是否运行；对应评测配置为
`pi05_libero_backview_acpd_v2_layer10_feature_fusion_capacity_control_lora`。默认值为
使用 contribution，不改变当前 H19。

| 项目 | 设置 |
|---|---|
| 学生/教师视角 | backview / agentview+wrist |
| 数据 | fixed_dataset `libero_multiview_tuned_6view_lerobot` |
| 初始化 | pi0.5 base，从 0 步训练 |
| 损失权重 | flow 1.0、contribution 0.2、ACL 0.5 |
| 训练 | 4×5090，FSDP LoRA，global BS64，累积 1，seed 42 |
| 学习率 | warmup 1K，peak 2.5e-5，cosine 到 30K，末端 2.5e-6 |
| Checkpoint | 每 5K 保存，`keep_period=1` |
| 执行 | debug smoke `136339` 完成两步；正式训练 `136345` 已完成 30K，保存六个 checkpoint |

## 验证与判定

5K、10K 原计划只观察训练轨迹；为检查早期收益是否出现，现追加两个探索性全量
评测。25K 同样追加为探索性全量评测，用于观察 H19 后期相对 H9-Fixed 的变化。
每套 LIBERO benchmark 共 500 回合，即每个任务 50 回合；固定与 H9-Fixed、SFT
相同的 episode seeds，不作为提前停训依据，也不替代 30K 主评测。30K 仍是主评测，
使用相同回合数，比较每套及 pooled 成功率，并计算配对区间。
若优于 H9，还需同容量、无预测贡献输入的对照，才能将收益归因于特权贡献，而非
新增网络容量。

25K 四套评测任务为 `139046`，汇总任务为 `139049`；结果目录为
`/opt/liutong/openpi-5090-evals/acpd-v2-h19-feature-fusion/24999-rerun-50-per-task`。
四套结果和汇总已完成，25K pooled success 为 `55.35%`。30K 评测使用数组
`139418`，汇总任务为 `139419`，四套各500回合完成，pooled success 为 `60.10%`。
相对 H9-Fixed 30K 的 `58.00%` 高 `2.10` 点，但相对匹配 SFT 30K 的 `60.45%`
低 `0.35` 点；因此尚未证明提高最终 pooled 上限。

旧 H19 动作读出方案在正式训练前放弃：训练任务 136147 排队中取消，
checkpoint 轮询任务 136164 取消；无 checkpoint 和 benchmark 结果。

## 35K 续训

为比较 H19 在 30K 后的短期轨迹，从完整的 H19 30K `29999` checkpoint
恢复模型与优化器，目标步数为 35000。使用2×5090、全局 micro BS32、
梯度累计2，有效BS64；模型结构、数据、损失权重、学习率设置与seed 42沿用原训练。
数据加载器按新micro batch配置从`30000×2=60000`处推导接续位置；
micro batch及每轮丢弃的尾部样本变化，不能声称与原4卡训练逐样本完全一致。
输出写入独立的 `feature_fusion_35k` 根目录，避免覆盖
30K 结果。续训使用 `save_interval=1000`、`keep_period=1`，保存并保留
`30999/31999/32999/33999/34999`，对应 31K–35K；原 0–30K 训练仍是每5K保存。
全量验证另行提交，在验证完成前不记录成功率结论。

2026-10-08取消尚未启动的4卡任务`141714`（此前替换`139791`），
提交2卡任务`141769`，无reservation。16:45按用户要求原地清除gpu05绑定，
已核对`ReqNodeList=(null)`；可在batch分区任意满足资源要求的节点运行，
保留原`gpu03`排除设置。任务仍为`PENDING (Priority)`，任务号与排队时间保留。
训练脚本为
`scripts/train_slurm/pi05_libero_backview_acpd_v2_feature_fusion_fsdp2_bs64_resume_35k.sbatch`；
任务使用 2×5090、16 CPU、48G 内存，时限 3 天。源 checkpoint 为完整的 H19 30K
`29999`，目标目录为
`/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/feature_fusion_35k`。
任务开始前不复制或移动源 checkpoint；35K checkpoint 和全量验证均尚未产生。

| 证据 | 路径 |
|---|---|
| Checkpoint | `/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/feature_fusion_30k/pi05_libero_backview_acpd_v2_feature_fusion/pi05_libero_backview_acpd_v2_feature_fusion_lora_fsdp4_bs64_30k/{4999,9999,14999,19999,24999,29999}` |
| 35K 续训目标 | `/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/feature_fusion_35k` |
| 训练日志 | `/opt/liutong/openpi-5090-research/acpd-v2-h19-feature-fusion/slurm-log` |
| 验证结果 | `/opt/liutong/openpi-5090-evals/acpd-v2-h19-feature-fusion/{4999,9999,24999-rerun-50-per-task,29999-rerun-50-per-task}` |
