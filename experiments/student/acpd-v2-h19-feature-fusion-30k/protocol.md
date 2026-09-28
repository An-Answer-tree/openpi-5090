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
提交，待 H19 结果出来后再决定是否运行；默认值为使用 contribution，不改变当前 H19。

| 项目 | 设置 |
|---|---|
| 学生/教师视角 | backview / agentview+wrist |
| 数据 | fixed_dataset `libero_multiview_tuned_6view_lerobot` |
| 初始化 | pi0.5 base，从 0 步训练 |
| 损失权重 | flow 1.0、contribution 0.2、ACL 0.5 |
| 训练 | 4×5090，FSDP LoRA，global BS64，累积 1，seed 42 |
| 学习率 | warmup 1K，peak 2.5e-5，cosine 到 30K，末端 2.5e-6 |
| Checkpoint | 每 5K 保存，`keep_period=1` |
| 执行 | debug smoke `136339` 完成两步；正式训练 `136345` 排队 |

## 验证与判定

5K、10K 只观察训练轨迹，不作提前停训依据。30K 用四套 LIBERO benchmark
各 500 回合，固定与 H9-Fixed、SFT 相同的 episode seeds，比较每套及 pooled
成功率，并计算配对区间。若优于 H9，还需同容量、无预测贡献输入的对照，
才能将收益归因于特权贡献，而非新增网络容量。

旧 H19 动作读出方案在正式训练前放弃：训练任务 136147 排队中取消，
checkpoint 轮询任务 136164 取消；无 checkpoint 和 benchmark 结果。

| 证据 | 路径 |
|---|---|
| Checkpoint | `/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/feature_fusion_30k` |
| 训练日志 | `/opt/liutong/openpi-5090-research/acpd-v2-h19-feature-fusion/slurm-log` |
| 验证结果 | `/opt/liutong/openpi-5090-research/acpd-v2-h19-feature-fusion/eval` |
