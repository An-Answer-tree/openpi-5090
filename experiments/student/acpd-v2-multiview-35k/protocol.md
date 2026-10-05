# ACPD-v2 多视角 Student 35K 训练协议

## 目的

在 H9-Fixed 的相同蒸馏设置下，分别训练 leftview、rightview 和 topview
student，获得与 backview H9 可直接比较的多视角结果。

## 固定设置

| 项目 | 设置 |
|---|---|
| Student | pi0.5 LoRA，单一弱视角（left/right/top） |
| Teacher | agentview + wrist，固定 `/opt/liutong/openpi_checkpoints/fixed_dataset/sft/agentview_wrist/29999/params` |
| Privileged target | layer 10 exact attention contribution |
| Loss | flow 1.0，contribution 0.2，ACL 0.5 |
| 注入 | 保留 H9 exact contribution fusion 与 residual injection |
| 数据 | `libero_multiview_tuned_6view_lerobot`，fixed_dataset 资产 |
| 训练 | 4×5090 FSDP，physical global BS64，gradient accumulation 1，seed 42 |
| 学习率 | warmup 1K，peak `2.5e-5`，cosine decay 30K 到 `2.5e-6` |
| 总步数 | 35K；5K 间隔保存，`keep_period=1` 保留全部检查点 |

三个实验只改变 Student 视角、对应 config 和输出目录；teacher、随机种子、损失、
优化器、数据集与硬件配置保持一致。`decay-steps=30000` 延续 H9，因此 30K--35K
阶段使用末端学习率平台，不把多视角结果与 H9 的学习率边界混为新的方法消融。

## 预注册比较

每个视角完成 5K--35K 检查点后，使用相同的四套 LIBERO benchmark 和相同 episode
协议验证 30K、35K；5K、10K、15K、20K、25K 检查点保留，是否评测由后续资源安排
决定。主要比较是各视角 ACPD-v2 与相同视角 BS64 SFT baseline，不能使用 backview
baseline 替代对应视角对照。

## 通过条件

训练必须完成首个检查点且无 NaN/Inf；最终结论只使用闭环成功率。若任务失败，记录
工程状态，不把训练 loss 当作方法结论。
