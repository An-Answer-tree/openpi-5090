# H14：多视角 SFT BS64 30K Baseline

## 目的

补齐 topview、leftview、rightview 的 view-matched SFT baseline，供后续各视角
ACPD-v2 实验比较。训练 loss 只用于检查运行健康，不作为任务成功率结论。

## 固定设置

| 设置 | 数值 |
|---|---|
| 数据集 | `libero_multiview_tuned_6view_lerobot` |
| Student 初始化 | pi0.5 base |
| Student 视角 | topview / leftview / rightview，每个任务一个视角 |
| LoRA | PaliGemma rank 16；action expert rank 32 |
| Seed | 42 |
| 设备 | 4×RTX 5090，FSDP4 |
| Global batch | physical BS64，每卡 16，无梯度累积 |
| 训练步数 | 30K |
| 学习率 | 1K warmup；cosine `2.5e-5`→`2.5e-6`，30K decay |
| Checkpoint | 每 5K 保留，最终保存 step 29,999 |

三项训练除 fixed-view 配置、实验名和输出目录外完全一致。checkpoint 验证范围
另行锁定；验证结果产生前不比较视角优劣。
