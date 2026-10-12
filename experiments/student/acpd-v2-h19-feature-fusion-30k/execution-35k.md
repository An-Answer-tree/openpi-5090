# H19 30K→35K 续训执行记录

提交时间：2026-10-12 14:54 CST。协议提交：`a6fb25f`。任务：`147300`。

| 项目 | 设置 |
|---|---|
| 状态 | 15:16:41在gpu04开始；15:28确认实际更新参数且训练持续推进 |
| 资源 | batch，4×5090，32 CPU，96G，3天，预约`res_gpu04` |
| 训练 | FSDP4 LoRA，physical global BS64，累计1，30K→35K |
| 恢复 | 源checkpoint只读，恢复模型、优化器与步数；seed42、BS64，从batch30000接续 |
| 模型与loss | H19 FeatureFusion、layer10；flow/contribution/ACL=1/0.2/0.5 |
| 学习率 | 沿用30K cosine；30K以后保持2.5e-6，不重启warmup |
| 保存 | 每1K保存，keep_period=1；保留30999/31999/32999/33999/34999 |
| 检查 | 5项恢复测试通过；bash语法、Slurm test-only及保存脚本通过；实际恢复step30000、batch30000，首个训练loss有限 |
| 验证 | 本次只提交训练；尚无35K模型和成功率结果 |

- 脚本：`scripts/train_slurm/pi05_libero_backview_acpd_v2_feature_fusion_fsdp4_bs64_resume_35k.sbatch`
- 源checkpoint：`/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/feature_fusion_30k/pi05_libero_backview_acpd_v2_feature_fusion/pi05_libero_backview_acpd_v2_feature_fusion_lora_fsdp4_bs64_30k/29999`
- 目标目录：`/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/feature_fusion_35k/pi05_libero_backview_acpd_v2_feature_fusion/pi05_libero_backview_acpd_v2_feature_fusion_lora_fsdp4_bs64_35k`
- 日志：`/opt/liutong/openpi-5090-research/acpd-v2-h19-feature-fusion-35k/slurm-log/pi05-bv-h19-feature-fusion-r35k_147300.out`

原三项35K验证及其汇总继续执行，本次未修改其他任务，也未移动或复制checkpoint。
