# Slurm 训练脚本

顶层只保留当前运行、排队、依赖任务及下一阶段继续训练需要的脚本。

| 类型 | 脚本 |
|---|---|
| H9 主 student | `pi05_libero_backview_acpd_v2_layer10_lora_fsdp4_bs64_30k.sbatch` |
| H9 轨迹恢复 | `pi05_libero_backview_acpd_v2_layer10_lora_fsdp4_bs64_20k_keep_all.sbatch` |
| H9 25K 恢复 | `restore_pi05_libero_backview_acpd_v2_bs64_25k.sbatch` |
| H13 loss-only | `pi05_libero_backview_acpd_v2_h13_loss_only_fsdp4_bs64_5k.sbatch` |
| BS64 baseline | `pi05_libero_{backview,topview,leftview,rightview}_lora_fsdp4_bs64_*.sbatch` |

`archive/legacy_bs32/` 保存旧 BS16/BS32 launcher，`archive/diagnostics/` 保存 probe
launcher，`archive/abandoned/` 保存不再继续的 H11 launcher。归档脚本不会用于新的正式提交。
