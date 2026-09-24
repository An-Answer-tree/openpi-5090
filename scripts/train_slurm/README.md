# Slurm 训练脚本

顶层保留正式训练 launcher；已结束的一次性恢复和诊断 launcher 放在 `archive/`。
Slurm 已提交任务的状态以 `squeue` 为准，脚本留在顶层不代表仍在运行。

| 类型 | 脚本 |
|---|---|
| H9 主 student | `pi05_libero_backview_acpd_v2_layer10_lora_fsdp4_bs64_30k.sbatch` |
| H9 轨迹恢复 | `pi05_libero_backview_acpd_v2_layer10_lora_fsdp4_bs64_20k_keep_all.sbatch` |
| H9 30K 后续训（已在45K停训） | `pi05_libero_backview_acpd_v2_layer10_lora_fsdp4_bs64_resume_60k.sbatch` |
| H13 loss-only | `pi05_libero_backview_acpd_v2_h13_loss_only_fsdp4_bs64_5k.sbatch` |
| H17 非零 contribution 衰减 | `pi05_libero_backview_acpd_v2_h17_decay_fsdp4_bs64_20k.sbatch` |
| BS64 baseline | `pi05_libero_{backview,topview,leftview,rightview}_lora_fsdp4_bs64_*.sbatch` |

`archive/legacy_bs32/` 保存旧 BS16/BS32 launcher；`archive/diagnostics/` 保存已完成的
probe launcher；`archive/completed/` 保存一次性25K恢复脚本；`archive/abandoned/`
保存不再继续的 H11 launcher。归档脚本不用于新的正式提交。
