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

## 数据缓存

现用训练和 LIBERO 验证入口明确设置 `HF_HOME=/opt/liutong/huggingface`、
`HF_DATASETS_CACHE=/opt/liutong/huggingface/datasets`，复用已有缓存，不依赖
交互式 `.bashrc` 或提交时继承的环境。离线设置只关闭联网，不关闭本地缓存。

修改仅对后续启动生效；不重启或修改正在训练的进程。2026-10-14确认任务147971
仍映射 `/home/liutong/.cache/huggingface/datasets/` 的重复Arrow缓存，训练结束且
确认无进程使用前不得清理；不要移动原数据或checkpoint。
