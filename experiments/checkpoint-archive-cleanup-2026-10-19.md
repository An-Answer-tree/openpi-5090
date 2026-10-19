# 最终汇总目录清理

日期：2026-10-19。按用户要求，仅清理最终汇总中未被图片选定的历史checkpoint入口。
根目录：`/opt/liutong/openpi_checkpoints/fixed_dataset/curated`。

| 删除的历史入口（相对根目录） | 原checkpoint来源（仍保留） | 文件数 |
|---|---|---:|
| `baseline/backview_bs64/4999` | `/opt/liutong/openpi_checkpoints/fixed_dataset/sft/backview_bs64_5k/pi05_libero_backview_lora/pi05_libero_backview_lora_fsdp4_bs64_5k/4999` | 26 |
| `baseline/backview_bs64/29999` | `/opt/liutong/openpi_checkpoints/fixed_dataset/sft/backview_bs64_5k/pi05_libero_backview_lora/pi05_libero_backview_lora_fsdp4_bs64_5k/29999` | 26 |
| `student/backview_acpdv2_layer10_bs64/24999` | `/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/batch_scaling_30k/pi05_libero_backview_acpd_v2_layer10/pi05_libero_backview_acpd_v2_lora_fsdp4_bs64_30k/24999` | 27 |
| `ablation/backview_acl_only_bs64/34999` | `/opt/liutong/openpi_checkpoints/fixed_dataset/ablation/acpd_v2_acl_only_bs64_35k/pi05_libero_backview_acl_only_bs64_35k/pi05_libero_backview_acl_only_lora_fsdp4_bs64_35k/34999` | 25 |

删除前逐文件核对设备、inode、大小和链接数，104个文件均与上表原文件共享inode。
删除后再次核对104个原文件，均完整保留；当前14个模型入口及13组验证证据保留。
仅遍历上述4个明确checkpoint目录，未扫描大型目录。

这些历史入口是硬链接，删除后原文件仍引用权重数据块，不释放原权重占用的空间。
如需恢复入口，使用`scripts/research/curate_checkpoint.sh 原checkpoint来源 汇总根目录/历史入口`。
未移动或删除原checkpoint，未修改Slurm任务、训练配置或验证结果。
