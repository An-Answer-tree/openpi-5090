# 最终模型与验证结果检索表

更新时间：2026-10-24。推理归档根目录：
[/opt/liutong/openpi_checkpoints/fixed_dataset/curated](/opt/liutong/openpi_checkpoints/fixed_dataset/curated)。
按用户最终图片的四套成功率匹配模型。当前归档不含步数目录或训练状态；
[内部清单](curated-checkpoints.yaml)保留真实来源、更新数、推理配置及完整备份对应路径。
完整备份：[/data0/liutong/ACPD_archive](/data0/liutong/ACPD_archive)，保留原目录、训练状态和原始验证日志。

## 四视角结果

每套500回合；LIBERO-10对应图片中的long。以下checkpoint路径均相对于汇总根目录。

| 视角 | 模型 | Spatial | Object | Goal | LIBERO-10 | 平均 | 成功/总数 | Checkpoint |
|---|---|---:|---:|---:|---:|---:|---:|---|
| backview | SFT | 65.80% | 68.40% | 64.60% | 29.80% | 57.15% | 1143/2000 | `baseline/backview_bs64` |
| backview | ACPD-v2 | 73.20% | 73.80% | 63.60% | 33.60% | 61.05% | 1221/2000 | `student/backview_acpdv2_layer10_bs64` |
| leftview | SFT | 84.00% | 79.60% | 79.40% | 71.60% | 78.65% | 1573/2000 | `baseline/leftview_bs64` |
| leftview | ACPD-v2 | 85.20% | 79.40% | 84.80% | 76.80% | 81.55% | 1631/2000 | `student/leftview_acpdv2_layer10_bs64` |
| rightview | SFT | 85.00% | 84.80% | 84.00% | 54.20% | 77.00% | 1540/2000 | `baseline/rightview_bs64` |
| rightview | ACPD-v2 | 90.20% | 88.60% | 85.80% | 59.40% | 81.00% | 1620/2000 | `student/rightview_acpdv2_layer10_bs64` |
| topview | SFT | 81.00% | 80.60% | 79.20% | 45.40% | 71.55% | 1431/2000 | `baseline/topview_bs64` |
| topview | ACPD-v2 | 85.00% | 81.60% | 80.60% | 54.80% | 75.50% | 1510/2000 | `student/topview_acpdv2_layer10_bs64` |

backview SFT原checkpoint编号35000实际包含35001次更新，通常称35K；backview ACPD为35K，
其余SFT为30K，其余student为32K。按图片选取结果不代表训练步数匹配的比较。
teacher位于`teacher/agentview_wrist`，图片未提供其验证结果。
当前所有归档模型仅保留`params/`与`assets/`；完整备份中除teacher外保留训练状态。

## Backview：第三方方法与消融分类

| 图片类别 | 实际配置 | Spatial | Object | Goal | LIBERO-10 | 平均 | 成功/总数 | Checkpoint |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Multi-View-to-Single-View Knowledge Distillation | MV-SV-KD，30K、BS64 | 61.80% | 59.80% | 61.00% | 27.00% | 52.40% | 1048/2000 | `ablation/backview_mv_sv_kd_bs64` |
| ACL Only | ACL-only，31K、BS64 | 69.00% | 72.80% | 65.60% | 31.60% | 59.75% | 1195/2000 | `ablation/backview_acl_only_bs64` |
| ACL+CrossattnDistillationLoss | 原论文ACPD，50K、BS32 | 60.20% | 62.80% | 64.60% | 24.80% | 53.10% | 1062/2000 | `ablation/backview_acl_crossattn_bs32` |
| ACL + DistillationLoss | H9 30K、BS64，仅推理关闭注入 | 65.00% | 73.00% | 64.40% | 33.20% | 58.90% | 1178/2000 | `ablation/backview_acl_distillation_no_injection_bs64` |
| ACL+DistillationLoss+Injection (Ours ACPD) | H9 35K、BS64，开启注入 | 73.20% | 73.80% | 63.60% | 33.60% | 61.05% | 1221/2000 | `ablation/backview_acpd_v2_full_bs64` |

ACL-only四套成功率与图片完全一致，平均应为59.75%，图片的57.75%为算术差异；
清单同时保留图片值和核算值。MV-SV-KD是在pi0.5上适配的离线基线，非原论文算法逐项复现。
58.90%来自有注入训练的H9权重，仅推理关闭注入，不是从头训练的H13。
这些是按图片结果整理的存储类别，BS、步数和推理设置不同，不属于单因素受控消融。

## 推理配置

目录名不会自动切换推理配置。加载归档checkpoint时，参数位于其`params/`。

| 模型 | policy_config |
|---|---|
| 四视角SFT | `pi05_libero_{view}_lora`，view为backview/leftview/rightview/topview |
| 四视角ACPD-v2及完整ACPD消融入口 | `pi05_libero_{view}_acpd_v2_layer10_lora` |
| ACL-only、H9推理关闭注入 | `pi05_libero_backview_acpd_v2_layer10_loss_only_lora`；注入关闭 |
| MV-SV-KD、原论文ACPD | `pi05_libero_backview_lora` |

## 验证结果与保留规则

每个已评测模型的验证文件位于`results/<checkpoint相对路径>/`，包含
`summary.json`与`summary.txt`，只记录suite顺序、回合数、成功数及成功率。例如：

`results/student/backview_acpdv2_layer10_bs64/summary.txt`

清单的`result_summary`指向原始汇总，`result_archive`指向当前推理归档。
`full_backup_archive`和`full_backup_result_archive`相对于`full_backup_root`，
可定位带编号的完整checkpoint和原始四套日志。
视频保留在原汇总同级的`{suite}/videos/`，未枚举或另行归档视频。

- 当前14个推理模型：teacher 1、baseline 4、student 4、ablation 5；13组结果入口、12组独立来源。
- 2026-10-24转换时核对199个权重及归一化文件的inode和大小；全部模型通过Orbax元数据读取及CPU动作输出层恢复检查，未重新运行GPU推理或仿真。
- 推理归档已移除训练状态、根checkpoint元数据、含来源路径的旧日志和图片；完整版本保留在`/data0`独立备份。不得原地改写硬链接权重。
- 原训练checkpoint及数据集保持原位；只检查选定归档文件，未扫描周围训练或视频目录。
- [转换记录](checkpoint-inference-export-2026-10-24.md)记录备份核对、移除范围及加载方式；[此前清理记录](checkpoint-archive-cleanup-2026-10-19.md)保留历史入口信息。

完整实验记录见[实验结果台账](/home/liutong/Projects/openpi-5090/EXPERIMENT_RESULTS.md)。
