# 最终模型与验证结果检索表

更新时间：2026-10-19。汇总根目录：
[/opt/liutong/openpi_checkpoints/fixed_dataset/curated](/opt/liutong/openpi_checkpoints/fixed_dataset/curated)。
按用户最终图片的四套成功率匹配模型，保留真实checkpoint编号，不沿用图片step。
[精确清单](curated-checkpoints.yaml)记录来源、归档位置、真实更新数、推理配置和成功数。

## 四视角结果

每套500回合；LIBERO-10对应图片中的long。以下checkpoint路径均相对于汇总根目录。

| 视角 | 模型 | Spatial | Object | Goal | LIBERO-10 | 平均 | 成功/总数 | Checkpoint |
|---|---|---:|---:|---:|---:|---:|---:|---|
| backview | SFT | 65.80% | 68.40% | 64.60% | 29.80% | 57.15% | 1143/2000 | `baseline/backview_bs64/35000` |
| backview | ACPD-v2 | 73.20% | 73.80% | 63.60% | 33.60% | 61.05% | 1221/2000 | `student/backview_acpdv2_layer10_bs64/34999` |
| leftview | SFT | 84.00% | 79.60% | 79.40% | 71.60% | 78.65% | 1573/2000 | `baseline/leftview_bs64/29999` |
| leftview | ACPD-v2 | 85.20% | 79.40% | 84.80% | 76.80% | 81.55% | 1631/2000 | `student/leftview_acpdv2_layer10_bs64/31999` |
| rightview | SFT | 85.00% | 84.80% | 84.00% | 54.20% | 77.00% | 1540/2000 | `baseline/rightview_bs64/29999` |
| rightview | ACPD-v2 | 90.20% | 88.60% | 85.80% | 59.40% | 81.00% | 1620/2000 | `student/rightview_acpdv2_layer10_bs64/31999` |
| topview | SFT | 81.00% | 80.60% | 79.20% | 45.40% | 71.55% | 1431/2000 | `baseline/topview_bs64/29999` |
| topview | ACPD-v2 | 85.00% | 81.60% | 80.60% | 54.80% | 75.50% | 1510/2000 | `student/topview_acpdv2_layer10_bs64/31999` |

backview SFT目录35000实际包含35001次更新，通常称35K；backview ACPD为35K，
其余SFT为30K，其余student为32K。按图片选取结果不代表训练步数匹配的比较。
teacher保留在`teacher/agentview_wrist/29999`，仅含推理权重，没有优化器状态；
图片未提供teacher验证结果，其余所选checkpoint均保留训练状态。

## Backview：第三方方法与消融分类

| 图片类别 | 实际配置 | Spatial | Object | Goal | LIBERO-10 | 平均 | 成功/总数 | Checkpoint |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Multi-View-to-Single-View Knowledge Distillation | MV-SV-KD，30K、BS64 | 61.80% | 59.80% | 61.00% | 27.00% | 52.40% | 1048/2000 | `ablation/backview_mv_sv_kd_bs64/29999` |
| ACL Only | ACL-only，31K、BS64 | 69.00% | 72.80% | 65.60% | 31.60% | 59.75% | 1195/2000 | `ablation/backview_acl_only_bs64/30999` |
| ACL+CrossattnDistillationLoss | 原论文ACPD，50K、BS32 | 60.20% | 62.80% | 64.60% | 24.80% | 53.10% | 1062/2000 | `ablation/backview_acl_crossattn_bs32/49999` |
| ACL + DistillationLoss | H9 30K、BS64，仅推理关闭注入 | 65.00% | 73.00% | 64.40% | 33.20% | 58.90% | 1178/2000 | `ablation/backview_acl_distillation_no_injection_bs64/29999` |
| ACL+DistillationLoss+Injection (Ours ACPD) | H9 35K、BS64，开启注入 | 73.20% | 73.80% | 63.60% | 33.60% | 61.05% | 1221/2000 | `ablation/backview_acpd_v2_full_bs64/34999` |

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
`summary.txt`以及`logs/{spatial,object,goal,libero_10}.log`。例如：

`results/student/backview_acpdv2_layer10_bs64/34999/summary.txt`

清单的`result_summary`指向原始汇总，`result_archive`指向本次归档。
视频保留在原汇总同级的`{suite}/videos/`，未枚举或另行归档视频。
汇总中的原日志绝对路径保留，同时归档对应四套日志，便于核对每任务成功数。

- 当前选择14个checkpoint入口：teacher 1、baseline 4、student 4、ablation 5，均已归档。
- 验证结果13个入口、12组独立来源；完整ACPD与backview student共用模型和验证来源。
- 汇总目录当前仅保留上述14个checkpoint入口；按用户要求删除了4个不再选用的历史硬链接入口。原训练checkpoint保留，可重新建立硬链接恢复入口；[清理记录](checkpoint-archive-cleanup-2026-10-19.md)列出准确路径。
- 本次新增3个checkpoint及65个验证文件入口，使用文件硬链接，不复制权重数据块。不得改写硬链接文件，否则原文件也会改变。
- 新增checkpoint逐文件核对inode；其余选择抽查元数据及一个权重分片。验证汇总与原始日志的四套成功数已逐项核对，未重新加载GPU模型。
- 原checkpoint、数据、验证文件及运行任务保持原路径，未移动或删除。清理仅删除汇总目录中的4个历史入口，未释放原权重占用的数据块；未扫描大型目录。

完整实验记录见[实验结果台账](/home/liutong/Projects/openpi-5090/EXPERIMENT_RESULTS.md)。
