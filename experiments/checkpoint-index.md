# Checkpoint 检索表

更新时间：2026-10-16。精选根目录：
[/opt/liutong/openpi_checkpoints/fixed_dataset/curated](/opt/liutong/openpi_checkpoints/fixed_dataset/curated)。
本次按用户指定更新正式选择；[机器可读清单](curated-checkpoints.yaml)保存来源、
归档相对路径、真实步数、推理配置及四套验证成功数。

## Teacher、SFT 与 student

以下路径均相对于精选根目录，已建立文件级硬链接。teacher仅含推理权重，
其余模型保留训练状态；原训练目录不变。

| 分类 | 模型 | 实际训练步数 | 当前选择 | 四套平均成功率 |
|---|---|---:|---|---:|
| teacher | agentview+wrist | 30K | `teacher/agentview_wrist/29999` | — |
| baseline | backview SFT BS64 | 30K | `baseline/backview_bs64/29999` | 60.45% |
| baseline | leftview SFT BS64 | 30K | `baseline/leftview_bs64/29999` | 78.65% |
| baseline | rightview SFT BS64 | 30K | `baseline/rightview_bs64/29999` | 77.00% |
| baseline | topview SFT BS64 | 30K | `baseline/topview_bs64/29999` | 71.55% |
| student | backview ACPD-v2 BS64 | 35K | `student/backview_acpdv2_layer10_bs64/34999` | 61.05% |
| student | leftview ACPD-v2 BS64 | 32K | `student/leftview_acpdv2_layer10_bs64/31999` | 81.55% |
| student | rightview ACPD-v2 BS64 | 32K | `student/rightview_acpdv2_layer10_bs64/31999` | 81.00% |
| student | topview ACPD-v2 BS64 | 32K | `student/topview_acpdv2_layer10_bs64/31999` | 75.50% |

## 消融与第三方方法：五类

按用户图中除SFT外的五行分类，依据验证数值识别模型，不沿用图中统一的30K标注。
所有成功率均来自四套各500回合的完整结果，LIBERO-10对应图中的long。

| 图中类别 | 实际模型 | Spatial | Object | Goal | LIBERO-10 | 平均 | 精选相对路径 |
|---|---|---:|---:|---:|---:|---:|---|
| Multi-View-to-Single-View Knowledge Distillation | MV-SV-KD，目标30K、BS64；训练142275进行中 | — | — | — | — | — | `ablation/backview_mv_sv_kd_bs64/`，空目录，尚无30K模型与验证结果 |
| ACL Only | ACL-only，35K、BS64 | 67.00% | **75.20%** | 63.60% | 31.60% | **59.35%** | `ablation/backview_acl_only_bs64/34999` |
| ACL+CrossattnDistillationLoss | 原论文ACPD，50K、BS32 | 60.20% | 62.80% | 64.60% | 24.80% | 53.10% | `ablation/backview_acl_crossattn_bs32/49999` |
| ACL + DistillationLoss | H9 30K、BS64，仅推理关闭注入 | 65.00% | 73.00% | 64.40% | 33.20% | 58.90% | `ablation/backview_acl_distillation_no_injection_bs64/29999` |
| ACL+DistillationLoss+Injection (Ours ACPD) | H9 35K、BS64，开启注入 | 73.20% | 73.80% | 63.60% | 33.60% | 61.05% | `ablation/backview_acpd_v2_full_bs64/34999` |

图中ACL-only的Object 73.20%、平均58.85%与原始记录不符；原始记录为376/500
（75.20%），四套合计1187/2000（59.35%），本表采用原始记录。
图中SFT的57.15%实际为35K；按用户本次要求，baseline归档统一使用30K，backview为60.45%。

58.90%不是H13从头训练的无注入消融。它复用有注入训练的H9 30K权重，必须使用
`pi05_libero_backview_acpd_v2_layer10_loss_only_lora`推理，才对应上述数值。
61.05%使用`pi05_libero_backview_acpd_v2_layer10_lora`。
模型目录名本身不会切换推理配置。

这五类是存储分类；实际BS、训练步数和推理设置不同，不视为仅改变单个组件的受控消融。

## 验证证据

各模型的精确来源和汇总文件见[清单](curated-checkpoints.yaml)。
详细结果继续维护于[实验结果台账](../EXPERIMENT_RESULTS.md)。

| 内容 | 证据 |
|---|---|
| 四视角SFT 30K | `sft-{view}-bs64-30k/29999/summary.txt`，位于`/opt/liutong/openpi-5090-evals/` |
| backview ACPD-v2 35K | `/opt/liutong/openpi-5090-evals/acpd-v2-training-trajectory/final-hidden/34999/summary.txt` |
| left/rightview ACPD-v2 32K | `/opt/liutong/openpi-5090-evals/acpd-v2-trajectory-31k-34k/{view}/31999/summary.txt` |
| topview ACPD-v2 32K | `/opt/liutong/openpi-5090-evals/acpd-v2-multiview-35k/topview/31999/summary.txt` |
| ACL-only 35K | `/opt/liutong/openpi-5090-evals/acpd-v2-acl-only-bs64-35k/34999/summary.txt` |
| 原论文ACPD 50K | `/opt/liutong/openpi-5090-evals/acpd-original-backview-bs32-50k/49999/summary.txt` |
| H9 30K推理关闭注入 | `/opt/liutong/openpi-5090-evals/acpd-v2-h9-inference-ablation/29999/summary.txt` |

## 保留与核对

- 正式选择共14项：13个checkpoint路径已归档，MV-SV-KD一项待30K完成；teacher无优化器状态。
- 实存共15个checkpoint目录：13个当前选择，加历史SFT5K与H9 25K两个；完整H9 35K在student与ablation各有一个硬链接入口。
- 历史归档`baseline/backview_bs64/4999`、`student/backview_acpdv2_layer10_bs64/24999`仍保留，不作为当前正式选择。
- 新增12个硬链接归档，不复制权重数据块；抽查每项元数据及一个权重分片的设备号/inode与来源一致，未逐文件认证或测试模型加载。
- 仅处理清单中的明确checkpoint，未扫描其他训练目录；不移动、删除或改写原checkpoint，不改变正在运行或排队任务。
- 已删除被新分类替代的空目录`ablation/backview_loss_only_bs64`，其中原本没有checkpoint。
