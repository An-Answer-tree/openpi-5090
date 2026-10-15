# Checkpoint 检索表

核对日期：2026-10-16。存储根目录：`/opt/liutong/openpi_checkpoints/fixed_dataset`。
本表区分实际归档与原训练目录，未列出的历史 checkpoint 不代表已删除。

## 精选目录实际内容

目录：[/opt/liutong/openpi_checkpoints/fixed_dataset/curated](/opt/liutong/openpi_checkpoints/fixed_dataset/curated)。
共4类、8个模型目录，实际归档3个 checkpoint，另外5个目录为空。
这是尚未补齐的精选集合，不是当前全部终点或最高观测点的集合。

| 分类 | 模型目录 | 已归档 checkpoint | 内容 |
|---|---|---|---|
| teacher | `teacher/agentview_wrist` | `29999`（30K） | `params`、`assets`；无 `train_state`，不能恢复优化器状态 |
| baseline | `baseline/backview_bs64` | `4999`（5K） | 含 `params`、`assets`、`train_state`、`_CHECKPOINT_METADATA` |
| baseline | `baseline/leftview_bs64` | 无 | 空目录 |
| baseline | `baseline/rightview_bs64` | 无 | 空目录 |
| baseline | `baseline/topview_bs64` | 无 | 空目录 |
| student | `student/backview_acpdv2_layer10_bs64` | `24999`（25K） | 含 `params`、`assets`、`train_state`、`_CHECKPOINT_METADATA` |
| ablation | `ablation/backview_acl_only_bs64` | 无 | 空目录 |
| ablation | `ablation/backview_loss_only_bs64` | 无 | 空目录 |

精选目录尚无 leftview/rightview/topview student 子目录。H11不进入精选集合。

### 三个归档的来源

| 归档 | 原始 checkpoint |
|---|---|
| teacher 30K | [sft/agentview_wrist/29999](/opt/liutong/openpi_checkpoints/fixed_dataset/sft/agentview_wrist/29999) |
| backview SFT 5K | [backview_bs64_5k / 4999](/opt/liutong/openpi_checkpoints/fixed_dataset/sft/backview_bs64_5k/pi05_libero_backview_lora/pi05_libero_backview_lora_fsdp4_bs64_5k/4999) |
| backview ACPD-v2 25K | [batch_scaling_30k / 24999](/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/batch_scaling_30k/pi05_libero_backview_acpd_v2_layer10/pi05_libero_backview_acpd_v2_lora_fsdp4_bs64_30k/24999) |

三个归档均抽查 `params/_METADATA`：归档与来源的设备号、inode相同，链接数至少2。
这与文件级硬链接归档一致；本次未逐文件检查或读取模型权重，不作完整性认证。

## 仍在原训练目录的主要模型

以下明确路径均已核对存在 `params/_METADATA`、`assets`、`train_state` 和
`_CHECKPOINT_METADATA`，尚未加入上述精选目录。本次未加载模型或测试恢复训练。
成功率沿用已有正式结果，每点四套各500回合；完整比较见[结果台账](../EXPERIMENT_RESULTS.md)。

| 模型 | 步数 | Checkpoint | 成功率与验证证据 |
|---|---:|---|---|
| ACPD-v2-Fixed backview | 35K | [34999](/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/batch_scaling_30k/pi05_libero_backview_acpd_v2_layer10/pi05_libero_backview_acpd_v2_lora_fsdp4_bs64_30k/34999) | [61.05%](/opt/liutong/openpi-5090-evals/acpd-v2-training-trajectory/final-hidden/34999/summary.txt) |
| ACPD-v2-Fixed leftview | 32K | [31999](/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/trajectory_31k_34k/leftview/pi05_libero_leftview_acpd_v2_layer10/pi05_libero_leftview_acpd_v2_layer10_lora_fsdp4_bs64_resume_34k/31999) | [81.55%](/opt/liutong/openpi-5090-evals/acpd-v2-trajectory-31k-34k/leftview/31999/summary.txt) |
| ACPD-v2-Fixed rightview | 32K | [31999](/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/trajectory_31k_34k/rightview/pi05_libero_rightview_acpd_v2_layer10/pi05_libero_rightview_acpd_v2_layer10_lora_fsdp4_bs64_resume_34k/31999) | [81.00%](/opt/liutong/openpi-5090-evals/acpd-v2-trajectory-31k-34k/rightview/31999/summary.txt) |
| ACPD-v2-Fixed topview | 32K | [31999](/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/multiview_35k/topview/pi05_libero_topview_acpd_v2_layer10/pi05_libero_topview_acpd_v2_layer10_lora_fsdp4_bs64_35k/31999) | [75.50%](/opt/liutong/openpi-5090-evals/acpd-v2-multiview-35k/topview/31999/summary.txt) |
| SFT backview BS64 | 30K | [29999](/opt/liutong/openpi_checkpoints/fixed_dataset/sft/backview_bs64_5k/pi05_libero_backview_lora/pi05_libero_backview_lora_fsdp4_bs64_5k/29999) | [60.45%](/opt/liutong/openpi-5090-evals/sft-backview-bs64-30k/29999/summary.txt) |
| SFT leftview BS64 | 30K | [29999](/opt/liutong/openpi_checkpoints/fixed_dataset/sft/leftview_bs64_30k/pi05_libero_leftview_lora/pi05_libero_leftview_lora_fsdp4_bs64_30k/29999) | [78.65%](/opt/liutong/openpi-5090-evals/sft-leftview-bs64-30k/29999/summary.txt) |
| SFT rightview BS64 | 30K | [29999](/opt/liutong/openpi_checkpoints/fixed_dataset/sft/rightview_bs64_30k/pi05_libero_rightview_lora/pi05_libero_rightview_lora_fsdp4_bs64_30k/29999) | [77.00%](/opt/liutong/openpi-5090-evals/sft-rightview-bs64-30k/29999/summary.txt) |
| SFT topview BS64 | 30K | [29999](/opt/liutong/openpi_checkpoints/fixed_dataset/sft/topview_bs64_30k/pi05_libero_topview_lora/pi05_libero_topview_lora_fsdp4_bs64_30k/29999) | [71.55%](/opt/liutong/openpi-5090-evals/sft-topview-bs64-30k/29999/summary.txt) |
| SFT backview BS64 | 60K | [59999](/opt/liutong/openpi_checkpoints/fixed_dataset/sft/backview_bs64_5k/pi05_libero_backview_lora/pi05_libero_backview_lora_fsdp4_bs64_5k/59999) | [62.05%](/opt/liutong/openpi-5090-evals/sft-backview-bs64-training-trajectory/59999/summary.txt) |
| ACL-only backview BS64 | 35K | [34999](/opt/liutong/openpi_checkpoints/fixed_dataset/ablation/acpd_v2_acl_only_bs64_35k/pi05_libero_backview_acl_only_bs64_35k/pi05_libero_backview_acl_only_lora_fsdp4_bs64_35k/34999) | [59.35%](/opt/liutong/openpi-5090-evals/acpd-v2-acl-only-bs64-35k/34999/summary.txt) |

四个ACPD-v2点为已有完整评测中的最高观测点，属于探索性checkpoint选择，
不代表独立seed最优或统计显著优势。rightview 34K在最近记录中尚未完成全量评测。
三视角SFT 30K→32K已提交，不将预期新增checkpoint列为现存文件。
H13仅有5K训练与验证结果，不把曾计划的30K记为已完成；其他历史模型路径见结果台账。

## 维护规则

- 本次仅检查精选目录的浅层条目和上述明确路径；未递归扫盘，未复制、移动或删除checkpoint。
- 空归档目录不表示模型不存在；应先按上表和结果台账检索原路径。
- 归档只处理已完成、且无任务正在写入的checkpoint；沿用[归档规则](README.md#归档规则)。
- 后续补充归档时更新本表；保留中间checkpoint，不以最高观测点替代训练轨迹。
