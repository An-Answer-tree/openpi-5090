# 实验索引

本目录保存实验协议、分析和小型结果文件。正式结果以
[`EXPERIMENT_RESULTS.md`](../EXPERIMENT_RESULTS.md) 为准，大型 checkpoint 与验证视频保存在
`/opt/liutong`。

## 目录

| 分类 | 内容 |
|---|---|
| `baseline/` | BS64 单视角 SFT 基线 |
| `student/` | ACPD-v2 主 student；`preliminary/` 保存早期 BS32 筛选 |
| `ablation/` | 正式消融；`preliminary/` 保存早期 BS32/2K/5K 筛选 |
| `mechanism/` | exact attention contribution probe 与层扫描 |
| `infrastructure/` | 5090、FSDP、LoRA 可行性验证 |
| `archive/` | 已结束且不进入最终 checkpoint 集合的实验 |

## 正式实验

| 类别 | 实验 | 协议/分析 | 状态 |
|---|---|---|---|
| baseline | backview BS64 5K--30K | `baseline/sft-backview-bs64-5k/` | 5K 完成并验证；30K 排队 |
| baseline | top/left/right BS64 30K | `baseline/sft-multiview-bs64-30k/` | 排队 |
| student | H9-scale-b ACPD-v2 BS64 | `student/acpd-v2-h9-batch-scaling-30k/` | 训练与恢复中 |
| ablation | H13 loss-only BS64 | `ablation/acpd-v2-h13-injection-ablation/` | 训练中 |
| ablation | ACL-only BS64 | 尚未创建协议 | 待运行 |

H11 已归档，不进入正式 checkpoint 集合，也不继续训练。

## 精选 Checkpoint 完整性

根目录：`/opt/liutong/openpi_checkpoints/fixed_dataset/curated`

符号说明：`Y` 已硬链接归档，`R` 正在运行或恢复，`Q` 已提交排队，`P` 待后续补训，`-` 不适用。

| 模型 | 5K | 10K | 15K | 20K | 25K | 30K |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| teacher/agentview_wrist | - | - | - | - | - | Y |
| baseline/backview_bs64 | Y | Q | Q | Q | Q | Q |
| baseline/topview_bs64 | Q | Q | Q | Q | Q | Q |
| baseline/leftview_bs64 | Q | Q | Q | Q | Q | Q |
| baseline/rightview_bs64 | Q | Q | Q | Q | Q | Q |
| student/backview_acpdv2_layer10_bs64 | R | R | R | R | Y | R |
| ablation/backview_loss_only_bs64 | R | P | P | P | P | P |
| ablation/backview_acl_only_bs64 | P | P | P | P | P | P |

原目录中的 checkpoint 本次不移动、不删除。旧 BS16/BS32 checkpoint 不进入该表，仍保留在原路径。

## 归档规则

- checkpoint 必须已经完整写入且对应 Slurm 任务不再写入该目录。
- 使用 `scripts/research/curate_checkpoint.sh SOURCE DEST` 创建文件级硬链接。
- `DEST` 必须位于同一 `/opt/liutong` 文件系统且事先不存在。
- 归档脚本逐文件比较相对路径、inode 和大小；验证完成前不得删除源目录。
- 每次只处理一个明确 checkpoint，不批量扫描存储盘。
