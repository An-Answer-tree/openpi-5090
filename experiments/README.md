# 实验索引

本目录保存实验协议、分析和小型结果文件。正式结果以
[`EXPERIMENT_RESULTS.md`](../EXPERIMENT_RESULTS.md) 为准，大型 checkpoint 与验证视频保存在
`/opt/liutong`。

名称统一为 H9-Fixed（固定 contribution 权重）、H17-Decay（非零 contribution
权重衰减）、H13-LossOnly（无注入）。详细配置对照见结果台账；旧 ID、目录和任务名
保留用于检索。“Decay”只表示 contribution 权重衰减，两组 LR 均为原30K cosine。

## 学术命名

论文和汇报使用下表的规范名称；H 编号、Slurm job 和文件路径只作为历史索引。

| 规范名称 | 含义 | 历史索引 |
|---|---|---|
| `SFT-View(BV/LV/RV/TV)` | 对应相机视角的 LoRA SFT 基线 | H12、H14 |
| `ACPD-v2-Fixed` | layer 10，固定 contribution 权重 0.2，保留 residual 注入 | H9-Fixed、H9-scale-b |
| `ACPD-v2-Decay` | 10K 后 contribution 权重由 0.2 衰减到 0.05 | H17-Decay |
| `ACPD-v2-NoInjection` | 保留 contribution 监督，关闭 residual 注入 | H13-LossOnly |
| `ACL-only` | 仅保留 action-level consistency loss，不学习或注入 contribution | ACL-only BS64 |
| `Contribution-Recovery-Probe` | 测量不同 transformer 层的 teacher contribution 可恢复性 | H7/H7.1 |
| `Gradient-Compatibility-Diagnostic` | 测量 flow、contribution、ACL 梯度的方向关系 | H15a |
| `Component-Sweep-BS32` | 早期 flow/ACL/contribution 组件筛选 | H4/H5/H8/H9（BS32） |
| `LoRA-FSDP5090-Infrastructure` | 5090 上的训练可行性和显存基线 | H1/H2 |

`ACPD` 在正文首次出现时写作 **Attention Contribution Privileged
Distillation**；`ACL` 写作 **Action-Consistency Learning**。不同 batch size、
训练步数或验证规模的实验不能仅因名称相近而合并。

## 目录

| 分类 | 内容 |
|---|---|
| `baseline/` | BS64 单视角 SFT 基线 |
| `student/` | ACPD-v2 主 student；`preliminary/` 保存早期 BS32 筛选 |
| `ablation/` | 正式消融；`preliminary/` 保存早期 BS32/2K/5K 筛选 |
| `mechanism/` | exact attention contribution probe 与层扫描 |
| `infrastructure/` | 5090、FSDP、LoRA 可行性验证 |
| `archive/` | 已结束且不进入最终 checkpoint 集合的实验 |

改进路线与低成本实验决策树见 [`acpd-v2-research-roadmap.md`](acpd-v2-research-roadmap.md)。

## 正式实验

| 类别 | 实验 | 协议/分析 | 状态 |
|---|---|---|---|
| baseline | backview BS64 5K--30K | `baseline/sft-backview-bs64-5k/` | 10K/15K/20K/25K/30K 正式验证完成 |
| baseline | backview BS64 30K--60K | `baseline/sft-backview-bs64-trajectory-60k/` | 续训完成；35K为`57.15%`，40K/50K/60K验证排队 |
| baseline | top/left/right BS64 30K | `baseline/sft-multiview-bs64-30k/` | 全部完成 |
| student | H9-Fixed 主实验（原H9-scale-b） | `student/acpd-v2-h9-batch-scaling-30k/` | 固定contribution=0.2；5K/10K/15K/25K/30K/35K有正式结果 |
| mechanism | H15a ACPD-v2 梯度冲突诊断 | `mechanism/acpd-v2-h15a-gradient-conflict/` | 完成；不支持后期梯度冲突假设 |
| student | H9-Fixed 延长轨迹 | `student/acpd-v2-h9-trajectory-60k/` | 45K checkpoint完整后停训；35K为`61.05%`，40K为`59.45%`，45K验证已取消；50K/60K未训练 |
| student | H9-Fixed 中期轨迹20K--30K | `student/acpd-v2-h9-20k-30k-trajectory/` | 恢复训练到20K；25K验证完成，20K验证运行中 |
| student | H9-Fixed/SFT 早期轨迹10K--15K | `student/acpd-v2-h9-early-trajectory/` | 10K/15K正式比较完成 |
| ablation | H13-LossOnly BS64 | `ablation/acpd-v2-h13-injection-ablation/` | 5K 训练和验证完成；支持显式注入 |
| ablation | H17-Decay BS64 | `ablation/acpd-v2-h17-contribution-decay/` | 从H9-Fixed 10K分支；训练134422运行，验证134423、配对分析134424等待依赖 |
| ablation | ACL-only BS64 35K | `ablation/acpd-v2-acl-only-bs64-35k/protocol.md` | 训练135724排队；30K/35K全量验证135725/135726、分析135729/135728等待依赖；尚无结论 |

H11 已归档，不进入正式 checkpoint 集合，也不继续训练。
H9-Fixed 的恢复、早中期验证及延长训练是同一方法的不同执行阶段，不作为独立消融。

## 精选 Checkpoint 完整性

根目录：`/opt/liutong/openpi_checkpoints/fixed_dataset/curated`

符号说明：`Y` 已硬链接归档，`C` 已完成但未归档，`R` 正在运行或恢复，`Q` 已提交排队，`P` 待后续补训，`-` 不适用。

| 模型 | 5K | 10K | 15K | 20K | 25K | 30K |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| teacher/agentview_wrist | - | - | - | - | - | Y |
| baseline/backview_bs64 | Y | C | C | C | C | C |
| baseline/topview_bs64 | C | C | C | C | C | C |
| baseline/leftview_bs64 | C | C | C | C | C | C |
| baseline/rightview_bs64 | C | C | C | C | C | C |
| student/backview_acpdv2_layer10_bs64（H9-Fixed） | C | C | C | C | Y | C |
| ablation/H17-Decay | - | 起点为H9-Fixed 10K | Q | Q | - | - |
| ablation/backview_loss_only_bs64 | C | P | P | P | P | P |
| ablation/backview_acl_only_bs64 | Q | Q | Q | Q | Q | Q |

H9-Fixed 35K/40K/45K checkpoint 已完整写入但未归档。原目录中的 checkpoint
本次不移动、不删除。旧 BS16/BS32 checkpoint 不进入该表，仍保留在原路径。

## 归档规则

- checkpoint 必须已经完整写入且对应 Slurm 任务不再写入该目录。
- 使用 `scripts/research/curate_checkpoint.sh SOURCE DEST` 创建文件级硬链接。
- `DEST` 必须位于同一 `/opt/liutong` 文件系统且事先不存在。
- 归档脚本逐文件比较相对路径、inode 和大小；验证完成前不得删除源目录。
- 每次只处理一个明确 checkpoint，不批量扫描存储盘。
