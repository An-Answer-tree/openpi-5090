# 实验结果台账

更新时间：2026-09-21（CST）

本文件只记录实际运行的配置、指标、结论和证据。详细协议与分析见
[`experiments/README.md`](experiments/README.md)。工程故障不作为实验结果。

## 当前结论

| 问题 | 结论 | 证据 |
|---|---|---|
| 4×5090 能否训练 pi0.5 LoRA | 可以。SFT 与 ACPD-v2 均已稳定运行 physical global BS64。 | H2、H9-scale-b |
| 当前最佳历史 SFT | cosine BS32 50K，四套 pooled success `57.05%`；该模型不是最终 BS64 公平基线。 | 4×500 episodes |
| 原始 ACPD 的 Cue 是否有效 | 没有可靠证据。Full 仅比 ACL-only 高 `0.15` 点，95% CI `[-1.15, 1.40]`。 | H5、H8 |
| 哪层 exact contribution 最可恢复 | layer 10；overall gap `0.3992`，比次优 layer 11 高 `0.0485`。 | H7/H7.1 |
| ACPD-v2 是否优于匹配 BS64 SFT | 5K 时提升 `9.30` 点；30K 时为 `58.00%` 对 `60.45%`，差值 `-2.45` 点，95% CI `[-5.00, 0.00]`。早期优势未保持到 30K。 | H9-scale-b、H12 |
| ACPD-v2 从 5K 继续训练是否有效 | 同一 H9-scale-b 训练在 30K 达到 `58.00%`，比 5K 高 `34.85` 点；匹配 SFT 30K 尚未完成验证。 | 4×500 episodes |
| 显式 contribution 注入是否有效 | 有正向证据。H9 为 `23.15%`，H13 loss-only 为 `20.60%`；差值 `+2.55` 点，配对 95% CI `[+0.40, +4.70]`。 | H13 |
| ACPD-v2 是否通过降低训练 MSE 获益 | 没有该证据。与 SFT 对齐的 299 个监督 loss 点相关系数为 `0.9985`，全程均值几乎相同；5K 成功率增益不能由更低训练 MSE 解释。 | H9/H12 loss 对齐 |

## 正式 BS64 实验

除 teacher 外，最终论文训练统一使用 4 GPU、physical global BS64、无梯度累积。

| ID | 目的 | 实际配置 | Job | 状态 | 结果 |
|---|---|---|---:|---|---|
| Teacher | 提供 agentview+wrist 特权信息 | 全量 SFT，30K | 历史任务 | 完成 | checkpoint `29999` |
| H12 | backview 单视角 baseline | LoRA，BS64，5K 后确定性续训至 30K | 130285/130491/130762/130889/132033 | 30K 完成；25K 验证排队 | 5K pooled `13.85%`；30K pooled `60.45%` |
| H14-top | topview baseline | LoRA，BS64，30K，每 5K 保存 | 130669/130890 | 完成 | 30K pooled `71.55%` |
| H14-left | leftview baseline | LoRA，BS64，30K，每 5K 保存 | 130670/130891 | 完成 | 30K pooled `78.65%` |
| H14-right | rightview baseline | LoRA，BS64，30K，每 5K 保存 | 130671/130892 | 30K 训练完成；验证排队 | 尚无结论 |
| H9-scale-b | backview ACPD-v2 主 student | layer 10，BS64，30K | 129728/130773 | 训练和 30K 验证完成 | 5K pooled `23.15%`；30K pooled `58.00%` |
| H9-mid-trajectory | 检查 30K 前是否已过峰值 | 验证 20K/25K；相同 2,000 episodes | 131861/131860 | 20K 等待 checkpoint；25K 排队 | 尚无结论 |
| H9-trajectory | 定位 30K 后最佳 checkpoint | H9 从 30K 精确续训至 60K；验证 40K/50K/60K | 131642/131643 | 用户取消；训练和验证均未运行 | 尚无结论 |
| H9-recovery | 恢复 H9 中间 checkpoint | 与 H9-scale-b 相同，训练至 20K | 130704 | 运行中 | 尚无新结论 |
| H13 | 判断 contribution 注入是否有效 | H9 去除 residual 注入，BS64，5K | 130599/130774 | 完成 | pooled `20.60%`；H9 高 `2.55` 点，配对 95% CI `[+0.40, +4.70]` |

## 前期筛选结果

下表用于记录方法形成过程，不作为最终 BS64 公平对照。

| ID | 设置 | 关键结果 | 结论 |
|---|---|---|---|
| H1/H2 | ACPD LoRA FSDP4，BS32 | 单卡峰值约 `17.3 GiB` | 4×5090 训练可行 |
| H3 | layer 6、12、6+12，BS32，5K | loss 差异小于 1% | 不支持必须使用 6+12 层 |
| H4 | Flow、Cue、ACL、Full，BS32，2K | supervised loss 差异小于 1% | loss 不能筛选组件 |
| H5 | Full 对 Flow-only，BS32，5K | `6.50%` 对 `4.45%`，差值 `+2.05` 点 | Full 有正向信号 |
| H7/H7.1 | layer 6-12 contribution probe | layer 10 overall gap `0.3992` | 选择 layer 10 |
| H8 | ACL-only，BS32，5K | `6.35%`；Full 仅高 `0.15` 点 | H5 增益主要由 ACL 解释 |
| H9 | ACPD-v2 layer 10，BS32，5K | `11.35%`，相对 H8 `+5.00` 点 | ACPD-v2 有正向信号 |

## 验证结果

每个 benchmark 500 episodes；pooled 为四套共 2,000 episodes。

| 模型 | Spatial | Object | Goal | LIBERO-10 | Pooled |
|---|---:|---:|---:|---:|---:|
| SFT-2GPU BS16 30K | 9.40% | 20.60% | 20.80% | 2.80% | 13.40% |
| SFT-4GPU BS32 30K | 11.20% | 32.40% | 34.20% | 2.40% | 20.05% |
| SFT-cosine BS32 30K | 47.80% | 59.40% | 57.60% | 20.00% | 46.20% |
| SFT-cosine BS32 40K | 60.40% | 56.60% | 58.80% | 21.00% | 49.20% |
| SFT-cosine BS32 50K | 63.20% | 72.20% | 66.80% | 26.00% | 57.05% |
| SFT-cosine BS32 60K | 63.40% | 68.60% | 59.40% | 25.60% | 54.25% |
| Flow-only BS32 5K | 1.80% | 7.60% | 8.40% | 0.00% | 4.45% |
| ACL-only BS32 5K | 2.00% | 15.40% | 7.60% | 0.40% | 6.35% |
| Full ACPD BS32 5K | 2.80% | 10.80% | 12.20% | 0.20% | 6.50% |
| ACPD-v2 BS32 5K | 5.20% | 24.20% | 15.60% | 0.40% | 11.35% |
| SFT backview BS64 5K | 9.40% | 18.40% | 24.80% | 2.80% | 13.85% |
| ACPD-v2 backview BS64 5K | 25.40% | 32.40% | 30.00% | 4.80% | 23.15% |
| ACPD-v2 backview BS64 30K | 64.00% | 73.00% | 61.80% | 33.20% | 58.00% |
| ACPD-v2 loss-only backview BS64 5K | 19.40% | 35.60% | 25.00% | 2.40% | 20.60% |
| SFT backview BS64 30K | 70.00% | 69.80% | 68.20% | 33.80% | 60.45% |
| SFT topview BS64 30K | 81.00% | 80.60% | 79.20% | 45.40% | 71.55% |
| SFT leftview BS64 30K | 84.00% | 79.60% | 79.40% | 71.60% | 78.65% |

## 精选 Checkpoint

精选目录使用文件级硬链接，不复制模型数据。原 checkpoint 本次不移动、不删除。

根目录：`/opt/liutong/openpi_checkpoints/fixed_dataset/curated`

| 分类 | 模型 | 已归档 | 状态 |
|---|---|---|---|
| teacher | agentview+wrist | `29999` | 完整 |
| baseline | backview BS64 | `4999` | 5K--30K checkpoint 完整；30K 验证完成；25K job 132033 排队 |
| baseline | topview BS64 | - | 5K--30K checkpoint 和 30K 验证完整 |
| baseline | leftview BS64 | - | 5K--30K checkpoint 和 30K 验证完整 |
| baseline | rightview BS64 | - | 5K--30K checkpoint 完整；job 130892 验证排队 |
| student | backview ACPD-v2 layer 10 BS64 | `24999` | 30K checkpoint 和验证完整；job 130704 正在恢复 5K--20K；60K 路线已取消 |
| ablation | backview loss-only BS64 | - | 5K checkpoint 和验证完整；之后补至 30K |
| ablation | backview ACL-only BS64 | - | 尚未运行 |

H11 不进入精选 checkpoint 目录。旧 BS16/BS32 checkpoint 暂不删除，也不进入正式索引。

## 证据路径

| 内容 | 路径 |
|---|---|
| 实验目录索引 | `experiments/README.md` |
| H7/H7.1 原始指标 | `/opt/liutong/openpi-5090-research/acpd-exact-attention-probe/results/` |
| H9 训练日志 | `slurm-log/pi05-bv-acpdv2-l10-pbs32-5k_129710.out` |
| H9-scale-b 训练日志 | `slurm-log/pi05-bv-acpdv2-l10-fsdp4-bs64-30k_129728.out` |
| H9/H12 loss 对齐指标 | `experiments/student/acpd-v2-h9-batch-scaling-30k/results/loss_comparison.json` |
| H9/H12 loss 对齐图 | `artifacts/pi05_backview_sft_vs_acpdv2_loss.png` |
| H9/H12 30K 配对分析 | `experiments/baseline/sft-backview-bs64-5k/results/h9_vs_h12_30k_paired_analysis.json` |
| H12 30K 验证 | `/opt/liutong/openpi-5090-evals/sft-backview-bs64-30k/29999/summary.txt` |
| H9-scale-b 30K 验证 | `/opt/liutong/openpi-5090-evals/acpd-v2-training-trajectory/final-hidden/29999/summary.txt` |
| H9 20K--30K 协议 | `experiments/student/acpd-v2-h9-20k-30k-trajectory/protocol.md` |
| H9 30K--60K 协议 | `experiments/student/acpd-v2-h9-trajectory-60k/protocol.md` |
| H12 分析 | `experiments/baseline/sft-backview-bs64-5k/analysis.md` |
| H13 分析 | `experiments/ablation/acpd-v2-h13-injection-ablation/analysis.md` |
| H14 top/left 结果 | `experiments/baseline/sft-multiview-bs64-30k/analysis.md` |
| SFT 验证目录 | `/opt/liutong/openpi-5090-evals/` |
| ACPD 验证目录 | `/opt/liutong/openpi-5090-evals/acpd-*` |
