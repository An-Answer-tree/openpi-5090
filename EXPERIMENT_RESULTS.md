# 实验结果台账

更新时间：2026-09-25（CST）

本文件只记录实际运行的配置、指标、结论和证据。详细协议与分析见
[`experiments/README.md`](experiments/README.md)。工程故障不作为实验结果。

## 当前结论

本文正式 BS64 结果中的 H9 统一称为 **H9-Fixed（ACPD-v2 固定权重）**。
新实验称为 **H17-Decay（ACPD-v2 非零权重衰减）**；两者关系如下。

| 简称 | 实验身份 | Contribution 权重 | ACL 权重 | 区别 |
|---|---|---|---:|---|
| H9-Fixed | 原 H9-scale-b，BS64 对照 | 全程 0.2 | 0.5 | 原始 ACPD-v2 |
| H17-Decay | 从 H9-Fixed recovery 10K 分支 | 10K–15K：0.2→0.05；之后 0.05 | 0.5 | 只改变 contribution 权重，训练至20K |
| H13-LossOnly | BS64 注入消融 | 全程 0.2 | 0.5 | 去除 residual 注入，保留两个蒸馏 loss |

上述三组均为 backview、layer 10；“Fixed/Decay”指 contribution loss 权重，
不是学习率。H9-Fixed 与 H17-Decay 均沿用原 30K cosine LR。
H9-recovery、mid-trajectory、trajectory 分别是 H9-Fixed 的轨迹恢复、验证和续训，
不是新的方法。早期 BS32 的 H9 保留原编号，并明确标注 BS32。

| 问题 | 结论 | 证据 |
|---|---|---|
| 4×5090 能否训练 pi0.5 LoRA | 可以。SFT 与 ACPD-v2 均已稳定运行 physical global BS64。 | H2、H9-scale-b |
| 当前最佳历史 SFT | cosine BS32 50K，四套 pooled success `57.05%`；该模型不是最终 BS64 公平基线。 | 4×500 episodes |
| 原始 ACPD 的 Cue 是否有效 | 没有可靠证据。Full 仅比 ACL-only 高 `0.15` 点，95% CI `[-1.15, 1.40]`。 | H5、H8 |
| 哪层 exact contribution 最可恢复 | layer 10；overall gap `0.3992`，比次优 layer 11 高 `0.0485`。 | H7/H7.1 |
| ACPD-v2 是否优于匹配 BS64 SFT | 5K 时提升 `9.30` 点；30K 时为 `58.00%` 对 `60.45%`，差值 `-2.45` 点，95% CI `[-5.00, 0.00]`。早期优势未保持到 30K。 | H9-scale-b、H12 |
| ACPD-v2 的早期优势在 10K 是否仍存在 | 存在。H9 为 `37.45%`，SFT 为 `23.25%`；差值 `+14.20` 点，配对 95% CI `[+11.90, +16.60]`。 | 相同 2,000 episodes |
| ACPD-v2 的早期优势何时消失 | 25K 时 ACPD-v2 为 `55.75%`，SFT 为 `55.80%`；差值 `-0.05` 点，95% CI `[-2.50, +2.45]`。25K 已无可检测优势。 | 相同 2,000 episodes |
| ACPD-v2 从 5K 继续训练是否有效 | 同一 H9-scale-b 训练在 30K 达到 `58.00%`，比 5K 高 `34.85` 点；匹配 SFT 30K 为 `60.45%`，尚未证明最终优势。 | 4×500 episodes |
| H9 25K 是否早于 30K 达峰 | 不支持。25K 为 `55.75%`，30K 为 `58.00%`；25K-30K 为 `-2.25` 点，95% CI `[-4.70, +0.20]`。 | 相同 2,000 episodes |
| H9 在 30K 后是否继续提高 | 35K 为 `61.05%`，比 30K 高 `3.05` 点，配对 95% CI `[+0.60, +5.50]`。但相对 SFT 30K 仅高 `0.60` 点，95% CI `[-1.90, +3.05]`，尚不能证明最终优于 SFT。 | 相同 2,000 episodes |
| H9 与同进度 SFT 在 35K 的比较 | H9 `61.05%`，SFT `57.15%`；探索性配对差值 `+3.90` 点，95% CI `[+1.45, +6.35]`。SFT 35K 比自身 30K 低 `3.30` 点，不能据此断言 H9 提高了最终上限。 | 相同 2,000 episodes |
| H9 40K 是否继续高于 35K | 40K `59.45%`，35K `61.05%`；探索性配对差值 `-1.60` 点，95% CI `[-4.05, +0.80]`，未检测到确定的下降。 | 相同 2,000 episodes |
| 显式 contribution 注入是否有效 | 有正向证据。H9 为 `23.15%`，H13 loss-only 为 `20.60%`；差值 `+2.55` 点，配对 95% CI `[+0.40, +4.70]`。 | H13 |
| ACPD-v2 是否通过降低训练 MSE 获益 | 没有该证据。与 SFT 对齐的 299 个监督 loss 点相关系数为 `0.9985`，全程均值几乎相同；5K 成功率增益不能由更低训练 MSE 解释。 | H9/H12 loss 对齐 |
| 辅助目标在 30K 前是否自然消失 | 没有。加权 contribution/ACL 从首个到末个窗口下降 `53.42/53.04%`，但在总目标中的窗口占比保持约 `59--61%/16--17%`。 | H9 0--30K loss |
| 后期是否出现明显辅助梯度冲突 | 不支持。正式 BS32 中组合冲突率在 5K/30K 均为 `0%`，cosine 中位数为 `0.6279/0.5920`。 | H15a，200 个成对 batches |
| Contribution 后期梯度是什么方向 | 30K 加权梯度/flow范数比中位数`1.175`，cosine中位数`0.0957`，接近正交；是否限制任务成功率尚无结论。 | H15a；工作假设见`findings.md` |

## 正式 BS64 实验

除 teacher 外，最终论文训练统一使用 4 GPU、physical global BS64、无梯度累积。

| ID | 目的 | 实际配置 | Job | 状态 | 结果 |
|---|---|---|---:|---|---|
| Teacher | 提供 agentview+wrist 特权信息 | 全量 SFT，30K | 历史任务 | 完成 | checkpoint `29999` |
| H12 | backview 单视角 baseline | LoRA，BS64，5K 后确定性续训至 60K | 130285/130491/130762/132791；35K验证134992/134995 | 60K训练完成；40K/50K/60K验证排队 | 5K `13.85%`；20K `47.20%`；25K `55.80%`；30K `60.45%`；35K `57.15%` |
| H14-top | topview baseline | LoRA，BS64，30K，每 5K 保存 | 130669/130890 | 完成 | 30K pooled `71.55%` |
| H14-left | leftview baseline | LoRA，BS64，30K，每 5K 保存 | 130670/130891 | 完成 | 30K pooled `78.65%` |
| H14-right | rightview baseline | LoRA，BS64，30K，每 5K 保存 | 130671/130892 | 完成 | 30K pooled `77.00%` |
| H9-Fixed | 固定权重 ACPD-v2 对照 | layer 10，BS64，contribution=0.2，ACL=0.5 | 129728/130773 | 30K完成；恢复与续训见下表 | 5K `23.15%`；10K `37.45%`；25K `55.75%`；30K `58.00%`；35K `61.05%` |
| H13-LossOnly | 判断 contribution 注入是否有效 | H9-Fixed 去除 residual 注入，BS64，5K | 130599/130774 | 完成 | pooled `20.60%`；H9-Fixed 高 `2.55` 点，配对 95% CI `[+0.40, +4.70]` |
| ACL-only BS64 | 检验 contribution 学习和注入在 ACL 之外的增益 | backview，layer 10，BS64，flow=1.0、ACL=0.5、contribution=0、关闭注入；训练至35K | 135724；30K/35K验证135725/135726；分析135729/135728 | 训练排队；两点全量验证等待依赖 | 尚无结论 |
| H17-Decay | 检验后期 contribution 权重是否过强 | H9-Fixed 10K完整状态→20K；BS64；10K–15K权重0.2→0.05，ACL=0.5 | 134422/134423/134424 | 训练运行中；20K全量验证与配对分析等待依赖 | 尚无结论 |

### H9-Fixed 轨迹任务

| 阶段（原标签） | 作用 | Job | 状态/已得结论 |
|---|---|---|---|
| 恢复（H9-recovery） | 重建被清理的早期 checkpoint，提供 H17 的同源10K起点与20K对照 | 130704/132198/132398 | 10K、15K、20K checkpoint 完整 |
| 中期验证（H9-mid-trajectory） | 测量10K/15K/20K/25K优势变化 | 132731/132732/134008/134009/132082/132084 | 10K相对SFT +14.20点；15K +5.70点；25K -0.05点；20K验证运行中 |
| 延长训练（H9-trajectory） | 原计划30K→60K；45K checkpoint完整后停训，验证35K/40K | 132390/132720/132724/134371/134372；取消135001/135002 | 35K为61.05%；40K为59.45%；45K验证已取消，尚无结论；50K/60K未训练 |

H17-Decay 的主比较为同源 **H9-Fixed recovery 20K**，次比较为 SFT BS64 20K。
H17 尚无验证结果，现有 H9 数值不属于 H17。

## 当前机制实验

| ID | 目的 | 实际配置 | Job | 状态 | 结论 |
|---|---|---|---:|---|---|
| H15a | 判断 ACPD-v2 后期是否存在辅助梯度干扰 | H9 5K/30K；每点200个相同BS32 batch；不更新参数 | smoke 132308；快速 132311；正式 132250 | 完成 | 5K/30K 组合冲突率均为 `0%`；不支持后期梯度冲突解释 |

## 探索性快速轨迹

每个 checkpoint 使用四套共 400 episodes，仅用于选择正式验证点，不作为论文最终数值。

| 模型 | Step | Spatial | Object | Goal | LIBERO-10 | Pooled | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| SFT backview BS64 | 10K | 21.00% | 30.00% | 28.00% | 3.00% | 20.50% | 完成 |
| SFT backview BS64 | 15K | 34.00% | 58.00% | 53.00% | 12.00% | 39.25% | 完成 |

H9-Fixed 的快速验证在运行前升级为正式2,000回合，因此没有400回合结果。

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
| H9 前期 ACPD-v2 BS32 5K | 5.20% | 24.20% | 15.60% | 0.40% | 11.35% |
| SFT backview BS64 5K | 9.40% | 18.40% | 24.80% | 2.80% | 13.85% |
| H9-Fixed backview BS64 5K | 25.40% | 32.40% | 30.00% | 4.80% | 23.15% |
| SFT backview BS64 10K | 23.40% | 37.80% | 29.20% | 2.60% | 23.25% |
| H9-Fixed backview BS64 10K | 41.60% | 50.60% | 47.60% | 10.00% | 37.45% |
| SFT backview BS64 15K | 43.60% | 58.00% | 47.40% | 14.60% | 40.90% |
| H9-Fixed backview BS64 15K | 49.40% | 64.40% | 53.60% | 19.00% | 46.60% |
| H9-Fixed backview BS64 25K | 63.20% | 69.40% | 62.00% | 28.40% | 55.75% |
| H9-Fixed backview BS64 30K | 64.00% | 73.00% | 61.80% | 33.20% | 58.00% |
| H9-Fixed backview BS64 35K | 73.20% | 73.80% | 63.60% | 33.60% | 61.05% |
| H9-Fixed backview BS64 40K | 68.20% | 76.00% | 60.00% | 33.60% | 59.45% |
| H13-LossOnly backview BS64 5K | 19.40% | 35.60% | 25.00% | 2.40% | 20.60% |
| SFT backview BS64 20K | 50.60% | 62.60% | 54.40% | 21.20% | 47.20% |
| SFT backview BS64 25K | 64.00% | 64.00% | 65.60% | 29.60% | 55.80% |
| SFT backview BS64 30K | 70.00% | 69.80% | 68.20% | 33.80% | 60.45% |
| SFT backview BS64 35K | 65.80% | 68.40% | 64.60% | 29.80% | 57.15% |
| SFT topview BS64 30K | 81.00% | 80.60% | 79.20% | 45.40% | 71.55% |
| SFT leftview BS64 30K | 84.00% | 79.60% | 79.40% | 71.60% | 78.65% |
| SFT rightview BS64 30K | 85.00% | 84.80% | 84.00% | 54.20% | 77.00% |

## 精选 Checkpoint

精选目录使用文件级硬链接，不复制模型数据。原 checkpoint 本次不移动、不删除。

根目录：`/opt/liutong/openpi_checkpoints/fixed_dataset/curated`

| 分类 | 模型 | 已归档 | 状态 |
|---|---|---|---|
| teacher | agentview+wrist | `29999` | 完整 |
| baseline | backview BS64 | `4999` | 5K--60K checkpoint 完整；20K、25K、30K、35K 验证完成，40K/50K/60K 待验证 |
| baseline | topview BS64 | - | 5K--30K checkpoint 和 30K 验证完整 |
| baseline | leftview BS64 | - | 5K--30K checkpoint 和 30K 验证完整 |
| baseline | rightview BS64 | - | 5K--30K checkpoint 和 30K 验证完整 |
| student | backview ACPD-v2 layer 10 BS64 | `24999` | 恢复轨迹至20K、延长轨迹至45K的 checkpoint 完整；40K验证完成，45K验证已取消 |
| ablation | backview loss-only BS64 | - | 5K checkpoint 和验证完整；之后补至 30K |
| ablation | backview ACL-only BS64 | - | 35K训练排队；30K/35K验证等待依赖；尚无结论 |

H11 不进入精选 checkpoint 目录。旧 BS16/BS32 checkpoint 暂不删除，也不进入正式索引。

## 证据路径

| 内容 | 路径 |
|---|---|
| 实验目录索引 | `experiments/README.md` |
| H7/H7.1 原始指标 | `/opt/liutong/openpi-5090-research/acpd-exact-attention-probe/results/` |
| H9 训练日志 | `slurm-log/pi05-bv-acpdv2-l10-pbs32-5k_129710.out` |
| H9-scale-b 训练日志 | `slurm-log/pi05-bv-acpdv2-l10-fsdp4-bs64-30k_129728.out` |
| H12 0--5K 训练日志 | `slurm-log/pi05-bv-sft-bs64-5k_130285.out` |
| H12 5K--30K 训练日志 | `slurm-log/pi05-bv-sft-bs64-r30k_130762.out` |
| H12 20K 验证 | `/opt/liutong/openpi-5090-evals/sft-backview-bs64-20k/20000/summary.txt` |
| H9早期轨迹快速验证 | `/opt/liutong/openpi-5090-evals/h9-early-trajectory-quick/` |
| H9/SFT 10K 正式验证 | `/opt/liutong/openpi-5090-evals/h9-early-trajectory-full/` |
| H9 15K 正式验证 | `/opt/liutong/openpi-5090-evals/h9-early-trajectory-full/14999/summary.txt` |
| SFT 35K 验证 | `/opt/liutong/openpi-5090-evals/sft-backview-bs64-training-trajectory/35000/summary.txt`；job 134992/134995 |
| SFT 35K/30K 配对分析 | `experiments/baseline/sft-backview-bs64-trajectory-60k/results/sft_35k_vs_30k_paired_analysis.json` |
| H9/SFT 35K 配对分析 | `experiments/student/acpd-v2-h9-trajectory-60k/results/h9_35k_vs_sft_35k_paired_analysis.json` |
| H9 40K/35K 配对分析 | `experiments/student/acpd-v2-h9-trajectory-60k/results/h9_40k_vs_h9_35k_paired_analysis.json` |
| H9/SFT 10K 配对分析 | `experiments/student/acpd-v2-h9-early-trajectory/results/h9_vs_sft_10k_paired_analysis.json` |
| H9/H12 loss 对齐指标 | `experiments/student/acpd-v2-h9-batch-scaling-30k/results/loss_comparison.json` |
| H9/H12 loss 对齐图 | `artifacts/pi05_backview_sft_vs_acpdv2_loss.png` |
| H9/H12 30K 配对分析 | `experiments/baseline/sft-backview-bs64-5k/results/h9_vs_h12_30k_paired_analysis.json` |
| H9/H12 25K 配对分析 | `experiments/baseline/sft-backview-bs64-5k/results/h9_vs_h12_25k_paired_analysis.json` |
| H12 30K 验证 | `/opt/liutong/openpi-5090-evals/sft-backview-bs64-30k/29999/summary.txt` |
| H9-scale-b 30K 验证 | `/opt/liutong/openpi-5090-evals/acpd-v2-training-trajectory/final-hidden/29999/summary.txt` |
| H9-scale-b 25K 验证 | `/opt/liutong/openpi-5090-evals/slurm-log/sum-bv-h9-25k_132084.out` |
| H9 25K/30K 成对分析 | `experiments/student/acpd-v2-h9-20k-30k-trajectory/results/h9_25k_vs_30k_paired_analysis.json` |
| H9 25K 协议 | `experiments/student/acpd-v2-h9-20k-30k-trajectory/protocol.md` |
| H9 30K--60K 协议 | `experiments/student/acpd-v2-h9-trajectory-60k/protocol.md` |
| H9 35K 验证 | `/opt/liutong/openpi-5090-evals/acpd-v2-training-trajectory/final-hidden/34999/summary.txt` |
| H9 40K 验证 | `/opt/liutong/openpi-5090-evals/acpd-v2-training-trajectory/final-hidden/39999/summary.txt` |
| H9 45K checkpoint（验证已取消） | `/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/batch_scaling_30k/pi05_libero_backview_acpd_v2_layer10/pi05_libero_backview_acpd_v2_lora_fsdp4_bs64_30k/44999` |
| H9 35K/30K 配对分析 | `experiments/student/acpd-v2-h9-trajectory-60k/results/h9_35k_vs_h9_30k_paired_analysis.json` |
| H9 35K/SFT 30K 配对分析 | `experiments/student/acpd-v2-h9-trajectory-60k/results/h9_35k_vs_sft_30k_paired_analysis.json` |
| H15a 梯度诊断协议 | `experiments/mechanism/acpd-v2-h15a-gradient-conflict/protocol.md` |
| H15a 快速诊断协议 | `experiments/mechanism/acpd-v2-h15a-gradient-conflict/fast_protocol.md` |
| H15a 快速诊断分析 | `experiments/mechanism/acpd-v2-h15a-gradient-conflict/analysis.md` |
| H15a 结果目录 | `/opt/liutong/openpi-5090-research/acpd-v2-gradient-conflict/` |
| ACPD-v2 loss问题定位 | `experiments/student/acpd-v2-h9-batch-scaling-30k/analysis.md` |
| H12 分析 | `experiments/baseline/sft-backview-bs64-5k/analysis.md` |
| H13 分析 | `experiments/ablation/acpd-v2-h13-injection-ablation/analysis.md` |
| H17 协议与任务记录 | `experiments/ablation/acpd-v2-h17-contribution-decay/` 内 `protocol.md`、`execution.md` |
| H17 checkpoint | `/opt/liutong/openpi_checkpoints/fixed_dataset/ablation/acpd_v2_h17_contribution_decay/pi05_libero_backview_acpd_v2_layer10/pi05_libero_backview_acpd_v2_h17_decay_bs64_20k` |
| H17 验证与配对分析 | `/opt/liutong/openpi-5090-evals/acpd-v2-h17-contribution-decay/19999/` |
| ACL-only BS64 协议 | `experiments/ablation/acpd-v2-acl-only-bs64-35k/protocol.md` |
| ACL-only BS64 checkpoint（待生成） | `/opt/liutong/openpi_checkpoints/fixed_dataset/ablation/acpd_v2_acl_only_bs64_35k/pi05_libero_backview_acl_only_bs64_35k/pi05_libero_backview_acl_only_lora_fsdp4_bs64_35k/` |
| ACL-only BS64 验证（待运行） | `/opt/liutong/openpi-5090-evals/acpd-v2-acl-only-bs64-35k/29999/`、`34999/` |
| H14 多视角结果 | `experiments/baseline/sft-multiview-bs64-30k/analysis.md` |
| SFT 验证目录 | `/opt/liutong/openpi-5090-evals/` |
| ACPD 验证目录 | `/opt/liutong/openpi-5090-evals/acpd-*` |
