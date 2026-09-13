# 实验结果记录

更新时间：2026-09-13

本文件是实验结果的长期中文索引。开始实验相关工作前读取；实验完成后更新。只记录已核实结果，原始日志和 checkpoint 保存在 `/opt/liutong`。

## 当前结论

| 问题 | 结论 |
|---|---|
| 5090 能否训练 pi0.5 LoRA | 可以。FSDP4 和 FSDP2 均已完成训练。 |
| 当前最佳 SFT checkpoint | cosine 50K，四套 pooled success `57.05%`。 |
| 继续训练到 60K 是否更好 | 否。pooled success 从 `57.05%` 降至 `54.25%`。 |
| ACPD 是否需要 6+12 层 | 不需要。layer 6 的 5K loss 最低，但层间差异均小于 1%。 |
| Cue 或 ACL 是否加快收敛 | 没有可靠证据。2K 差异均未达到预设 1% 阈值。 |
| Full ACPD 是否优于 Flow-only | 5K loss 基本相同；任务成功率尚未验证。 |

## SFT 训练

数据：`libero_multiview_tuned_6view_lerobot`；学生视角：backview；LoRA：PaliGemma rank 16、action expert rank 32。

| ID | GPU / global BS | 学习率 | 步数 | 最后记录 loss | Job | 状态 |
|---|---:|---|---:|---:|---:|---|
| SFT-2GPU | 2 / 16 | 默认 | 30K | 0.0294 | 125793 | 完成 |
| SFT-4GPU | 4 / 32 | 默认 | 30K | 0.0261 | 125794 | 完成 |
| SFT-cosine-30K | 4 / 32 | 1K warmup，`2.5e-5`→`2.5e-6` | 30K | 0.0222 | 126023 | 完成 |
| SFT-cosine-60K | 4 / 32 | 30K 后保持 `2.5e-6` | 60K | 0.0197 | 126409 | 完成 |

## SFT 验证

每个 benchmark 500 episodes；pooled 为四套共 2,000 episodes。

| 模型 | Spatial | Object | Goal | LIBERO-10 | Pooled |
|---|---:|---:|---:|---:|---:|
| SFT-2GPU 30K | 9.40% | 20.60% | 20.80% | 2.80% | 13.40% |
| SFT-4GPU 30K | 11.20% | 32.40% | 34.20% | 2.40% | 20.05% |
| SFT-cosine 30K | 47.80% | 59.40% | 57.60% | 20.00% | 46.20% |
| SFT-cosine 40K | 60.40% | 56.60% | 58.80% | 21.00% | 49.20% |
| **SFT-cosine 50K** | **63.20%** | **72.20%** | **66.80%** | **26.00%** | **57.05%** |
| SFT-cosine 60K | 63.40% | 68.60% | 59.40% | 25.60% | 54.25% |

结论：cosine 明显优于默认学习率；50K 是当前最佳停止点。60K 的训练 loss 更低，但任务成功率下降。

## H3：ACPD 层消融

共同配置：teacher=`agentview+wrist/29999`，student=backview，FSDP2，global BS32，5K，seed 42。

| ACPD 层 | 100–4.9K loss | 4K–4.9K loss | step 4.9K loss | Job |
|---|---:|---:|---:|---:|
| **6** | **0.040565** | **0.032250** | **0.0314** | 127589 |
| 12 | 0.040882 | 0.032570 | 0.0317 | 127248 |
| 6+12 | 0.040682 | 0.032370 | 0.0317 | 127143 |

结论：H3 不成立。layer 6 最简单且 loss 最低；单 seed 差异不足 1%，不能推断任务成功率。

## H4：ACPD 组件消融

共同配置：FSDP2，global BS32，2K。正 delta 表示比 Flow-only 更差。

| 方法 | 100–1.9K loss | Delta | 1K–1.9K loss | Delta |
|---|---:|---:|---:|---:|
| Flow-only | 0.050495 | 0.00% | 0.042920 | 0.00% |
| Cue-only | 0.050900 | +0.80% | 0.043340 | +0.98% |
| ACL-only | 0.050337 | -0.31% | 0.042650 | -0.63% |
| Full ACPD | 0.050553 | +0.11% | 0.042990 | +0.16% |

结论：H4 在预设 1% 分辨率下不成立。teacher 在约 99.9% 样本上优于 student；Cue 和 ACL 均产生梯度，但没有显著改善 supervised loss。

## H5：Full ACPD 对照

| 方法 | 100–4.9K loss | 4K–4.9K loss | Checkpoint | Benchmark |
|---|---:|---:|---|---|
| Flow-only | 0.040767 | 0.032500 | step 4999 | 待验证 |
| Full ACPD 6+12 | 0.040682 | 0.032370 | step 4999 | 待验证 |

结论：Full ACPD 仅低 `0.21%/0.40%`，训练 loss 无法区分。下一步只比较两者四套 pooled success；Full ACPD 至少提升 2 个百分点才继续组件实验。

## 证据

| 内容 | 位置 |
|---|---|
| SFT 训练日志 | `slurm-log/pi05-bv-lora*.out` |
| SFT 验证 summary | `/opt/liutong/openpi-5090-evals/*/*/summary.txt` |
| H3 分析 | `experiments/acpd-layer-ablation-5k/analysis.md` |
| H4 分析 | `experiments/acpd-component-ablation-2k/analysis.md` |
| H5 协议 | `experiments/acpd-task-success-5k/protocol.md` |
