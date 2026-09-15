# 实验结果记录

更新时间：2026-09-15

本文件是唯一长期实验结果台账。开始实验相关工作前读取；实验状态变化或产生最终结果后立即更新。只记录实际运行的配置和已核实结果；未完成项不填写结论，原始日志和 checkpoint 保存在 `/opt/liutong`。

## 当前结论

| 问题 | 结论 |
|---|---|
| 5090 能否训练 pi0.5 LoRA | 可以。FSDP4 和 FSDP2 均已完成训练。 |
| 当前最佳 SFT checkpoint | cosine 50K，四套 pooled success `57.05%`。 |
| 继续训练到 60K 是否更好 | 否。pooled success 从 `57.05%` 降至 `54.25%`。 |
| ACPD 是否需要 6+12 层 | 不需要。layer 6 的 5K loss 最低，但层间差异均小于 1%。 |
| Cue 或 ACL 是否加快收敛 | 没有可靠证据。2K 差异均未达到预设 1% 阈值。 |
| Full ACPD 是否优于 Flow-only | 是。5K pooled success 为 `6.50%` 对 `4.45%`，提升 `2.05` 个百分点，配对 95% CI 为 `[0.90%, 3.25%]`。 |

## ACPD 实验总表

| ID | 目的或比较 | 关键配置 | 状态 | 已核实结论 |
|---|---|---|---|---|
| H1 | 梯度累积能否在 4×5090 上运行 ACPD LoRA | FSDP4，global micro BS8，accumulation 4，effective BS32，2 steps | 支持 | 训练正常，单卡峰值 `17,291 MiB`。 |
| H2 | 物理 global BS32 能否直接运行 | FSDP4，global BS32，accumulation 1，2 steps | 支持 | 原始 ACPD 训练图正常，单卡峰值 `17,337 MiB`。 |
| H3 | layer 6+12 是否优于单层 | FSDP2，global BS32，5K，seed 42 | 不支持 | layer 6 的训练 loss 最低；三组差异小于 1%，不能推断成功率。 |
| H4 | Cue 或 ACL 是否改善早期收敛 | Flow、Cue、ACL、Full；FSDP2，global BS32，2K | 不支持 | 相对 Flow 的差异均未达到预设 1% 阈值。 |
| H5 | Full ACPD 是否优于 Flow-only | 相同 5K 预算；四套共 2,000 episodes | 支持 | `6.50%` 对 `4.45%`，提升 `2.05` 点，配对 95% CI `[0.90, 3.25]` 点。 |
| H6 | 启发式 visual-message 是否可恢复 | 原计划 500-step probe | 废弃 | target 近似全局视觉平均，shuffle 对照无效；任务在首批数据前取消，无实验结果。 |
| H6.1 | 在无效 proxy 上比较 layer 6/9/12 | 原计划三层短实验 | 废弃 | 父实验设计无效，未运行，不产生层选择结论。 |
| H7 | 精确 attention contribution 是否可恢复，并选择层 | 256 个 episode-held-out 样本，64 个 hard 样本，3 个 probe seeds | 支持 | layer 9 和 12 可恢复；layer 9 按预注册规则胜出，layer 6 不可用。 |
| H8 | ACL-only 能否解释 H5 提升 | FSDP2，global BS32，5K，seed 42；2,000 episodes | 训练完成，验证排队 | Job 128417 已保存 4999 checkpoint；尚无验证结果。 |
| H9 | 部署式 ACPD-v2 是否优于 H8 | layer 9 exact contribution；FSDP4，micro BS8×accumulation 4，5K；2,000 episodes | 排队 | Job 128513 等待资源；尚无训练和验证结果。 |

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

| 方法 | 100–4.9K loss | 4K–4.9K loss | Spatial | Object | Goal | LIBERO-10 | Pooled |
|---|---:|---:|---:|---:|---:|---:|---:|
| Flow-only | 0.040767 | 0.032500 | 1.80% | 7.60% | 8.40% | 0.00% | 4.45%（89/2000） |
| Full ACPD 6+12 | 0.040682 | 0.032370 | 2.80% | 10.80% | 12.20% | 0.20% | 6.50%（130/2000） |

结论：H5 通过。Full ACPD 提升 `2.05` 个百分点，达到预设 2 点门槛；按任务分层的配对 bootstrap 95% CI 为 `[0.90%, 3.25%]`，下界大于 0。该结果只证明单 seed、5K 筛选有效，不替代重复 seed 和完整训练。

## H6/H7：特权视觉信息可恢复性

| 实验 | 状态 | 结论 |
|---|---|---|
| H6 启发式 visual-message probe | 废弃 | target 近似全局视觉平均，shuffle 对照也不正确，未产生有效结果。 |
| H6.1 启发式 layer 6/9/12 scan | 废弃 | H6 proxy 无效，因此未运行三层比较，也没有层选择结果。 |
| H7 精确 attention contribution probe | 完成 | layer 9 和 12 通过；layer 9 相对 layer 12 的 overall gap 优势为 `0.1050`，超过预注册 `0.02` 门槛，按协议选择 layer 9。 |

H7 使用 teacher 的真实 Q/K/V、完整 attention softmax、action expert 输出投影和 AdaRMS residual gate。对照只打乱同一 query 对应的视觉 K/V。

| Layer | Overall gap（95% CI） | Hard gap（95% CI） | Explained variance | 决策 |
|---:|---:|---:|---:|---|
| 6 | `0.0997`（`[0.0846, 0.1091]`） | `0.0706`（`[0.0497, 0.0884]`） | `-0.1468` | 不可用 |
| **9** | **`0.3417`（`[0.3095, 0.3756]`）** | **`0.3194`（`[0.2717, 0.3875]`）** | **`0.2822`** | **通过并选中** |
| 12 | `0.2367`（`[0.2127, 0.2546]`） | `0.2080`（`[0.1692, 0.2349]`） | `0.0903` | 通过 |

## H8/H9：ACPD-v2 筛选

| 实验 | 配置 | Job | 状态 |
|---|---|---:|---|
| H8 ACL-only | FSDP2，global BS32，5K，seed 42 | 128417 | 完成；step 4900 supervised loss `0.0316`，4999 checkpoint 已完整保存 |
| H8 四套验证 | 4 GPU，2,000 episodes，依赖 H8 | 128421 | 训练依赖已满足，等待资源 |
| H9 ACPD-v2 | layer 9 exact contribution，FSDP4，micro BS8 × accumulation4，effective BS32，5K | 128513 | 等待资源 |
| H9 四套验证 | 4 GPU，2,000 episodes，依赖 H9 | 128514 | 等待依赖 |

H9 不包含重复 seed 或其他学生视角。通过标准为 pooled success 比 H8 高至少 `1.5` 个百分点，且配对 bootstrap 95% CI 下界大于 0。

## 证据

| 内容 | 位置 |
|---|---|
| SFT 训练日志 | `slurm-log/pi05-bv-lora*.out` |
| SFT 验证 summary | `/opt/liutong/openpi-5090-evals/*/*/summary.txt` |
| H3 分析 | `experiments/acpd-layer-ablation-5k/analysis.md` |
| H4 分析 | `experiments/acpd-component-ablation-2k/analysis.md` |
| H5 协议 | `experiments/acpd-task-success-5k/protocol.md` |
| H5 分析 | `experiments/acpd-task-success-5k/analysis.md` |
| H7 协议 | `experiments/acpd-exact-attention-probe/protocol.md` |
| H7 分析 | `experiments/acpd-exact-attention-probe/analysis.md` |
| H8 协议 | `experiments/acpd-acl-only-5k/protocol.md` |
| H9 协议 | `experiments/acpd-v2-5k/protocol.md` |

## Checkpoint 路径检索

| 模型 | Checkpoint 路径 |
|---|---|
| pi0.5 base | `/home/liutong/.cache/openpi-5090/pi05_base/params` |
| Teacher agentview+wrist 30K | `/opt/liutong/openpi_checkpoints/fixed_dataset/sft/agentview_wrist/29999` |
| SFT-2GPU 30K | `/opt/liutong/openpi-5090-checkpoints/pi05_libero_backview_lora/pi05_libero_backview_lora_fsdp2_bs16_30k/29999` |
| SFT-4GPU 30K | `/opt/liutong/openpi-5090-checkpoints/pi05_libero_backview_lora/pi05_libero_backview_lora_fsdp4_bs32_30k/29999` |
| SFT-cosine 30K | `/opt/liutong/openpi-5090-checkpoints/pi05_libero_backview_lora/pi05_libero_backview_lora_fsdp4_bs32_cosine_30k/29999` |
| SFT-cosine 40K | `/opt/liutong/openpi-5090-checkpoints/pi05_libero_backview_lora/pi05_libero_backview_lora_fsdp4_bs32_cosine_30k/40000` |
| SFT-cosine 50K | `/opt/liutong/openpi-5090-checkpoints/pi05_libero_backview_lora/pi05_libero_backview_lora_fsdp4_bs32_cosine_30k/50000` |
| SFT-cosine 60K | `/opt/liutong/openpi-5090-checkpoints/pi05_libero_backview_lora/pi05_libero_backview_lora_fsdp4_bs32_cosine_30k/59999` |
| ACPD layer 6，5K | `/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_lora/ablations/layers_5k/pi05_libero_backview_acpd_lora_layer6/pi05_libero_backview_acpd_lora_fsdp2_layer6_bs32_5k/4999` |
| ACPD layer 12，5K | `/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_lora/ablations/layers_5k/pi05_libero_backview_acpd_lora_layer12/pi05_libero_backview_acpd_lora_fsdp2_layer12_bs32_5k/4999` |
| Full ACPD 6+12，5K | `/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_lora/ablations/layers_5k/pi05_libero_backview_acpd_lora_layers6_12/pi05_libero_backview_acpd_lora_fsdp2_layers6_12_bs32_5k/4999` |
| Flow-only，5K | `/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_lora/ablations/task_success_5k/pi05_libero_backview_flow_only_5k/pi05_libero_backview_lora_fsdp2_bs32_5k/4999` |
| H8 ACL-only，5K | `/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_lora/ablations/task_success_5k/pi05_libero_backview_acl_only_5k/pi05_libero_backview_acpd_lora_fsdp2_bs32_5k/4999` |
| H9 ACPD-v2 layer 9，5K | `/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/task_success_5k/pi05_libero_backview_acpd_v2_layer9/pi05_libero_backview_acpd_v2_lora_fsdp4_mbs8_acc4_bs32_5k/4999` |
| H4 组件消融，2K | 按协议不保存 checkpoint |

## 验证结果路径检索

每个目录的 `summary.txt` 是四套汇总；`logs/` 和 benchmark 子目录保存详细日志与视频。

| 模型 | 验证结果目录 | 状态 |
|---|---|---|
| SFT-2GPU 30K | `/opt/liutong/openpi-5090-evals/pi05_libero_backview_lora_fsdp2_bs16_30k/29999` | 完成 |
| SFT-4GPU 30K | `/opt/liutong/openpi-5090-evals/pi05_libero_backview_lora_fsdp4_bs32_30k/29999` | 完成 |
| SFT-cosine 30K | `/opt/liutong/openpi-5090-evals/pi05_libero_backview_lora_fsdp4_bs32_cosine_30k/29999` | 完成 |
| SFT-cosine 40K | `/opt/liutong/openpi-5090-evals/pi05_libero_backview_lora_fsdp4_bs32_cosine_30k/40000` | 完成 |
| SFT-cosine 50K | `/opt/liutong/openpi-5090-evals/pi05_libero_backview_lora_fsdp4_bs32_cosine_30k/50000` | 完成 |
| SFT-cosine 60K | `/opt/liutong/openpi-5090-evals/pi05_libero_backview_lora_fsdp4_bs32_cosine_30k/59999` | 完成 |
| Full ACPD 6+12，5K | `/opt/liutong/openpi-5090-evals/acpd-task-success-5k/full-acpd-layers6-12/4999` | 完成，`6.50%` pooled |
| Flow-only，5K | `/opt/liutong/openpi-5090-evals/acpd-task-success-5k/flow-only/4999` | 完成，`4.45%` pooled |
| H8 ACL-only，5K | `/opt/liutong/openpi-5090-evals/acpd-task-success-5k/acl-only/4999` | 等待验证资源 |
| H9 ACPD-v2 layer 9，5K | `/opt/liutong/openpi-5090-evals/acpd-v2-task-success-5k/layer9-exact-contribution/4999` | 等待训练 |
