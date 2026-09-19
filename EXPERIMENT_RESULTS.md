# 实验结果记录

更新时间：2026-09-19（11:25 CST）

本文件是唯一长期实验结果台账。开始实验相关工作前读取；实验状态变化或产生最终结果后立即更新。只记录实际运行的配置和已核实结果；未完成项标记“尚无结论”，原始日志和 checkpoint 保存在 `/opt/liutong`。

## 记录规则

| 内容 | 要求 |
|---|---|
| 实验目的 | 说明实验要回答的单一问题。 |
| 实验做法 | 只写实际运行的关键变量、对照和资源。 |
| 实验结论 | 只依据已核实指标；未完成或未验证时写“尚无结论”。 |
| 证据 | 详细协议和分析放在 `experiments/`；台账只保留日志、checkpoint 和验证结果路径。 |
| 工程故障 | 不写入本表，不作为实验结果；仅保留当前有效任务状态。 |

## 当前结论

| 问题 | 结论 |
|---|---|
| 5090 能否训练 pi0.5 LoRA | 可以。FSDP4 和 FSDP2 均已完成训练。 |
| 当前最佳 SFT checkpoint | cosine 50K，四套 pooled success `57.05%`。 |
| 继续训练到 60K 是否更好 | 否。pooled success 从 `57.05%` 降至 `54.25%`。 |
| ACPD 是否需要 6+12 层 | 不需要。layer 6 的 5K loss 最低，但层间差异均小于 1%。 |
| Cue 或 ACL 是否加快收敛 | 没有可靠证据。2K 差异均未达到预设 1% 阈值。 |
| Full ACPD 是否优于 Flow-only | 是。5K pooled success 为 `6.50%` 对 `4.45%`，提升 `2.05` 个百分点，配对 95% CI 为 `[0.90%, 3.25%]`。 |
| H5 的 5K 增益由什么解释 | ACL-only 为 `6.35%`，相对 Flow-only 提升 `1.90` 点；Full 仅比 ACL-only 高 `0.15` 点，95% CI 为 `[-1.15, 1.40]`。按预注册规则，该增益由 ACL 解释。 |
| 哪层 exact contribution 最可恢复 | layer 10。其 overall gap 为 `0.3992`，比次优 layer 11 高 `0.0485`，超过预注册 `0.02` 选择门槛。 |
| layer-10 ACPD-v2 是否提高 5K 成功率 | 是。H9 为 `11.35%`，H8 ACL-only 为 `6.35%`；差值 `+5.00` 点，配对 95% CI `[+3.60, +6.45]`。 |

## ACPD 实验总表

| ID | 实验目的 | 实验做法 | 状态 | 实验结论 |
|---|---|---|---|---|
| H1 | 梯度累积能否在 4×5090 上运行 ACPD LoRA | FSDP4，global micro BS8，accumulation 4，effective BS32，2 steps | 完成 | 支持。训练正常，单卡峰值 `17,291 MiB`。 |
| H2 | 物理 global BS32 能否直接运行 | FSDP4，global BS32，accumulation 1，2 steps | 完成 | 支持。原始 ACPD 训练图正常，单卡峰值 `17,337 MiB`。 |
| H3 | layer 6+12 是否优于单层 | FSDP2，global BS32，5K，seed 42 | 完成 | 不支持。layer 6 的训练 loss 最低；三组差异小于 1%，不能推断成功率。 |
| H4 | Cue 或 ACL 是否改善早期收敛 | Flow、Cue、ACL、Full；FSDP2，global BS32，2K | 完成 | 不支持。相对 Flow 的差异均未达到预设 1% 阈值。 |
| H5 | Full ACPD 是否优于 Flow-only | 相同 5K 预算；四套共 2,000 episodes | 完成 | 支持。`6.50%` 对 `4.45%`，提升 `2.05` 点，配对 95% CI `[0.90, 3.25]` 点。 |
| H7/H7.1 | 精确 attention contribution 是否可恢复并选层 | layer 6--12；256 个 held-out 样本；3 个 probe seeds | 完成 | 支持可恢复性；layer 10 的 overall gap 最高，为 `0.3992`。 |
| H8 | ACL-only 能否解释 H5 提升 | FSDP2，global BS32，5K，seed 42；2,000 episodes | 完成 | 支持。ACL-only 为 `6.35%`，与 Full 的 `6.50%` 相差 `0.15` 点，配对 95% CI `[-1.15, 1.40]`。 |
| H9 | 部署式 ACPD-v2 是否优于 H8 | layer 10 exact contribution；FSDP4，physical global BS32，无梯度累积；5K | 完成 | 支持。`11.35%` 对 `6.35%`，提升 `5.00` 点，配对 95% CI `[+3.60, +6.45]`；gate 末步为 `0.0048`，直接注入机制仍需 H11 对照。 |
| H9-scale-b | 4 卡 BS64 能否提高 ACPD-v2 吞吐 | layer 10；FSDP4，physical global BS64，无梯度累积；30K | 运行中 | 约 step `19.3K`；5K checkpoint 为 `23.15%`；因样本预算与 H9 不同，不作受控 batch 效果结论。 |
| H11 | 同层预测与注入是否优于最终层融合 | layer 10 attention hidden 作 query 并在 FFN 前注入；FSDP4，physical BS64，30K | 验证排队 | 训练约 step `8.3K`；4999 checkpoint 已生成；验证 job `130490` 尚未开始。 |
| H12 | ACPD-v2 是否优于同 BS64 的单视角 SFT | backview-only SFT；FSDP4，physical global BS64，无梯度累积；5K | 训练完成，验证排队 | step 4900 loss `0.0290`；4999 checkpoint 已完整保存；验证 job `130491` 尚未开始。 |

## SFT 训练

数据：`libero_multiview_tuned_6view_lerobot`；学生视角：backview；LoRA：PaliGemma rank 16、action expert rank 32。

| ID | GPU / global BS | 学习率 | 步数 | 最后记录 loss | Job | 状态 |
|---|---:|---|---:|---:|---:|---|
| SFT-2GPU | 2 / 16 | 默认 | 30K | 0.0294 | 125793 | 完成 |
| SFT-4GPU | 4 / 32 | 默认 | 30K | 0.0261 | 125794 | 完成 |
| SFT-cosine-30K | 4 / 32 | 1K warmup，`2.5e-5`→`2.5e-6` | 30K | 0.0222 | 126023 | 完成 |
| SFT-cosine-60K | 4 / 32 | 30K 后保持 `2.5e-6` | 60K | 0.0197 | 126409 | 完成 |
| SFT-BS64-5K | 4 / 64 | 1K warmup，`2.5e-5`→`2.5e-6` | 5K | 0.0290（step 4900） | 130285 | 完成 |

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

## H7：特权视觉信息可恢复性

| 实验 | 状态 | 结论 |
|---|---|---|
| H7 精确 attention contribution probe | 完成 | layer 9 和 12 通过；layer 9 相对 layer 12 的 overall gap 优势为 `0.1050`，超过预注册 `0.02` 门槛，按协议选择 layer 9。 |
| H7.1 layer 7/8/10/11 补充扫描 | 完成 | layer 10 在 layer 6–12 中最高，并以 `0.0485` 优势超过次优 layer 11，按协议改选 layer 10。 |

### H7 设计

| 项目 | 设置 |
|---|---|
| Teacher | 冻结的 agentview+wrist SFT step 29,999 |
| Student | 冻结的 backview Flow-only step 4,999 |
| 数据划分 | episode seed 42；1,800 train episodes，200 validation episodes |
| Probe | H7 测 layer 6/9/12，H7.1 补测 7/8/10/11；每层、每个 teacher 视角独立线性 probe；seeds 11/29/47 |
| 优化 | 500 steps；global micro BS8；accumulation 4；effective BS32；Adam `1e-3` |
| 验证 | 32 个固定 batch，共 256 个 held-out 样本；teacher advantage 最高的 64 个样本为 hard subset |

Teacher target 是每个视角对 action attention 的真实残差贡献：使用真实 Q/K/V、对全部有效 image/language/action key 的完整 softmax、action expert 输出投影和 AdaRMS residual gate。agentview 和 wrist 分开预测。负对照仅将目标视角的 K/V 在 batch 内错配一位，保持 action query、噪声动作、flow time 和其他 key 不变。重建单元测试的相对误差小于 `1e-3`。

指标定义：`gap = correct cosine - shuffled cosine`；explained variance 使用 validation target 均值作为基线。可用层必须同时满足 overall gap `>=0.10` 且 95% CI 下界大于 0、hard gap `>=0.05` 且 CI 下界大于 0、explained variance 大于 0。最佳层还需领先次优层至少 `0.02`。

| Layer | Overall gap（95% CI） | Hard gap（95% CI） | Explained variance | 决策 |
|---:|---:|---:|---:|---|
| 6 | `0.0997`（`[0.0846, 0.1091]`） | `0.0706`（`[0.0497, 0.0884]`） | `-0.1468` | 不可用 |
| 7 | `0.3261`（`[0.2938, 0.3602]`） | `0.2776`（`[0.2300, 0.3390]`） | `0.2833` | 通过 |
| 8 | `0.2858`（`[0.2491, 0.3030]`） | `0.2373`（`[0.1970, 0.2794]`） | `0.3420` | 通过 |
| 9 | `0.3417`（`[0.3095, 0.3756]`） | `0.3194`（`[0.2717, 0.3875]`） | `0.2822` | 通过 |
| **10** | **`0.3992`（`[0.3524, 0.4312]`）** | **`0.3779`（`[0.3235, 0.4578]`）** | **`0.3556`** | **通过并选中** |
| 11 | `0.3508`（`[0.3170, 0.3824]`） | `0.3045`（`[0.2435, 0.3634]`） | `0.3113` | 通过 |
| 12 | `0.2367`（`[0.2127, 0.2546]`） | `0.2080`（`[0.1692, 0.2349]`） | `0.0903` | 通过 |

结论：冻结 backview student 能恢复 layer 7–12 中除 layer 6 外的真实 attention contribution。固定协议下 layer 10 比次优 layer 11 高 `0.0485`，超过预注册 `0.02` 门槛，因此局部扫描选择 layer 10。该结果不代表 layer 10 已提升任务成功率，任务效果由 H9 验证。

## H8/H9/H11：ACPD-v2 筛选

| 实验 | 配置 | Job | 状态 |
|---|---|---:|---|
| H8 ACL-only | FSDP2，global BS32，5K，seed 42 | 128417 | 完成；step 4900 supervised loss `0.0316`，4999 checkpoint 已完整保存 |
| H8 四套验证 | 4 GPU，2,000 episodes | 128421 | 完成；`127/2,000`，pooled `6.35%` |
| H9 layer-10 训练 | FSDP4，physical global BS32，无梯度累积，5K | 129710 | 完成；4999 checkpoint 已完整保存 |
| H9 layer-10 验证 | 4 GPU，2,000 episodes | 129711 | 完成；`227/2,000`，pooled `11.35%` |
| H9-scale-b | 最终 hidden 注入；FSDP4，physical global BS64，无梯度累积，30K | 129728 | 运行中；约 step 8,490；4999 checkpoint 验证为 `463/2,000`、`23.15%` |
| H11 smoke | layer 10 attention hidden 作 query 并在 FFN 前注入；FSDP4，physical global BS64，2 steps | 129807 | 完成；有限 loss、非零目标梯度且无 OOM |
| H11 正式训练 | 与 H9-scale-b 相同训练设置；query 与注入共同对齐到 layer 10 attention 后；30K | 129808 | 运行中；约 step 1,460，尚无任务成功率结论 |

H9 的 BS32 5K 筛选与 H8 比较。H11 与 H9-scale-b 都使用 BS64，step 4,999
checkpoint 构成严格注入位置对照；通过标准为 H11 pooled success 提升至少 `1.5`
个百分点，且配对 bootstrap 95% CI 下界大于 0。

### H8 结果

| 方法 | Spatial | Object | Goal | LIBERO-10 | Pooled |
|---|---:|---:|---:|---:|---:|
| Flow-only | 1.80% | 7.60% | 8.40% | 0.00% | 4.45% |
| ACL-only | 2.00% | 15.40% | 7.60% | 0.40% | 6.35% |
| Full ACPD | 2.80% | 10.80% | 12.20% | 0.20% | 6.50% |

| 配对比较 | Pooled 差值 | 95% CI | 结论 |
|---|---:|---:|---|
| ACL-only - Flow-only | `+1.90` 点 | `[+0.75, +3.10]` | ACL 有正向信号。 |
| Full ACPD - ACL-only | `+0.15` 点 | `[-1.15, +1.40]` | 未检测到 Cue 的额外收益。 |

结论：ACL-only 距 Full ACPD 仅 `0.15` 点，小于预注册的 `0.5` 点等效范围，因此 H5 的 5K 增益按协议由 ACL 解释。该结论仅适用于单 seed、5K 筛选；不同 suite 的变化方向不一致。

### H9/H9-scale-b 5K 结果

| 方法 | Spatial | Object | Goal | LIBERO-10 | Pooled |
|---|---:|---:|---:|---:|---:|
| H8 ACL-only，BS32 | 2.00% | 15.40% | 7.60% | 0.40% | 6.35% |
| H9 ACPD-v2，BS32 | 5.20% | 24.20% | 15.60% | 0.40% | 11.35% |
| H9-scale-b，BS64 | 25.40% | 32.40% | 30.00% | 4.80% | 23.15% |

H9 相对 H8 提升 `5.00` 个百分点，task-stratified paired bootstrap 95% CI
为 `[+3.60, +6.45]`，通过预注册门槛。H9-scale-b 的 5K 成绩只作为 H11 的
同 BS64、同注入目标对照；它在 5K steps 处理的样本数是 BS32 的两倍，不能据此
推断 batch size 本身提高了成功率。

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
| H7 原始指标 | `/opt/liutong/openpi-5090-research/acpd-exact-attention-probe/results/metrics_128248.json` |
| H7 Slurm 日志 | `/opt/liutong/openpi-5090-research/acpd-exact-attention-probe/slurm-log/pi05-bv-exact-attn_128248.out` |
| H7.1 原始指标 | `/opt/liutong/openpi-5090-research/acpd-exact-attention-probe/results/metrics_128789.json` |
| H7.1 Slurm 日志 | `/opt/liutong/openpi-5090-research/acpd-exact-attention-probe/slurm-log/pi05-bv-exact-attn-scan_128789.out` |
| H8 协议 | `experiments/acpd-acl-only-5k/protocol.md` |
| H8 分析 | `experiments/acpd-acl-only-5k/analysis.md` |
| H8 ACL/Flow 配对分析 | `/opt/liutong/openpi-5090-evals/acpd-task-success-5k/acl_vs_flow_paired_analysis.json` |
| H8 Full/ACL 配对分析 | `/opt/liutong/openpi-5090-evals/acpd-task-success-5k/full_vs_acl_paired_analysis.json` |
| H9 协议 | `experiments/acpd-v2-layer10-pbs32-5k/protocol.md` |
| H9 分析 | `experiments/acpd-v2-layer10-pbs32-5k/analysis.md` |
| H9 配对分析 | `experiments/acpd-v2-layer10-pbs32-5k/results/h9_vs_h8_paired_analysis.json` |
| H9 layer-10 训练日志 | `slurm-log/pi05-bv-acpdv2-l10-pbs32-5k_129710.out`（完成） |
| H9 loss 曲线 | `artifacts/pi05_acpdv2_h9_loss.png` / `.pdf` |
| H9 batch-scaling 协议 | `experiments/acpd-v2-h9-batch-scaling-30k/protocol.md` |
| H9 batch-scaling 分析 | `experiments/acpd-v2-h9-batch-scaling-30k/analysis.md` |
| H9-scale-b 训练日志 | `slurm-log/pi05-bv-acpdv2-l10-fsdp4-bs64-30k_129728.out`（运行中，约 step 19.3K） |
| H11 协议 | `experiments/acpd-v2-h11-injection-location/protocol.md` |
| H11 smoke 日志 | `slurm-log/smoke-bv-h11-l10-aligned-bs64_129807.out`（完成） |
| H11 正式训练日志 | `slurm-log/pi05-bv-h11-l10-aligned-bs64-30k_129808.out`（运行中，约 step 8.3K） |
| H11 验证任务 | `examples/libero/eval_slurm/pi05_libero_backview_acpd_v2_h11_aligned_fsdp4_bs64_5k.sbatch`，job `130490`（排队） |
| H12 协议 | `experiments/sft-backview-bs64-5k/protocol.md` |
| H12 SFT-BS64 训练日志 | `slurm-log/pi05-bv-sft-bs64-5k_130285.out`（完成，step 4900 loss 0.0290） |
| H12 验证任务 | `examples/libero/eval_slurm/pi05_libero_backview_lora_fsdp4_bs64_5k.sbatch`，job `130491`（排队） |

## Checkpoint 路径检索

保留全部完整 SFT checkpoint 和 ACPD checkpoint。下表只列已验证模型、当前研究对照和在途实验的关键检索点；未列出的 SFT 中间 checkpoint 仍保留在原目录。

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
| H9 ACPD-v2 layer 10，5K | `/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/task_success_5k/pi05_libero_backview_acpd_v2_layer10/pi05_libero_backview_acpd_v2_lora_fsdp4_pbs32_5k/4999` |
| H9-scale-b，4 卡 BS64 当前训练点 | `/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/batch_scaling_30k/pi05_libero_backview_acpd_v2_layer10/pi05_libero_backview_acpd_v2_lora_fsdp4_bs64_30k/14999` |
| H9-scale-b，4 卡 BS64 30K（预期，尚未生成） | `/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/batch_scaling_30k/pi05_libero_backview_acpd_v2_layer10/pi05_libero_backview_acpd_v2_lora_fsdp4_bs64_30k/29999` |
| H11，4 卡 BS64 5K | `/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/injection_location_30k/pi05_libero_backview_acpd_v2_h11_layer10_aligned/pi05_libero_backview_acpd_v2_h11_layer10_aligned_lora_fsdp4_bs64_30k/4999` |
| H12 backview SFT，4 卡 BS64 5K（预期，尚未生成） | `/opt/liutong/openpi_checkpoints/fixed_dataset/sft/backview_bs64_5k/pi05_libero_backview_lora/pi05_libero_backview_lora_fsdp4_bs64_5k/4999` |
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
| H8 ACL-only，5K | `/opt/liutong/openpi-5090-evals/acpd-task-success-5k/acl-only/4999` | 完成，`6.35%` pooled |
| H9 ACPD-v2 layer 10，5K | `/opt/liutong/openpi-5090-evals/acpd-v2-task-success-5k/layer10-exact-contribution-pbs32/4999` | 完成，`11.35%` pooled |
| H9-scale-b 最终 hidden，BS64 5K | `/opt/liutong/openpi-5090-evals/acpd-v2-injection-location-5k/final-hidden/4999` | 完成，`23.15%` pooled |
| H11 同层注入，BS64 5K | `/opt/liutong/openpi-5090-evals/acpd-v2-injection-location-5k/aligned-attention/4999` | 尚未验证 |
