# Research Log

本文件只记录影响研究方向的决定和最终结果。任务状态见 `research-state.yaml`，完整指标与证据路径见 `EXPERIMENT_RESULTS.md`。

| 日期 | 类型 | 记录 |
|---|---|---|
| 2026-09-11 | 方法 | 对照论文与历史实现后，采用联合训练 selector/predictor、cue variance 和 ACL 的 ACPD 目标。 |
| 2026-09-11 | 结果 | H1/H2 均通过：FSDP4 effective BS32 可运行；physical BS32、accumulation 1 的单卡峰值为 17,337 MiB，后续原始 ACPD 使用该配置。 |
| 2026-09-12 | 审计 | 论文旧表格混用了 `historical_sg`、gated 和 ungated 配置，不能作为受控组件消融证据；后续结果以当前复现实验为准。 |
| 2026-09-13 | 结果 | H3 不支持 6+12 层：layer 6 的 5K supervised loss 最低，三组差异小于 1%，不能推断任务成功率。 |
| 2026-09-13 | 结果 | H4 未发现 Cue 或 ACL 改善 2K supervised-loss 收敛；相对 Flow-only 的差异均未达到预设 1% 阈值。 |
| 2026-09-14 | 结果 | H5 支持 Full ACPD：5K pooled success 为 6.50%，Flow-only 为 4.45%，差值 +2.05 点，配对 95% CI `[+0.90, +3.25]`。 |
| 2026-09-14 | 方法 | H7 改用 teacher 真实 Q/K/V attention contribution、held-out episode 和同 query 的 shuffled-view 负对照，先检验 privileged cue 是否可恢复。 |
| 2026-09-14 | 结果 | H7 支持可恢复性：layer 9 和 12 通过，layer 6 因 explained variance 为负而失败；粗扫描选择 layer 9。 |
| 2026-09-16 | 结果 | H8 ACL-only 为 6.35%，Full ACPD 仅高 0.15 点，95% CI `[-1.15, +1.40]`；旧 Cue 没有检测到额外收益。 |
| 2026-09-16 | 结果 | H7.1 完成 layer 6--12 局部扫描；layer 10 overall gap 为 0.3992，比次优 layer 11 高 0.0485，因此 ACPD-v2 选择 layer 10。 |
| 2026-09-17 | 方法 | H9 使用 layer-10 exact contribution、ACL、最终 hidden gated injection，FSDP4 physical BS32，无梯度累积。 |
| 2026-09-18 | 结果 | H9 为 11.35%，H8 为 6.35%；差值 +5.00 点，配对 95% CI `[+3.60, +6.45]`，通过预注册门槛。 |
| 2026-09-19 | 结果 | H11 同层注入为 21.20%，H9-scale-b 最终 hidden 注入为 23.15%；差值 -1.95 点，95% CI `[-4.10, +0.20]`，不支持前移注入。 |
| 2026-09-19 | 结果 | H12 匹配 BS64 SFT 为 13.85%，H9-scale-b 为 23.15%；差值 +9.30 点，95% CI `[+7.30, +11.35]`，支持 ACPD-v2 组合方案。 |
| 2026-09-19 | 方法 | H13 仅关闭 contribution residual injection，保留 contribution loss、ACL、batch、seed 和 schedule，用于判断 H9 是否需要显式注入。 |
