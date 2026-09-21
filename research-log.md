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
| 2026-09-19 | 方法 | H14 锁定 topview、leftview、rightview 的匹配 SFT baseline：FSDP4、physical BS64、seed 42、cosine 30K，每 5K 保留 checkpoint。 |
| 2026-09-19 | 方法 | H9-scale-b 增加探索性 20K checkpoint 验证；使用与 5K 相同的 2,000 episodes，仅报告训练轨迹变化，不作 ACPD-v2 对 SFT 的方法结论。 |
| 2026-09-19 | 方法 | H9-scale-b 完整 25K checkpoint 使用同文件系统硬链接保护；job 130703 在原训练完成后恢复 25K，job 130704 按相同训练设置重跑至 20K 并保留 5K/10K/15K/20K。 |
| 2026-09-20 | 管理 | 正式 checkpoint 按 `teacher`、`baseline`、`student`、`ablation` 建立同文件系统硬链接索引；不移动或删除源 checkpoint，不纳入 H11，正式学生与消融实验统一以 BS64、每 5K 保存为准。 |
| 2026-09-20 | 方法 | H12 从 5K 恢复至 30K 时恢复模型、优化器和逻辑 batch 5000 的确定性随机采样顺序；只推进 sampler 随机状态，不重读已训练样本。 |
| 2026-09-21 | 结果 | H9-scale-b 30K pooled success 为 58.00%，较同一训练 5K 的 23.15% 提高 34.85 点；该结果支持继续训练，但匹配 backview BS64 SFT 30K 验证完成前不作方法优越性结论。 |
| 2026-09-21 | 方法 | H9 从完整 30K checkpoint 恢复 optimizer、随机数据顺序和 step-conditioned RNG，以末端学习率 2.5e-6 续训至 60K；只验证 40K、50K、60K，不复测第二 seed。 |
| 2026-09-21 | 结果 | H13 loss-only 5K 为 20.60%，H9 为 23.15%；H9 高 2.55 点，配对 95% CI `[+0.40, +4.70]`，达到预注册判据，支持显式 contribution 注入。 |
| 2026-09-21 | 结果 | BS64 SFT 30K 的 topview 为 71.55%，leftview 为 78.65%；backview 正在验证，rightview 排队。 |
| 2026-09-21 | 方法 | 提交 job 131849，使用同一 H9 30K checkpoint 关闭推理注入，区分后期差异来自 residual 注入还是训练期辅助目标。 |
| 2026-09-21 | 管理 | 按用户决定取消未运行的 job 131849，改为优先验证 H9 20K/25K checkpoint 是否早于 30K 达峰。 |
| 2026-09-21 | 方法 | 提交 H9 25K 验证 job 131860；H9 20K 验证 job 131861 依赖恢复训练 130704，均使用固定 2,000 episodes，不复测第二 seed。 |
| 2026-09-21 | 管理 | 按用户决定取消未运行的 H9 30K--60K 续训 job 131642 及其依赖验证 job 131643；保留 30K 及更早 checkpoint 路线。 |
| 2026-09-21 | 结果 | H9/H12 的 299 个对齐监督 loss 点相关系数为 0.9985，全程平均 MSE 几乎相同；5K 的 ACPD-v2 成功率增益不能由更低训练 MSE 解释。 |
| 2026-09-21 | 结果 | 匹配 BS64 SFT 30K 为 60.45%，H9-scale-b 为 58.00%；H9-SFT 差值 -2.45 点，配对 95% CI `[-5.00, 0.00]`，5K 的早期优势没有保持到 30K。 |
| 2026-09-21 | 管理 | 为降低连续多卡资源等待，将未启动的 H9/SFT 25K 验证替换为各四个独立单卡 suite array；验证协议、seed、episode 数和 checkpoint 不变。 |
| 2026-09-21 | 管理 | 保留 rightview baseline 验证；取消 H9 20K 验证 job 131861；H9 恢复训练 130704 只运行到完整 `4999` checkpoint，由 CPU watchdog job 132198 检查明确路径后停止。 |
| 2026-09-21 | 方法 | H15 预注册为持续蒸馏：从 H9 5K 完整状态继续，5K--15K 将 contribution/ACL 权重由 `0.2/0.5` 余弦退火至非零下限 `0.02/0.1`，15K--30K 保持下限；目标是在保留 teacher 信号的同时提高 30K 上限。 |
| 2026-09-21 | 方法 | H15a 预注册为梯度冲突诊断：在 H9 5K/30K 上用相同的200个BS32 batch，测量共享 LoRA 参数中 flow、contribution、ACL 梯度的 cosine、冲突率和范数比；结果决定测试 conflict-aware ACPD 还是动态 gate。 |
| 2026-09-22 | 管理 | H9 `4999` checkpoint 完整写入后，在 debug01 提交 H15a 单卡单 batch smoke job 132308；正式2卡BS32 job 132250 依赖 smoke 成功后运行5K与30K各200个 batch。 |
| 2026-09-22 | 结果 | H9-scale-b 25K pooled success 为 55.75%，30K 为 58.00%；25K-30K 差值 -2.25 点，配对 95% CI `[-4.70, +0.20]`，不支持 25K 已经早于 30K 达峰。 |
| 2026-09-22 | 方法 | H15a 正式任务等待 2 卡期间，预注册单卡 BS1、5K/30K 各100个相同样本的探索性快速诊断；只用于选择下一步方向，不替代正式 BS32 结论。 |
| 2026-09-22 | 管理 | H15a 单卡 smoke job 132308 成功；提交探索性快速诊断 job 132311，正式 BS32 job 132250 继续排队。 |
| 2026-09-22 | 结果 | H15a-fast 完成100个成对BS1 samples：组合冲突率从5K的6%降至30K的3%，cosine中位数从0.4228降至0.3795但仍为正；不支持后期组合梯度冲突增强，下一步转向动态视角 gate。 |
| 2026-09-22 | 方法 | H16 预注册为 token-conditioned dynamic view gate：以零初始化的每token双视角门控严格包含H9全局gate，先运行4卡BS64 5K；仅在任务成功率显著提高后续训到30K。 |
