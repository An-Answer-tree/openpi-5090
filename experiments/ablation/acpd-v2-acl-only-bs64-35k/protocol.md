# BS64 ACL-only 35K 消融协议

## 问题与预测

检验正式 ACPD-v2 中的 exact contribution 学习与注入，在已有 ACL 之外是否带来额外任务收益。
ACL-only 可能优于 SFT，但历史 BS32、5K 结果不能预测本次 BS64、35K 的差值。
预测 H9-Fixed 35K 成功率高于同进度 ACL-only；反向或无差异结果同样报告。

## 锁定对照

| 项目 | ACL-only | H9-Fixed |
|---|---|---|
| Teacher / student | agentview+wrist 30K / backview pi0.5 LoRA | 相同 |
| 数据与随机性 | fixed_dataset，seed 42，原始确定性采样顺序 | 相同 |
| 资源 | 4×5090 FSDP，physical global BS64，累积 1 | 相同 |
| 训练目标 | flow 1.0 + ACL 0.5；contribution 权重 0，关闭注入 | flow 1.0 + ACL 0.5 + contribution 0.2，开启注入 |
| 优化 | warmup 1K，peak 2.5e-5，30K cosine 至 2.5e-6，之后保持末值 | 相同 |
| 终点与保存 | 35K；每 5K 保存，keep_period=1 | 已有 30K/35K checkpoint |

ACL-only 复用现有 `exact_contribution_fusion` 且关闭注入的模型配置，以匹配 H9
的参数树；contribution 指标仍计算，但权重为零，不参与参数更新。teacher、dataset、
已有 checkpoint 仅只读使用；新训练写独立目录。不从 H9 或 SFT checkpoint 恢复。

## 验证和判据

- 只验证 ACL-only 的 30K (`29999`) 与 35K (`34999`)；不验证 5K。
- 每个 checkpoint 全量运行 LIBERO Spatial、Object、Goal、LIBERO-10；每套 10 个任务，
  每任务 50 回合，四套共 2,000 回合。使用 fixed 相机位姿、seed 7，与 H9/SFT 相同。
- 主比较：35K 的 H9-Fixed 减 ACL-only；次比较：30K 的同进度差值。
  对每项给出四套成功率、pooled 差值及 task-stratified paired bootstrap 95% CI。
- 同时报告 ACL-only 减同进度 SFT 的差值。主指标的 95% CI 下界大于零，才称为
  检测到 Full 相对 ACL-only 的额外收益；否则报告证据不足或反向结果。
- 本次不以训练 loss、单项 benchmark、不同步数峰值比较代替上述主指标。
  单 seed 结果不代表跨训练 seed 的稳健性。

训练成功后以 Slurm `afterok` 自动触发 30K/35K 验证和配对分析；完整结果出来前
只记录运行状态，不填任务成功率结论。
