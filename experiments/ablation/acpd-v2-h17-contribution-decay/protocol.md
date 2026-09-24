# H17：非零 contribution 权重衰减

## 问题与预测

检验 H9 在 10K 后持续较强的 contribution 监督是否限制任务成功率。
H15a 的近正交梯度和训练 cosine 平台只提供假设依据，不构成因果证明。
预测：减弱后期 contribution 约束后，20K 成功率高于固定权重 H9。

## 锁定配置

| 项目 | 设置 |
|---|---|
| 起点 | H9 recovery 的 `9999`，恢复完整模型、Adam 状态和 optimizer step=10000 |
| 唯一实验变量 | contribution 权重：10K 为 0.2，10K–15K cosine 衰减，15K 后保持 0.05 |
| 其余 loss | flow=1.0，ACL=0.5；保留原始 final-hidden 注入及 stop-gradient |
| Teacher / student | agentview+wrist teacher 30K / backview student，layer 10 |
| 数据 | 原 fixed_dataset；恢复 sampler 的第 10000 个 batch，seed=42 |
| 资源 | batch 分区，4×5090，32 CPU，96 GiB，5 天 |
| Batch | physical global BS64，梯度累积=1 |
| LR | 保留原 30K cosine，warmup=1000，peak=2.5e-5，end=2.5e-6；不重启 warmup |
| 终点 | optimizer step=20000，新增 10000 updates |
| 保存 | 每 5K，keep_period=1；保存 `14999`、`19999` |
| 网络 | Hugging Face 等离线；清除代理 |

权重按全局 optimizer step 计算：
`w(s)=0.05+0.15*(1+cos(pi*clip((s-10000)/5000,0,1)))/2`。
本次只测试一个预设权重下限，不扫描参数，不新增 seed。

## 比较与判据

- 主比较（CONFIRMATORY）：H17 20K 对同源 H9 recovery 20K；相同 seed 7、
  四套 LIBERO 每套 500 episodes，共 2000 episodes，使用原 fixed 相机位姿。
- 主指标：pooled 成功率差与现有 task-stratified paired bootstrap 95% CI。
  预先设定实用改善为至少 +2 个百分点且 CI 下界大于 0。
- 次比较：匹配 SFT BS64 20K（47.20%）；分别报告四套结果，不能仅选择提高的 suite。
- 机制辅助指标：匹配 step 的 supervised loss、未加权 contribution MSE、cosine、
  gate、ACL 和实际 contribution 权重。使用最后 1K 窗口比较；cosine 比 H9 下降
  超过 0.05 记为明显拟合损失，不能声称无代价保留 teacher 信息。
- 总 loss 会因权重降低而机械下降，不能作为改善证据。
- 若主指标未改善，则不能支持该权重调度；单 seed 结果不等于训练 seed 稳健性。
  即使改善，也只支持该干预在此设置下有效，不能证明 LoRA 容量是唯一机制。

## 实现与核验

直接从明确 checkpoint 路径只读恢复，输出到独立目录；不复制、不移动、不删除源模型。
CPU 检查权重边界、JIT/梯度比例和小模型完整状态恢复；复用已成功的 4 卡训练配置。
正式训练结束后用 Slurm afterok 触发单卡验证数组，再依赖 H9 20K 验证执行配对分析。
没有可用的 agent 定时唤醒工具；自动化覆盖训练、验证、原始结果汇总，后续据结果更新结论。

## 路径

| 内容 | 路径 |
|---|---|
| 源 checkpoint | `/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/trajectory_recovery_20k/pi05_libero_backview_acpd_v2_layer10/pi05_libero_backview_acpd_v2_lora_fsdp4_bs64_20k_keep_all/9999` |
| 新 checkpoint 根目录 | `/opt/liutong/openpi_checkpoints/fixed_dataset/ablation/acpd_v2_h17_contribution_decay` |
| 验证与分析 | `/opt/liutong/openpi-5090-evals/acpd-v2-h17-contribution-decay/19999/` |
| 训练日志 | `/opt/liutong/openpi-5090-research/acpd-v2-h17-contribution-decay/slurm-log/` |

按近期每步 11–13 秒估计，新增 10K updates 需约 31–36 小时，另计保存、加载及排队。
