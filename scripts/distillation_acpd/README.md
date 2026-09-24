# ACPD LoRA 蒸馏

`train_distill.py` 使用冻结的 agentview+wrist teacher 监督单视角 pi0.5 LoRA
student。`--exact-contribution-fusion` 开启 ACPD-v2：student 预测 teacher 的真实
attention contribution，默认还在最终 hidden 注入预测贡献。训练目标为
`Lflow + 0.2 Lcontribution + 0.5 LACL`；H17 仅将 contribution 权重在
10K--15K 从 `0.2` 降至 `0.05`。

正式 H9/H17 使用 4×5090、physical global BS64、梯度累积 1，每 5K 保存且
`keep_period=1`。实际路径和参数以 `scripts/train_slurm/` 中对应 launcher 为准；
实验状态和成功率见 `EXPERIMENT_RESULTS.md`。未开启 exact contribution 时运行的是
早期 ACPD cue 目标，不应与 ACPD-v2 的结果混称。

保存的 `params` 与相应单视角 LoRA 推理配置兼容，常规 policy loader 忽略蒸馏辅助头。
