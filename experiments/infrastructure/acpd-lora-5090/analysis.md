# H1/H2：5090 ACPD LoRA 可行性

| 实验 | 配置 | Job | 单卡峰值显存 | 结果 |
|---|---|---:|---:|---|
| H1 | FSDP4，global micro BS8，累积 4，effective BS32，2 steps | 126759 | 17,291 MiB | 通过 |
| H2 | FSDP4，physical global BS32，累积 1，2 steps | 126936 | 17,337 MiB | 通过 |

两组实验均得到有限 loss，selector、predictor 和 LoRA 梯度非零，且没有 OOM。
H2 在几乎相同的显存下省去梯度累积，因此后续原始 ACPD 实验采用 physical BS32、
accumulation 1。该结论只证明训练配置可运行，不代表任务成功率。
