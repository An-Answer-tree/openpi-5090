# H9 Batch-Scaling 分析

## 运行结果

| 配置 | Job | 状态 | 稳定单步时间 | 单卡峰值显存 | Checkpoint |
|---|---:|---|---:|---:|---|
| 2 GPU，BS32 | 129727 | step 83 主动终止 | 12.0--12.6 s | 31,388 MiB | 无 |
| 4 GPU，BS32 参考 | 129710 | 运行中 | 8.1--8.6 s | 17,402 MiB | 尚未生成 |
| 4 GPU，BS64 | 129728 | 运行中 | 约 6.0 s | 31,464 MiB | 尚未生成 |

2 卡任务运行正常，但每个 optimizer step 比 4 卡 BS32 慢约 1.45 倍，因此按
用户决定主动终止。4 卡 BS64 的实测吞吐最高，约 10.7 samples/s。

这些结果只支持运行时间和资源选择。BS64 每步处理的样本数是 BS32 的两倍；
两个 30K 扩展性任务目前都没有完整 checkpoint 或任务成功率结果。

## 证据

| 内容 | 路径 |
|---|---|
| 2-GPU BS32 log | `slurm-log/pi05-bv-acpdv2-l10-fsdp2-bs32-30k_129727.out` |
| 2-GPU BS32 GPU log | `slurm-log/pi05-bv-acpdv2-l10-fsdp2-bs32-30k_129727_gpu.csv` |
| 4-GPU BS32 log | `slurm-log/pi05-bv-acpdv2-l10-pbs32-5k_129710.out` |
| 4-GPU BS64 log | `slurm-log/pi05-bv-acpdv2-l10-fsdp4-bs64-30k_129728.out` |
| 4-GPU BS64 GPU log | `slurm-log/pi05-bv-acpdv2-l10-fsdp4-bs64-30k_129728_gpu.csv` |
