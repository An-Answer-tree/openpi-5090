# H9 Batch-Scaling 分析

## 运行结果

| 配置 | Job | 状态 | 稳定单步时间 | 单卡峰值显存 | Checkpoint |
|---|---:|---|---:|---:|---|
| 2 GPU，BS32 | 129727 | step 83 主动终止 | 12.0--12.6 s | 31,388 MiB | 无 |
| 4 GPU，BS32 参考 | 129710 | 完成 5K | 8.1--8.6 s | 17,402 MiB | `4999` |
| 4 GPU，BS64 | 129728 | 完成 30K | 约 6.0 s | 31,464 MiB | `29999` |

2 卡任务运行正常，但每个 optimizer step 比 4 卡 BS32 慢约 1.45 倍，因此按
用户决定主动终止。4 卡 BS64 的实测吞吐最高，约 10.7 samples/s。

这些结果支持 4 GPU、physical global BS64 作为正式训练配置。BS64 每步处理的
样本数是 BS32 的两倍。

## 训练轨迹验证

| Step | Spatial | Object | Goal | LIBERO-10 | Pooled |
|---:|---:|---:|---:|---:|---:|
| 5K | 25.40% | 32.40% | 30.00% | 4.80% | 23.15% |
| 30K | 64.00% | 73.00% | 61.80% | 33.20% | 58.00% |

30K 比同一训练的 5K 高 `34.85` 点，说明该配置在 5K 后仍有明显训练收益。
匹配的 backview BS64 SFT 30K 尚未完成验证，因此该结果不能单独证明
ACPD-v2 优于 SFT。

## 证据

| 内容 | 路径 |
|---|---|
| 2-GPU BS32 log | `slurm-log/pi05-bv-acpdv2-l10-fsdp2-bs32-30k_129727.out` |
| 2-GPU BS32 GPU log | `slurm-log/pi05-bv-acpdv2-l10-fsdp2-bs32-30k_129727_gpu.csv` |
| 4-GPU BS32 log | `slurm-log/pi05-bv-acpdv2-l10-pbs32-5k_129710.out` |
| 4-GPU BS64 log | `slurm-log/pi05-bv-acpdv2-l10-fsdp4-bs64-30k_129728.out` |
| 4-GPU BS64 GPU log | `slurm-log/pi05-bv-acpdv2-l10-fsdp4-bs64-30k_129728_gpu.csv` |
| 30K 验证结果 | `/opt/liutong/openpi-5090-evals/acpd-v2-training-trajectory/final-hidden/29999/summary.txt` |
