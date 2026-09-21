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

## 与匹配 SFT 的训练 Loss

将 H12 SFT 的 0--5K 与确定性续训 5K--30K 日志合并，并与 H9-scale-b 在
相同步数对齐。SFT 的 `loss` 和 ACPD-v2 的 `supervised_loss` 都是相同的
flow-matching MSE，可以直接比较；ACPD-v2 总 `loss` 还包含 contribution 和 ACL，
不能直接作为 SFT loss 的对应量。

| Step 区间 | SFT supervised | ACPD-v2 supervised | ACPD-v2 相对差异 | Contribution cosine | Gate |
|---|---:|---:|---:|---:|---:|
| 0--5K | 0.037453 | 0.037494 | +0.11% | 0.6067 | 0.00283 |
| 5K--10K | 0.027380 | 0.027406 | +0.09% | 0.7388 | 0.00482 |
| 10K--15K | 0.024764 | 0.024714 | -0.20% | 0.7644 | 0.00512 |
| 15K--20K | 0.022464 | 0.022330 | -0.60% | 0.7807 | 0.00490 |
| 20K--25K | 0.020312 | 0.020326 | +0.07% | 0.7908 | 0.00450 |
| 25K--30K | 0.018728 | 0.018822 | +0.50% | 0.7972 | 0.00412 |

299 个共同日志点的监督 loss Pearson 相关系数为 `0.9985`；全程平均 SFT
为 `0.0251425`，ACPD-v2 为 `0.0251408`。因此，5K 时 `+9.30` 点任务成功率
不能由更低的训练 flow MSE 解释。ACPD-v2 的正向信号更符合辅助监督改变内部
表示或动作结构，而不是改善训练集拟合误差；这仍是机制线索，不是因果证明。

Contribution cosine 从 step 100 的 `0.0874` 增长到 5K 的 `0.7157`，30K 为
`0.7999`，说明大部分可预测信号在早期已经形成，后续仍缓慢改善。Gate 在 step
11,900 达到 `0.0054`，随后降至 step 29,900 的 `0.0040`。该变化支持验证
20K/25K checkpoint，但 gate 大小不能单独预测任务成功率，因为实际注入量还取决于
预测 residual 和下游输出投影。

在 step 29,900，ACPD-v2 总目标的数值构成为 supervised `23.29%`、加权
contribution `60.25%`、加权 ACL `16.46%`。这些比例只描述标量 loss，不能替代
各参数组梯度或任务验证结果。

## 证据

| 内容 | 路径 |
|---|---|
| 2-GPU BS32 log | `slurm-log/pi05-bv-acpdv2-l10-fsdp2-bs32-30k_129727.out` |
| 2-GPU BS32 GPU log | `slurm-log/pi05-bv-acpdv2-l10-fsdp2-bs32-30k_129727_gpu.csv` |
| 4-GPU BS32 log | `slurm-log/pi05-bv-acpdv2-l10-pbs32-5k_129710.out` |
| 4-GPU BS64 log | `slurm-log/pi05-bv-acpdv2-l10-fsdp4-bs64-30k_129728.out` |
| 4-GPU BS64 GPU log | `slurm-log/pi05-bv-acpdv2-l10-fsdp4-bs64-30k_129728_gpu.csv` |
| 30K 验证结果 | `/opt/liutong/openpi-5090-evals/acpd-v2-training-trajectory/final-hidden/29999/summary.txt` |
| SFT/ACPD-v2 loss 图 | `artifacts/pi05_backview_sft_vs_acpdv2_loss.png` |
| SFT/ACPD-v2 loss 指标 | `results/loss_comparison.json` |
