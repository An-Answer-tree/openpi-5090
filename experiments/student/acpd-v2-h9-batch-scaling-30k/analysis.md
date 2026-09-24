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

| Step | SFT pooled | ACPD-v2 pooled | ACPD-v2 - SFT |
|---:|---:|---:|---:|
| 5K | 13.85% | 23.15% | +9.30 点 |
| 25K | 55.80% | 55.75% | -0.05 点 |
| 30K | 60.45% | 58.00% | -2.45 点 |

ACPD-v2 从 5K 到 30K 的绝对成功率提高 `34.85` 点，但相对匹配 SFT 的优势从
`+9.30` 点变为 `-2.45` 点。20K 的匹配验证正在运行；在结果完成前，不能从
5K、25K、30K 三个点确定优势消失的具体 step。

## 与匹配 SFT 的训练 Loss

将 H12 SFT 的 0--5K 与确定性续训 5K--30K 日志合并，并与 H9-scale-b 在
相同步数对齐。SFT 的 `loss` 和 ACPD-v2 的 `supervised_loss` 都是相同的
flow-matching MSE，可以直接比较；ACPD-v2 总 `loss` 还包含 contribution 和 ACL，
不能直接作为 SFT loss 的对应量。

| Step 区间 | 点数 | SFT supervised | ACPD-v2 supervised | 相对差异 | 成对绝对差均值 |
|---|---:|---:|---:|---:|---:|
| 0--5K | 49 | 0.037453 | 0.037494 | +0.11% | 0.000224 |
| 5K--10K | 50 | 0.027380 | 0.027406 | +0.09% | 0.000402 |
| 10K--15K | 50 | 0.024764 | 0.024714 | -0.20% | 0.000390 |
| 15K--20K | 50 | 0.022464 | 0.022330 | -0.60% | 0.000290 |
| 20K--25K | 50 | 0.020312 | 0.020326 | +0.07% | 0.000298 |
| 25K--30K | 50 | 0.018728 | 0.018822 | +0.50% | 0.000222 |

299 个共同日志点的监督 loss Pearson 相关系数为 `0.9985`；全程平均 SFT
为 `0.0251425`，ACPD-v2 为 `0.0251408`。从首个到末个 5K 窗口，SFT 与
ACPD-v2 的监督 loss 分别下降 `50.00%` 和 `49.80%`。六个窗口的相对差异均在
`[-0.60%, +0.50%]`。因此，两种方法对 flow 训练目标的拟合轨迹基本相同。

5K 时 ACPD-v2 的窗口平均监督 loss 略高 `0.11%`，任务成功率却高 `9.30` 点；
25K 时监督 loss 差 `+0.07%`，成功率几乎相同；30K 时监督 loss 差 `+0.50%`，
成功率低 `2.45` 点。现有数据支持“训练 flow MSE 不能解释或预测成功率差异”，
但不能仅凭 loss 确认内部表示改变这一因果机制。

## ACPD-v2 辅助目标轨迹

| Step 区间 | 加权 contribution | 加权 ACL | Contribution cosine | Gate | Teacher 更优比例 |
|---|---:|---:|---:|---:|---:|
| 0--5K | 0.105039 | 0.028559 | 0.6067 | 0.00283 | 99.856% |
| 5K--10K | 0.069540 | 0.020408 | 0.7388 | 0.00482 | 99.813% |
| 10K--15K | 0.061734 | 0.018222 | 0.7644 | 0.00512 | 99.761% |
| 15K--20K | 0.055988 | 0.016250 | 0.7807 | 0.00490 | 99.704% |
| 20K--25K | 0.051786 | 0.014626 | 0.7908 | 0.00450 | 99.676% |
| 25K--30K | 0.048932 | 0.013410 | 0.7972 | 0.00412 | 99.608% |

加权 contribution 和 ACL 从首个到末个窗口分别下降 `53.42%` 和 `53.04%`；
同期 contribution cosine 从 `0.6067` 增至 `0.7972`。按窗口均值计算，全部
cosine 增量的 `69.35%` 已在 5K--10K 窗口实现，之后仍持续增加但速度较慢。

ACPD-v2 标量总目标中，监督/contribution/ACL 的窗口占比从
`21.91/61.39/16.69%` 变为 `23.19/60.29/16.52%`，比例基本稳定；辅助目标没有
在数值上自然消失。该占比只描述 loss 数值，不等于梯度贡献。

Student 的 7 维任务误差窗口均值从 `0.16979` 降至 `0.08511`，下降 `49.88%`；
teacher 同期保持在 `0.0155--0.0157`，每个窗口 teacher 更优比例均高于 `99.6%`。
这说明在训练批次的该指标上，teacher 到 30K 仍提供明显不同于 student 的动作信号；
它不直接等价于仿真成功率。

Predictor gradient norm 从 `0.04132` 降至 `0.02285`，gate gradient norm 从
`0.00286` 降至 `0.00223`，而 LoRA gradient norm 从 `0.08726` 增至 `0.11084`。
因此后期辅助头仍在更新，但其记录到的梯度范数下降；仅凭范数不能判断这些梯度是否
提高任务成功率。

Gate 在 step 11,900 达到观测峰值 `0.0054`，step 29,900 为 `0.0040`；末个
5K 窗口均值相对峰值低 `23.63%`。Gate 大小不能单独给出注入量，因为实际 residual
还取决于 predictor 输出。

在 step 29,900，ACPD-v2 总目标的数值构成为 supervised `23.29%`、加权
contribution `60.25%`、加权 ACL `16.46%`。这些比例只描述标量 loss，不能替代
各参数组梯度或任务验证结果。

## 问题定位

### 10K 后 contribution 学习快速进入平台

| 区间 | Supervised | Contribution cosine | Gate | 加权辅助/监督 |
|---|---:|---:|---:|---:|
| 25K--30K | 0.018822 | 0.7972 | 0.00412 | 3.31 |
| 30K--35K | 0.018034 | 0.8007 | 0.00386 | 3.33 |
| 35K--40K | 0.017476 | 0.8026 | 0.00366 | 3.35 |
| 40K--42.3K | 0.017283 | 0.8038 | 0.00351 | 3.35 |

从 25K--30K 到 35K--40K，supervised loss 继续下降 `7.15%`，但 contribution
cosine 只增加 `0.0054`，gate 同时下降 `11.25%`。因此，35K 相对 30K 的任务提升
更可能来自额外的低学习率任务训练，而不是新获得了大量 privileged contribution。
这需要匹配的 SFT 35K 结果才能最终区分。

### Contribution 梯度不冲突，但持续施加较强的近正交约束

H15a 正式 BS32 诊断显示：30K 时加权 contribution 梯度范数中位数是 flow 梯度的
`1.175` 倍，但两者 cosine 中位数仅为 `0.0957`；ACL 对应值为 `0.851` 倍和
`0.9012`。组合辅助梯度与 flow 的 cosine 为 `0.5920`，没有负冲突。

这排除了“辅助梯度直接反向破坏 flow”的解释，但没有排除容量和轨迹约束：强度接近
flow 的 contribution 梯度主要把共享 LoRA 参数推向与当前 flow 改善近乎正交的方向。
对容量有限的 LoRA，这可能保持早期正则化收益，同时限制后期任务最优解。

### 当前注入接口只让 gate 接收任务梯度

`ExactContributionHead.fuse()` 与 Gemma 内部注入都对预测 residual 使用
`stop_gradient`。因此：

- contribution predictor 只能由 contribution MSE 训练；layer-10 共享表征还同时接收主任务梯度；
- 注入 residual 的方向不能通过 supervised flow loss 调整；
- 只有一个全局标量 gate 能根据任务 loss 调整注入强度。

H13 已证明这种注入在 5K 带来 `+2.55` 点，但 gate 在约 12K 后持续下降。结合上述
梯度路径，当前最符合数据的工作假设是：预测 contribution 早期提供有用的表示偏置，
后期却缺少任务自适应接口；模型只能通过减小 gate 来降低不再稳定有益的注入。

### 仍不能确定的部分

- 15K 目前只完成 Spatial/Object；两套合计 ACPD-v2 为 `56.90%`，SFT 为
  `50.80%`，临时差值 `+6.10` 点。Goal/LIBERO-10 完成前不作正式结论。
- 现有 BS64 消融没有单独的 ACL-only，因此 H13 相对 SFT 的收益不能严格拆分为
  contribution loss 与 ACL 两部分。
- 上述“后期过约束”和“注入缺少任务适配”仍是由数据支持的工作假设，不是因果结论。

最小可证伪实验应从完整 10K 状态继续到 20K：保持 ACL `0.5` 和非零 contribution
信号，只把 contribution 权重从 `0.2` 平滑降到 `0.05`。若 20K 成功率提高且
contribution cosine 没有明显崩溃，则支持后期 contribution 过约束；若无改善，再测试
在 detached contribution 后增加一个只接收 flow 梯度的小型零初始化 adapter，使注入
方向可任务适配而不破坏 contribution target。

## 证据

| 内容 | 路径 |
|---|---|
| 2-GPU BS32 log | `slurm-log/pi05-bv-acpdv2-l10-fsdp2-bs32-30k_129727.out` |
| 2-GPU BS32 GPU log | `slurm-log/pi05-bv-acpdv2-l10-fsdp2-bs32-30k_129727_gpu.csv` |
| 4-GPU BS32 log | `slurm-log/pi05-bv-acpdv2-l10-pbs32-5k_129710.out` |
| 4-GPU BS64 log | `slurm-log/pi05-bv-acpdv2-l10-fsdp4-bs64-30k_129728.out` |
| 4-GPU BS64 GPU log | `slurm-log/pi05-bv-acpdv2-l10-fsdp4-bs64-30k_129728_gpu.csv` |
| 30K--60K 续训 log | `slurm-log/pi05-bv-acpdv2-l10-bs64-r60k_132390.out` |
| H15a 正式梯度分析 | `experiments/mechanism/acpd-v2-h15a-gradient-conflict/analysis.md` |
| 注入实现 | `src/openpi/models/pi0_distill_acpd.py`、`src/openpi/models/gemma.py` |
| 30K 验证结果 | `/opt/liutong/openpi-5090-evals/acpd-v2-training-trajectory/final-hidden/29999/summary.txt` |
| SFT/ACPD-v2 loss 图 | `artifacts/pi05_backview_sft_vs_acpdv2_loss.png` |
| SFT/ACPD-v2 loss 指标 | `results/loss_comparison.json` |
