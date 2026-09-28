# ACPD-v2 注入对动作速度的离线影响

## 实际运行

冻结 H9-Fixed backview 10K（重建轨迹）与 30K（原训练）checkpoint，分别运行
128 个 BS8 的真实 fixed_dataset batch。每点 1,024 个样本、194 个 episode；
两个 checkpoint 的 episode 顺序完全一致，误差与附加指标均为有限值。
同一 checkpoint 的四个分支使用同一 noisy action 与 flow time。两项 Slurm
子任务均 exit 0：10K 为 `136879_0`（实际 job 136880），30K 为 `136879_1`。

| Checkpoint | 关闭注入 MSE | 完整注入 MSE | 注入−关闭 MSE | 按 episode 配对 95% 区间 |
|---|---:|---:|---:|---:|
| 10K | 0.12407959 | 0.12408793 | +0.00000834 | [-0.00001629, +0.00003204] |
| 30K | 0.09762943 | 0.09763204 | +0.00000261 | [-0.00001384, +0.00001952] |

| Checkpoint | agentview 单路−关闭 MSE | wrist 单路−关闭 MSE | 样本级速度 RMS 比均值 | 注入改变量与理想修正 cosine | gate |
|---|---:|---:|---:|---:|---:|
| 10K | -0.00000992 | +0.00001732 | 0.290% | 0.0209 | 0.00470 |
| 30K | -0.00000544 | +0.00000464 | 0.187% | 0.0154 | 0.00397 |

所有完整注入和单路注入相对关闭注入的 MSE 区间都跨零，因此该离线抽样没有
检测到任一分支的局部 flow MSE 改善。注入确实改变了速度预测，但改变量小；
表中的 RMS 比是先按样本计算 `||v_on-v_off||/||v_off||`，再按 episode 平均，
不是对所有样本合并后取 RMS 之比。这一点明确限定了协议次指标的实现口径。

10K 的 H9 相对 SFT 闭环成功率高 14.20 点，但本实验不能把该差异归因于
推理期注入：小的局部速度差仍可能在闭环轨迹中放大，辅助损失也可能改变训练
得到的主干。已提交的同 checkpoint 10K 推理期关闭注入仿真 `136866` 才能
直接检验部署时该分支是否重要。10K 和 30K 来自不同训练运行，因此这里不把
两个点的数值下降解释为已证实的训练阶段因果效应。

原始摘要：
`/opt/liutong/openpi-5090-research/acpd-v2-gate-effect/results/136880/summary.json`
和 `/opt/liutong/openpi-5090-research/acpd-v2-gate-effect/results/136879/summary.json`；
逐样本误差及 episode 编号见各自同目录 `per_sample.npz`。
