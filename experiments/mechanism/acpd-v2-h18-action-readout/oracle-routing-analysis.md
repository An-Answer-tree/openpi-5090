# H18 动作读出条件路由上限分析

## 实际数据

复用 H18 的冻结验证结果：
`/opt/liutong/openpi-5090-research/acpd-v2-h18-action-readout/results/136130/validation_errors.npz`。
数据包含 1,024 个样本、194 个 episode，以及四列 flow MSE：冻结 H9、hidden-only、
hidden-layer10、hidden-contribution。

## 结果

| 读出策略 | episode mean flow MSE |
|---|---:|
| 冻结 H9 | 0.087586 |
| 固定 hidden-only | 0.086893 |
| 固定 hidden-layer10 | 0.086849 |
| 固定 hidden-contribution | 0.086874 |
| episode-level 理想选择 | 0.084997 |

最佳固定读出头为 hidden-layer10。episode-level 理想选择相对它的差值为
`-0.001851`，相对下降 `2.131%`，episode bootstrap 95% 区间为
`[-0.002264, -0.001492]`。

理想选择的获胜比例为：冻结 H9 `19.59%`、hidden-only `31.44%`、hidden-layer10
`18.04%`、hidden-contribution `30.93%`。

## 结论与限制

不同 episode 的最佳读出头不一致，理想条件选择有稳定的局部上限。这支持继续研究
由 student 可见特征预测路由或融合权重，而不是把 contribution 固定加到所有状态。
该选择使用真实 flow target，不能作为部署结果，也不能直接推出仿真成功率提升。下一步
必须使用不访问真实 target 的可观测路由器，并与同参数量的固定融合对照比较。

证据脚本：`scripts/distillation_acpd/analyze_action_readout_oracle.py`。
