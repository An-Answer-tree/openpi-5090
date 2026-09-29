# H18 动作读出固定融合分析

## 数据

复用 Job `136935` 的 1,024 个样本、194 个 episode 特征，以及 H18 的
`readout_params.npz`。没有重新训练模型。

## 结果

| 策略 | 相对固定 hidden-layer10 的 MSE 变化 |
|---|---:|
| 50% hidden-only + 50% hidden-contribution | `+0.036%`；MSE差值95% CI `[-0.00029722,+0.00036291]` |
| 两折选择 alpha（0/0.25/0.5/0.75/1） | `+0.125%`；MSE差值95% CI `[-0.00024911,+0.00045101]` |

两折选择最终只选 pure hidden-contribution 或 pure hidden-layer10，没有选择中间混合值。

## 结论

不同读出头的 oracle 互补性不能通过固定平均或简单 alpha 选择转化为稳定的动作
误差下降。不提交该方向的正式闭环验证；后续优先使用训练期端到端融合或直接
teacher-action 蒸馏。

证据：`/opt/liutong/openpi-5090-research/acpd-v2-h18-action-readout/results/136935/ensemble.json`。
