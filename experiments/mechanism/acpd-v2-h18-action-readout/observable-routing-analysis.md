# H18 可观测动作读出路由分析

## 实际运行

Job `136935` 在 debug01 使用 1×5090、24G 内存，加载 H9 30K checkpoint 和 H18
已训练的三个 readout head，跳过 readout 再训练，仅提取 1,024 个样本、194 个
episode 的 Student 特征。随后在 CPU 上进行两折 episode-held-out ridge 路由。

## 设置

- 输入：每个 action token 的 final hidden、layer10 hidden、predicted contribution；
  经过 action-horizon 平均和每32维分组平均，得到288维特征。
- 目标：训练集中的四路 flow MSE（冻结 H9、hidden-only、hidden-layer10、
  hidden-contribution）。
- 路由器：ridge 线性回归，正则系数 `1e-2`；episode id 奇偶两折。
- 测试选择只使用预测误差，不访问真实 flow target。

## 结果

| 指标 | 数值 |
|---|---:|
| 固定 hidden-layer10 episode mean MSE | 0.085064 |
| 可观测路由 episode mean MSE | 0.085087 |
| 路由 - 固定 layer10 | `+0.00010792` |
| 相对变化 | `-0.127%` |
| episode bootstrap 95% CI | `[-0.00029520, +0.00053006]` |

路由选择比例为冻结 H9 `28.61%`、hidden-only `18.75%`、hidden-layer10 `15.82%`、
hidden-contribution `36.82%`。虽然选择了不同分支，但测试 episode 上没有降低误差。

## 结论

理想 target-based oracle 的 2.131% 上限不能由当前简单、Student-only 的线性路由器
恢复。该结果不支持继续堆叠小型 gate；后续应优先使用训练期端到端特征融合或直接
teacher-action distillation，并用闭环 benchmark 验证。

证据：

- 特征提取：`/opt/liutong/openpi-5090-research/acpd-v2-h18-action-readout/results/136935/validation_features.npz`
- 路由结果：`/opt/liutong/openpi-5090-research/acpd-v2-h18-action-readout/results/136935/observable_router.json`
- 分析脚本：`scripts/distillation_acpd/analyze_observable_action_router.py`
