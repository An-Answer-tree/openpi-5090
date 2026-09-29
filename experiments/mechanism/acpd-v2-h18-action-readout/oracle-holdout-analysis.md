# H18 动作读出 oracle 的 episode 内交叉复核

## 数据与设置

复用 H18 验证误差 `/opt/liutong/openpi-5090-research/acpd-v2-h18-action-readout/results/136130/validation_errors.npz`，
共 1,024 个样本、194 个 episode。179 个 episode 至少有两个样本，纳入交替划分。
每个 episode 用一半样本选择误差最低的读出头，在另一半测试，再交换两半；固定
hidden-layer10 作为对照。

## 结果

| 指标 | 数值 |
|---|---:|
| holdout oracle - 固定 layer10 | `+0.00002332` MSE |
| 相对变化 | `-0.027%` |
| episode bootstrap 95% CI | `[-0.00047134, +0.00055144]` |
| 选择次数（冻结H9/hidden-only/layer10/contribution） | `88/113/54/103` |

## 结论

同样本上的 episode-level oracle 降低 `2.131%`，但在 episode 内未参与选择的样本上不再
下降。当前数据不支持存在稳定、可泛化的状态条件读出互补；原 oracle 结果只能作为
不可部署的同样本上限。H18 不进入动态路由或固定读出融合的正式训练。
