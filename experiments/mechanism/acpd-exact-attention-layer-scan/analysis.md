# H7.1：Exact-Attention 层扫描

H7.1 沿用 H7 的 teacher、student、数据划分、负对照、三个 probe seeds 和 500-step
预算，仅补测 layer 7、8、10、11。

| Layer | Overall gap（95% CI） | Hard gap（95% CI） | Explained variance | 结论 |
|---:|---:|---:|---:|---|
| 7 | 0.3261（[0.2938, 0.3602]） | 0.2776（[0.2300, 0.3390]） | 0.2833 | 通过 |
| 8 | 0.2858（[0.2491, 0.3030]） | 0.2373（[0.1970, 0.2794]） | 0.3420 | 通过 |
| **10** | **0.3992（[0.3524, 0.4312]）** | **0.3779（[0.3235, 0.4578]）** | **0.3556** | **通过并选中** |
| 11 | 0.3508（[0.3170, 0.3824]） | 0.3045（[0.2435, 0.3634]） | 0.3113 | 通过 |

结合 H7 的 layer 6、9、12 后，layer 10 的 overall gap 比次优 layer 11 高
0.0485，超过预注册的 0.02 选择门槛，因此后续 ACPD-v2 选择 layer 10。该实验
只证明 privileged contribution 可恢复，不证明任务成功率提升。

原始结果：
`/opt/liutong/openpi-5090-research/acpd-exact-attention-probe/results/metrics_128789.json`。
