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

## 证据边界

上述选择依据是预注册的点估计门槛。现有结果只保存各层分别的 episode bootstrap
区间，没有保存 layer 10 减 layer 11 的逐样本配对差值，因此不能把 `0.0485`
解释为两层差异已达到统计显著。独立 seed 的完整复核应重新保存逐样本结果并对
层间差值做配对 bootstrap；历史同类 probe 约需 6 小时、2 张 GPU。本轮正式训练
占用资源期间不提交该长实验。

原始结果：
`/opt/liutong/openpi-5090-research/acpd-exact-attention-probe/results/metrics_128789.json`。

## Layer 10 两路目标的描述性差异

| 教师视角 | Correct-shuffled cosine gap | Explained variance | Contribution/总attention范数比 |
|---|---:|---:|---:|
| agentview | 0.4029 | 0.4722 | 0.3122 |
| wrist | 0.3956 | 0.2389 | 0.5945 |

wrist 目标在教师 attention 输出中的相对范数更大，但线性 probe 的恢复方差较低。
这提示两路预测质量和尺度不同，为分别建模/融合提供设计动机；
现有结果未保存逐样本视角间差值，不能声称两视角差异达到统计显著，
也不能由 probe 指标推断 wrist 注入提高或降低闭环成功率。
