# ACPD-v2 两路注入的离线差异：探索性补充分析

预注册日期：2026-09-29。此问题是在已看到完整注入及两路单独注入的总体均值后提出，
因此无论下述区间如何，结果都属于**探索性**，不是独立确认实验。

## 问题与数据

H9 的 agentview 与 wrist 预测贡献相加后通过同一 scalar gate 注入。
检查两路是否在同一 checkpoint、同一动作输入上的局部 flow 误差方向不同，
以及两路相加是否产生明显交互。只读取现有 10K、30K `per_sample.npz`：

| Checkpoint | 文件 | 样本 |
|---|---|---:|
| 10K | `/opt/liutong/openpi-5090-research/acpd-v2-gate-effect/results/136880/per_sample.npz` | 1,024 |
| 30K | `/opt/liutong/openpi-5090-research/acpd-v2-gate-effect/results/136879/per_sample.npz` | 1,024 |

四列误差顺序固定为 `off, agent_only, wrist_only, on`。误差已经是每样本真实
7 维动作的 flow MSE。同一 checkpoint 的四列共用样本、noise 和 flow time。
无需重新训练、加载模型或运行仿真。

## 固定分析

先按 episode 求各分支 MSE 均值，再对 episode 求总体均值。每个 checkpoint
独立以 seed 42 有放回重采样 episode 2,000 次，报告以下三个配对差值及其
百分位 95% 区间：

| 指标 | 定义 | 正值的含义 |
|---|---|---|
| 分路差异（主指标） | `wrist_only - agent_only` | agentview 单路的局部 MSE 更低 |
| 加入 wrist 的增量 | `on - agent_only` | 在 agentview 单路上加入 wrist 后局部 MSE 更高 |
| 两路交互（次指标） | `on - agent_only - wrist_only + off` | 两路合用的误差增量高于单路增量之和 |

若主指标两个 checkpoint 的区间均严格大于零，才称本抽样观察到一致的
agentview 局部优势；否则只报告点估计。交互指标只用于判断简单相加是否
在**离线误差**上产生额外影响。三个指标与两个 checkpoint 均不做多重比较校正。

这项分析不能测量闭环成功率，不能隔离训练期间贡献预测损失的作用，
也不能证明某个视角本身无用。10K 与 30K 来自不同训练运行，不能据此作
严格的纵向因果推断。已提交的同 checkpoint 关闭注入仿真仍是部署效果证据。
