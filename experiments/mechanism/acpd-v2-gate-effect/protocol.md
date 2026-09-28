# ACPD-v2 注入对动作速度的离线影响

预注册日期：2026-09-29。性质：同 checkpoint 离线机制诊断；不训练、不仿真，
不移动 checkpoint 或数据集。正式仿真因果验证另见
`../acpd-v2-h9-inference-injection-10k/`。

## 问题

H9 的 scalar gate 数值很小，但它乘上的预测贡献向量未必小。检验该注入
在 10K 和 30K 实际改变多少动作速度预测，以及改变的方向是否降低真实
7维 LIBERO 动作的 flow MSE。

## 固定分析

| 项目 | 设置 |
|---|---|
| 模型 | H9-Fixed backview BS64 layer 10，10K `9999` 与 30K `29999` checkpoint 各自单独读取 |
| 数据 | fixed_dataset；按 episode seed 42 取固定 10% 诊断集；每点 128 个 BS8 batch |
| 随机输入 | 每 batch 固定 seed 42 生成同一 noisy action 与 flow time；同 checkpoint 的所有分支共用 |
| 对照 | 原 H9 的完整注入；同一 checkpoint 仅关闭注入；分别只加 agentview 或 wrist 预测贡献 |
| 主指标 | 完整注入减关闭注入的 7维 flow MSE；按 episode 平均，再对 episode 重采样 2,000 次得配对 95% 区间 |
| 次指标 | 两路单独注入的 MSE 差；完整注入造成的速度变化 RMS 与关闭注入速度 RMS 的比；注入速度变化与理想误差修正方向的 cosine |
| 资源 | 冻结模型，单卡；不加载 teacher，不更新参数 |

10K 与 30K 独立报告，不在看到结果后改变采样数或阈值。若完整注入的 MSE 差
小于零且区间上界小于零，只能说明该 checkpoint 在此离线抽样中改善局部
速度预测；若区间跨零，则未检测到局部改善。两路结果仅帮助定位影响来源，
不能分离训练期间的因果作用。无论结果如何，不能以离线 MSE 代替闭环成功率；
已有的 10K 推理关闭注入仿真仍是部署效果的主证据。

H9 10K 来自同配置轨迹恢复，30K 来自原训练，故两点变化不视为单次运行的
严格纵向效应。诊断集对本脚本未参与训练，但 H9 主干已见过这些数据集 episode。
