# 可观测 gate 可靠性分析

## 结果

Job `136904_0`（实际 Slurm 子任务 `136974`）读取 H9-Fixed 10K checkpoint，完成
128 个 BS8 batch、194 个 episode。只使用 student 可见的两路动作改变量，按 episode
奇偶做两折交叉验证；真实 flow target 只用于生成离线标签。

| 可观测量 | 留出 AUROC（两折） | episode 相关系数 |
|---|---:|---:|
| 两路改变量 cosine | `0.5600 / 0.4906` | `-0.1412` |
| 两路改变量分歧度 | `0.5795 / 0.4962` | `+0.1439` |
| 两路改变量幅度不平衡度 | `0.5769 / 0.5098` | `+0.1297` |

作为不可部署上限，使用真实 flow target 选择 on/off 可将 episode mean MSE 相对关闭
注入降低 `0.00006326`；受益 episode 占 `99/194=51.03%`。该上限不能在推理时使用。

预注册判据要求 10K 与 30K 两个 checkpoint 的两个留出折 AUROC 均高于 `0.60` 且
方向一致。10K 两折均未达到。30K 结果同样未达到：cosine `0.5268/0.5387`、
分歧度 `0.5128/0.5241`、幅度不平衡度 `0.5357/0.5041`；各特征的方向也没有
稳定区分受益和有害 episode。因此不提交动态 gate 训练。

30K 的不可部署 target oracle 相对关闭注入降低 MSE `0.00004116`，说明理论上有很小
的条件选择上限，但 student-only 特征无法恢复该上限。

证据：
10K：`/opt/liutong/openpi-5090-research/acpd-v2-gate-effect/results/136974/observable_gate.json`；
30K：`/opt/liutong/openpi-5090-research/acpd-v2-gate-effect/results/136904/observable_gate.json`。
