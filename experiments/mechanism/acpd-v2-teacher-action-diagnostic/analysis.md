# H9 teacher action signal 训练诊断

## 目的

检查 H9 训练日志中的 teacher action target 是否始终比 student 当前动作更接近真实
flow target。该分析用于判断动作级 teacher 信号是否值得继续研究；它不是独立的
benchmark 验证，也不是因果消融。

## 数据与指标

- 来源：`slurm-log/pi05-bv-acpdv2-l10-fsdp4-bs64-30k_129728.out`
- 模型：H9-Fixed，backview，layer 10，4 GPU，physical BS64
- 数据：训练过程中记录的当前 batch；每个 step 为日志中的一次聚合统计
- `student_task_loss`：student 动作相对真实 flow target 的任务误差
- `teacher_task_loss`：teacher 动作相对同一真实 target 的任务误差
- `teacher_better_ratio`：当前 batch 中 teacher 误差低于 student 误差的样本比例

## 结果

| Step | Student task loss | Teacher task loss | Teacher better ratio | Action-cosine | Gate |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.4084 | 0.0215 | 1.0000 | 0.0036 | 0.0000 |
| 5K | 0.1339 | 0.0169 | 0.9987 | 0.7157 | 0.0043 |
| 10K | 0.1170 | 0.0151 | 0.9983 | 0.7536 | 0.0051 |
| 15K | 0.1064 | 0.0151 | 0.9972 | 0.7731 | 0.0051 |
| 20K | 0.0961 | 0.0160 | 0.9975 | 0.7890 | 0.0046 |
| 25K | 0.0882 | 0.0163 | 0.9958 | 0.7948 | 0.0043 |
| 29.9K | 0.0832 | 0.0159 | 0.9958 | 0.7999 | 0.0040 |

按 H9 已有 5K 窗口聚合日志计算，teacher/student task-loss 比例如下：

| 训练窗口 | Student loss | Teacher loss | Student / Teacher |
|---|---:|---:|---:|
| 0--5K | 0.16979 | 0.01559 | 10.89× |
| 5K--10K | 0.12411 | 0.01553 | 7.99× |
| 10K--15K | 0.11194 | 0.01554 | 7.20× |
| 15K--20K | 0.10111 | 0.01574 | 6.42× |
| 20K--25K | 0.09197 | 0.01564 | 5.88× |
| 25K--30K | 0.08511 | 0.01550 | 5.49× |

## 结论

1. 在所有记录 step，teacher 的动作误差均明显低于 student；`teacher_better_ratio`
   从 1.0000 到 0.9958，始终高于 99.5%。
2. 这提供了一个正向机制证据：teacher 的动作级监督信号在训练期间是有效的，
   不是随机的内部特征目标。
3. 该证据不能证明 ACPD-v2 的 contribution 注入带来了 benchmark 提升，因为统计
   来自 H9 训练 batch，未与 SFT 做同 batch 对照，也没有闭环仿真或置信区间。
4. teacher 优势在 30K 仍存在，但 H9 相对 SFT 的成功率优势在 25K--30K 消失，
   说明主要瓶颈不是 teacher 目标质量，而是 student 将特权信息转化为最终动作的接口。
5. 后续更值得检验的是直接利用 teacher 的动作方向或动作相关特征，而不是继续增加
   contribution predictor 的容量。动作方向诊断 `136897` 已通过 smoke，正在排队。
