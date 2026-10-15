# SFT-View 30K→32K 执行记录

协议提交：`f775536`，先于Slurm提交。提交时间：2026-10-16 00:14:09 CST。
状态核对：2026-10-16 00:14 CST。

| 视角 | Job | 状态 | 资源 | checkpoint |
|---|---:|---|---|---|
| leftview | 149303 | PENDING，Priority | 4GPU、32CPU、96G、24小时 | 从29999恢复，新增31000/31999 |
| rightview | 149304 | PENDING，Priority | 同上 | 同上 |
| topview | 149305 | PENDING，Priority | 同上 | 同上 |

三个任务均无节点绑定、排除节点或reservation参数；`ReqNodeList`和`ExcNodeList`
均为null，由batch分区自动调度。核对时均尚未启动，GPU恢复和新checkpoint尚未验证。
模型、优化器、数据接续及保存设置见[协议](protocol.md)；尚无新增成功率结论。

## 已完成检查

| 检查 | 结果 |
|---|---|
| 源checkpoint | 三个29999均有完整提交标记、params和train_state元数据；各run最新保存为29999 |
| 归一化 | 原assets中的`libero_multiview/norm_stats.json`存在，沿用原统计 |
| CLI | 三个脚本均可解析；原run目录、resume、resume_data_loader、BS64、FSDP4、32000步及保留规则一致 |
| 学习率 | step30000/31000/31999均为2.5e-6 |
| 数据恢复 | 3项CPU测试通过，6项未选择；涵盖0/2 worker恢复顺序和不读取已跳过样本 |
| 调度 | bash语法及3次`sbatch --test-only`通过；实际3个job资源与日志路径已核对 |

## 路径

| 项目 | 路径 |
|---|---|
| checkpoint | [协议中的三个原run目录](protocol.md#来源与输出)；已有内容保留，新增31000/31999 |
| leftview日志 | `/opt/liutong/openpi-5090-research/sft-multiview-bs64-resume-32k/slurm-log/pi05-lv-sft-bs64-r32k_149303.out` |
| rightview日志 | `/opt/liutong/openpi-5090-research/sft-multiview-bs64-resume-32k/slurm-log/pi05-rv-sft-bs64-r32k_149304.out` |
| topview日志 | `/opt/liutong/openpi-5090-research/sft-multiview-bs64-resume-32k/slurm-log/pi05-tv-sft-bs64-r32k_149305.out` |

沿用原SFT周期保存规则，31000包含31001个优化步，31999为精确32000步。
未提交额外评测，未修改其他运行或排队任务，未移动、复制或删除数据和checkpoint。
