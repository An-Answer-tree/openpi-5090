# ACL-only 31K–34K 补训执行记录

协议：[trajectory-31k-34k-protocol.md](trajectory-31k-34k-protocol.md)，
提交前锁定commit `8e96607`；提交时间2026-10-16 00:58:31 CST。

| 项目 | 实际状态 |
|---|---|
| 训练任务 | 149325，00:58:31在gpu04启动；00:59:38恢复optimizer step30000，00:59:50确认data batch30000；10:18进度约33.9K，最新完整loss记录为33800且数值有限 |
| 资源 | 4GPU、32CPU、96G、24小时；BS64、累计1、FSDP4 |
| 调度 | batch自动选节点；脚本未绑定节点或预约，Slurm自动关联磁性预约res_gpu04 |
| 已取消任务 | SFT续训149303/149304/149305；leftview已运行，right/top启动前取消；原checkpoint保留 |
| 训练目标 | 原ACL-only：flow=1、ACL=0.5、contribution=0、无注入，layer10 |
| 目标保存 | 原30K完整状态和batch30000接续；31K/32K/33K已完整保存，34K尚未产出；每1K全部保留 |
| 验证 | 31K–34K全量验证149556/149558/149560/149562已提交，各4个单卡子任务及1个汇总；前三项等待Priority，34K等待训练成功；尚无新增成功率结论，见[验证协议与任务](evaluation-31k-34k.md) |

## 已完成检查

| 检查 | 结果 |
|---|---|
| 实际CLI与原ACL-only比对 | 训练目标、teacher、数据、模型、BS、随机种子及LR均相同；只改变输出、终点、保存间隔和恢复选项 |
| 源checkpoint | 29999完整提交标记、params/train_state元数据及norm_stats存在 |
| 保存边界 | 实际配置准确保存30999/31999/32999/33999；keep_period=1，无overwrite |
| 数据与LR | data_start_batch=30000；30K–34K的LR保持2.5e-6，不重启warmup |
| CPU测试 | 4虚拟设备，筛选checkpoint_schedule/resume_data_loader/external_resume：8 passed、24 deselected |
| 脚本与调度 | bash语法、git diff检查、sbatch test-only和提交后资源核对通过 |

源路径与输出完整路径见协议。日志：
`/opt/liutong/openpi-5090-research/acpd-v2-acl-only-bs64-trajectory-31k-34k/slurm-log/pi05-bv-acl-only-bs64-r34k_149325.out`。
没有移动、复制或删除checkpoint与数据，没有改变其他活动任务。

2026-10-16 10:18 CST核对：31K–33K的完成标记、params/train_state元数据和
norm stats均存在且JSON可解析；34K四项文件尚未产出。10:07日志step33800的
supervised_loss=0.0177、action_corr_loss=0.0247、total_loss=0.0301，
contribution权重和注入gate均为0。训练仍正常推进；loss不代表任务成功率。
