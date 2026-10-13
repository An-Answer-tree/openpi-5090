# Original ACPD 5K→30K执行记录

核对时间：2026-10-13 00:24 CST。

| 项目 | 实际记录 |
|---|---|
| 协议提交 | `7965373`，先锁定方法、接续和保存设置，再提交训练 |
| 训练任务 | `147454`，PENDING，Resources；尚未开始GPU恢复或训练 |
| 节点与预约 | `ReqNodeList=gpu04`，`Reservation=res_gpu04`；预约截至2026-10-16 12:48 CST |
| 资源 | 4 GPU、32 CPU、96G、时限8天；BS32，累计1 |
| 原地调度修改 | `scontrol update JobId=147454 ReqNodeList=gpu04 ReservationName=res_gpu04`；保留任务号，不取消重提 |
| 现有任务 | H19 `147300`与topview `147339`仍在gpu04运行，各4 GPU；第三方baseline `142275`仍在gpu02运行；H19验证`147342/147343`依赖不变 |
| 检查 | shell语法、CLI帮助、保存与接续规则测试通过；6 passed、2 skipped；单设备CPU检查未覆盖实际跨GPU分片恢复 |
| 来源 | 完整5K目录存在；只读恢复，路径见[协议](protocol.md)；没有移动、复制或删除checkpoint及数据 |
| 保存 | 10K/15K/20K/25K和26K–30K每1K全部保留，`keep_period=1` |
| 日志 | `/opt/liutong/openpi-5090-research/acpd-original-backview-bs32-30k/slurm-log/pi05-bv-acpd-original-r30k_147454.out`；父目录已在启动前创建 |
| 结论 | 尚无30K训练或验证结果 |

Slurm当前按已有任务时限估算开始时间，不能据此认定实际等待时长。

## 30K训练完成复核

核对时间：2026-10-13 21:41 CST；未修改任务或checkpoint。

| 项目 | 实际记录 |
|---|---|
| 训练完成 | 日志21:24:51达到30K/30K；21:25:52最终保存完成且后台保存无错误 |
| 最终checkpoint | `/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_original/backview_bs32_30k/pi05_libero_backview_acpd_original_layers6_12/pi05_libero_backview_acpd_original_lora_fsdp4_bs32_30k/29999` |
| 保存证据 | 最终目录`_CHECKPOINT_METADATA`存在，时间21:25:52；日志`CheckpointManager Save Finalize is done on all hosts` |
| Slurm资源 | 147454已不在队列；gpu04共8卡，仅topview147628分配4卡，另4卡空闲 |
| 当前验证 | 无运行或排队的验证任务；尚无30K成功率结论 |

## 30K→50K提交

核对时间：2026-10-13 23:51 CST。

| 项目 | 实际记录 |
|---|---|
| 协议与脚本 | [50K协议](resume-50k.md)，git `c56ec50`，早于Slurm提交 |
| 停止评测 | 147939四个子任务及147944汇总已取消，部分日志与视频保留 |
| 训练任务 | 147971，23:37:17在gpu04启动，res_gpu04预约；23:38:44确认参数与Adam恢复至optimizer step30000 |
| 资源 | 4GPU、32CPU、96G，时限2天，BS32、累计1 |
| 续训 | 原29999、优化器step30000→50000；数据从batch30000接续，原LR在30K后保持2.5e-6 |
| 保存 | 31K–50K每1K，keep_period=1全部保留，保持原checkpoint目录 |
| 检查 | bash语法、完整CLI合并、保存序列和30K后LR检查通过；仅改变训练终点与恢复来源 |
| 当前阶段 | 数据初始化尚未完成，首步loss与推进未确认；只读进程检查显示`Dl/folio_wait_bit_common`，当前在文件读取等待；没有修改或重新提交任务 |
| 日志 | `/opt/liutong/openpi-5090-research/acpd-original-backview-bs32-50k/slurm-log/pi05-bv-acpd-original-r50k_147971.out` |
| 其他训练 | topview147628仍在gpu04运行，MV-SV-KD142275仍在gpu02运行；未修改任务或checkpoint |
| 结论 | 50K尚无结论；30K评测未完成，不将临时值记为正式成功率 |
