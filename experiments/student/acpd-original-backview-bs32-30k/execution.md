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
