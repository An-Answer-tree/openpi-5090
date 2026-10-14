# 三视角31K–34K补训执行记录

协议：[protocol.md](protocol.md)，提交前锁定commit `9f3c9cc`。
提交时间：2026-10-14 22:52:53 CST；检查时间：2026-10-14 23:01 CST。

| Student | 任务 | 实际状态 | 恢复与训练证据 |
|---|---|---|---|
| backview | 148740 | gpu04运行，首步训练通过 | 22:54:54恢复optimizer step30000；22:55:32接续data batch30000；22:57:59首步loss=0.0791，BS64、累计1，随后持续推进 |
| leftview | 148741 | 等待Resources | 尚未启动，尚无GPU恢复证据 |
| rightview | 148742 | 等待ReqNodeNotAvail / May_be_reserved_for_other_job | 尚未启动，尚无GPU恢复证据 |

三个任务均为4GPU/32CPU/96G/24小时，batch分区，gpu04的`res_gpu04`预约。
预约到2026-10-16 12:48结束；Slurm排队预测不是确定启动时间。
没有改变其他任务的资源、优先级或预约。

## 配置与检查

复用H9-Fixed、layer10、flow/contribution/ACL=1.0/0.2/0.5、seed42，
physical BS64、累计1，30K后LR保持2.5e-6。从完整30K状态恢复模型与Adam，
数据接续batch30000。目标34000，每1K保存、`keep_period=1`，
四个新增目录为`30999/31999/32999/33999`。

| 检查 | 实际结果 |
|---|---|
| 三个脚本bash语法与git diff检查 | 通过 |
| 三个实际命令CLI解析与保存边界 | BS、累计、FSDP、loss、LR、源路径、数据接续与四个保存点一致 |
| Slurm test-only与提交后资源 | 接受配置；每任务实际请求4GPU/32CPU/96G/24小时 |
| CPU四虚拟设备恢复、保存及数据接续测试 | `scripts/distillation_acpd/train_distill_test.py`筛选`checkpoint_schedule or resume_data_loader or external_resume`：8 passed、24 deselected |
| backview真实GPU恢复与首步 | 已通过；left/right尚待资源 |

上述CPU测试不加载大型真实checkpoint，也没有申请额外GPU smoke。
checkpoint源元数据按明确路径核对；没有递归扫盘。
源checkpoint只读加载，新输出独立；没有复制、移动、覆盖或删除原checkpoint与数据。
已有35K不变；本次不重复训练35K、不提交新增验证，尚无成功率结论。
这是30K分支重建，不宣称逐位复现历史35K运行。

## 路径

三视角精确源路径和脚本见[协议](protocol.md#固定配置)。新checkpoint完整目录模板：

```text
/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/trajectory_31k_34k/{view}/pi05_libero_{view}_acpd_v2_layer10/pi05_libero_{view}_acpd_v2_layer10_lora_fsdp4_bs64_resume_34k/{30999,31999,32999,33999}
```

日志根目录：`/opt/liutong/openpi-5090-research/acpd-v2-trajectory-31k-34k/slurm-log/`。

| Student | 日志文件 |
|---|---|
| backview | `pi05-bv-acpdv2-bs64-r34k_148740.out` |
| leftview | `pi05-lv-acpdv2-bs64-r34k_148741.out` |
| rightview | `pi05-rv-acpdv2-bs64-r34k_148742.out` |
