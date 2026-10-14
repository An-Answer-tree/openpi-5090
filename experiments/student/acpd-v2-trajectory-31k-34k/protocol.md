# ACPD-v2-Fixed：补齐 backview/leftview/rightview 的31K–34K

## 目的与判定

三个视角已有35K模型。本次从各自30K完整状态重新接续，只补齐31/32/33/34K，
不新增方法、不重复训练35K。通过条件是恢复optimizer step和数据batch均为30000，
训练正常推进，四个新增checkpoint全部完整保存。尚不提交验证任务，尚无成功率结论。

## 固定配置

| 项目 | 设置 |
|---|---|
| 方法 | H9 ACPD-v2-Fixed，layer 10，两路exact contribution预测及原gate注入 |
| Teacher | `/opt/liutong/openpi_checkpoints/fixed_dataset/sft/agentview_wrist/29999/params` |
| 数据 | `/opt/liutong/libero_multiview_tuned_6view_lerobot`；复用fixed_dataset资产 |
| 资源 | gpu04预约`res_gpu04`，batch，4×5090 FSDP，32 CPU、96G，每任务24小时 |
| Batch | physical global BS64，梯度累计1 |
| 恢复 | `--resume-from`只读加载模型、Adam状态及step；seed42，`--resume-data-loader`接续batch30000 |
| Loss | flow/contribution/ACL = 1.0/0.2/0.5，不改变网络和损失 |
| LR | 原30K cosine结束值`2.5e-6`，不重启warmup、不延长cosine周期 |
| 保存 | 目标34000，`save_interval=1000`、`keep_period=1`；保留`30999/31999/32999/33999` |
| 隔离 | 新目录`/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/trajectory_31k_34k/{view}/`；不复制、移动、覆盖或删除原checkpoint |
| 环境 | conda `openpi`；HF缓存在`/opt/liutong/huggingface`，联网关闭、代理清除 |
| 日志 | `/opt/liutong/openpi-5090-research/acpd-v2-trajectory-31k-34k/slurm-log/` |

| Student | 只读30K源checkpoint | 脚本 |
|---|---|---|
| backview | `/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/batch_scaling_30k/pi05_libero_backview_acpd_v2_layer10/pi05_libero_backview_acpd_v2_lora_fsdp4_bs64_30k/29999` | `scripts/train_slurm/pi05_libero_backview_acpd_v2_layer10_fsdp4_bs64_resume_34k.sbatch` |
| leftview | `/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/multiview_35k/leftview/pi05_libero_leftview_acpd_v2_layer10/pi05_libero_leftview_acpd_v2_layer10_lora_fsdp4_bs64_35k/29999` | `scripts/train_slurm/pi05_libero_leftview_acpd_v2_layer10_fsdp4_bs64_resume_34k.sbatch` |
| rightview | `/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/multiview_35k/rightview/pi05_libero_rightview_acpd_v2_layer10/pi05_libero_rightview_acpd_v2_layer10_lora_fsdp4_bs64_35k/29999` | `scripts/train_slurm/pi05_libero_rightview_acpd_v2_layer10_fsdp4_bs64_resume_34k.sbatch` |

复用现有完整状态恢复与保存实现，不申请GPU smoke任务。
提交前检查CLI解析、保存边界、保留策略和只读恢复，协议先commit再提交Slurm。
不停止或修改正在运行的验证与MV-SV-KD训练；资源由Slurm正常分配。

## 结果使用边界

这是30K分支的轨迹重建，不宣称与历史35K运行逐位相同。后续若验证，注明分支来源，
以相同视角、相同回合协议比较；不能把尚未评测的31K–34K写成优于现有35K的结论。
