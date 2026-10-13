# Original ACPD 30K全量验证

本协议在提交验证任务前锁定；不修改模型、checkpoint或现有训练任务。

| 项目 | 固定设置 |
|---|---|
| 目的 | 测量论文原版ACPD backview学生在30K的闭环成功率 |
| 模型 | layers6+12、BS32旧版ACPD续训147454的最终29999；辅助cue网络不参与推理 |
| checkpoint | `/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_original/backview_bs32_30k/pi05_libero_backview_acpd_original_layers6_12/pi05_libero_backview_acpd_original_lora_fsdp4_bs32_30k/29999` |
| 推理配置 | `pi05_libero_backview_lora`；backview单视角，无wrist输入或ACPD-v2注入 |
| 相机 | fixed_dataset对应的`task_operation_cameras.yaml`；复用现有多视角验证器 |
| 回合 | Spatial、Object、Goal、LIBERO-10各10任务×50回合，共2000回合；seed7 |
| 动作执行 | 图像224×224，replan5，等待10步；其余使用既有正式验证默认值 |
| 调度 | gpu04的res_gpu04预约；单卡数组0-3，不限制并行；每项8CPU、24G、24小时 |
| 汇总 | 四项全部成功后运行1CPU、2G、20分钟汇总；不完整评测不生成正式pooled结果 |
| 环境 | policy使用conda openpi，仿真使用conda openpi-libero；Hugging Face离线且清除代理 |
| 日志和视频 | `/opt/liutong/openpi-5090-evals/acpd-original-backview-bs32-30k/29999/`，按suite归档 |
| 脚本 | `examples/libero/eval_slurm/run_fixed_backview_benchmark_array.sbatch`与`summarize_fixed_backview_benchmark.sbatch` |
| 比较限制 | 原版训练BS32，H9/H19与SFT正式对照为BS64；对比须注明，不归为仅改变方法的匹配实验 |

模型加载默认移除多余参数，保留学生LoRA和原动作网络；历史旧版5K验证使用同一
推理配置。最终checkpoint完成标记与归一化统计已核对。本协议不预设成功率方向，
尚无30K验证结论。

## 实际提交

| 项目 | 实际记录 |
|---|---|
| 协议提交 | `ca45855`，早于Slurm提交 |
| 验证任务 | `147939_[0-3]`，2026-10-13 22:24:13在gpu04同时启动；22:26四套均已加载29999模型、连接policy server并进入首回合 |
| 汇总任务 | `147944`，依赖`afterok:147939`；四项全部成功后自动生成`summary.txt` |
| 检查 | 三份复用shell脚本语法通过；checkpoint完成标记、norm stats存在；Slurm资源、预约、输出目录与依赖核对一致 |
| 现有训练 | topview147628和MV-SV-KD142275保留，不修改任务或checkpoint |
| 状态 | 尚无验证成功率；日志中ROCm/TPU不可用提示不表示CUDA验证失败 |
