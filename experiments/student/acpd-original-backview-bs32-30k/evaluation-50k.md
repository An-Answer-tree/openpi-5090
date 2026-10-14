# Original ACPD 50K 全量验证

本协议在提交验证前锁定；复用既有验证器，不修改训练或 checkpoint。

| 项目 | 设置 |
|---|---|
| 目的 | 测量原论文 ACPD backview 学生延长训练至50K后的闭环成功率 |
| 模型 | layers6+12、BS32旧版ACPD；任务147971的49999；14:28:38最终保存完成 |
| 推理 | `pi05_libero_backview_lora`，backview单视角；辅助cue网络不参与推理，无ACPD-v2注入 |
| 相机 | fixed_dataset对应的`task_operation_cameras.yaml`，与30K评测一致 |
| 回合 | Spatial、Object、Goal、LIBERO-10各10任务×50回合，共2000回合；seed7 |
| 执行 | 图像224×224，replan5，等待10步；其余沿用正式验证默认值 |
| 调度 | gpu04、res_gpu04预约；单卡数组0-3，不限制并行；每项8CPU、24G、24小时 |
| 汇总 | 四项成功后运行1CPU、2G、20分钟汇总；只接受完整2000回合结果 |
| 环境 | conda openpi提供policy、openpi-libero运行仿真；离线并清除代理；HF缓存显式位于`/opt/liutong/huggingface` |
| 结果、日志和视频 | `/opt/liutong/openpi-5090-evals/acpd-original-backview-bs32-50k/49999/`，按suite归档 |
| 脚本 | `examples/libero/eval_slurm/run_fixed_backview_benchmark_array.sbatch`及`summarize_fixed_backview_benchmark.sbatch` |
| 比较限制 | 原ACPD为BS32，正式H9/H19/SFT为BS64；不是仅改变方法的匹配对照。30K评测提前停止，不能用部分结果判断50K增益 |

checkpoint：
`/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_original/backview_bs32_30k/pi05_libero_backview_acpd_original_layers6_12/pi05_libero_backview_acpd_original_lora_fsdp4_bs32_30k/49999`。

完成标记和`assets/libero_multiview/norm_stats.json`已核对。不移动、复制或删除模型；
不修改topview147628或MV-SV-KD142275。50K尚无验证结论。
