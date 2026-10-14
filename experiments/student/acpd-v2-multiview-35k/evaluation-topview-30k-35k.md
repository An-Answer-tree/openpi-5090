# Topview ACPD-v2 30K / 35K 全量验证

本协议在Slurm提交前锁定；复用现有评测器，不修改模型或训练代码。

| 项目 | 设置 |
|---|---|
| 目的 | 测量topview学生30K与35K成功率；30K与相同视角BS64 SFT比较，35K观察延长训练效果 |
| 方法 | ACPD-v2-Fixed（H9），layer10，两路预测贡献与最终hidden的gate注入均保留 |
| 模型 | 任务147628所属原训练目录的`29999`与`34999`；完成标记、params元数据及norm stats已核对 |
| 推理配置 | `pi05_libero_topview_acpd_v2_layer10_lora`；`topview_image`，无teacher输入 |
| 相机 | fixed_dataset的`task_operation_cameras.yaml`，沿用原topview验证器 |
| 回合 | Spatial、Object、Goal、LIBERO-10各10任务×50回合；每模型2000回合，共4000回合；seed7 |
| 执行 | 图像224×224、replan5、等待10步；其余沿用既有全量验证默认值 |
| 调度 | gpu04、res_gpu04预约；每模型单卡数组0-3，无并发限制；每项8CPU、24G、24小时 |
| 汇总 | 每模型一个1CPU、2G、15分钟任务，afterok依赖对应数组；必须四套全部完成、共2000回合 |
| 环境 | conda openpi策略服务、openpi-libero仿真；显式CONDA_ENVS_PATH；离线、清除代理；HF缓存在`/opt/liutong/huggingface` |
| 输出 | `/opt/liutong/openpi-5090-evals/acpd-v2-multiview-35k/topview/{29999,34999}/`；logs为逐套日志、各套videos为视频、slurm-log为调度日志、summary.txt为最终汇总 |
| 脚本 | `examples/libero/eval_slurm/run_fixed_backview_benchmark_array.sbatch`及`summarize_fixed_backview_benchmark.sbatch`，显式传入topview配置与图像键 |

Checkpoint根目录：
`/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/multiview_35k/topview/pi05_libero_topview_acpd_v2_layer10/pi05_libero_topview_acpd_v2_layer10_lora_fsdp4_bs64_35k/`。

主要对照为已有topview SFT BS64 30K（71.55%）；30K为匹配训练步数比较，
35K对SFT 30K不是同进度比较。最终记录四套成功数及总成功率；复用逐回合日志
作探索性分层配对分析，不把回合重采样区间解释为跨训练seed稳定性。
不移动、复制或删除checkpoint；不影响MV-SV-KD任务142275。尚无验证结论。
