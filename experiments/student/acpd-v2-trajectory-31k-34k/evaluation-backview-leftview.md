# ACPD-v2-Fixed backview/leftview 31K–34K全量验证

## 目的与固定协议

补齐两个视角在30K–35K之间的成功率轨迹。沿用原H9推理配置，不修改模型，
不关闭任何contribution分支或gate注入。这是探索性checkpoint比较，尚无结果。
训练来源见[补训协议](protocol.md)；不宣称逐位复现历史35K轨迹。

| 项目 | 设置 |
|---|---|
| 模型 | backview、leftview各31K/32K/33K/34K，对应30999/31999/32999/33999 |
| 配置 | `pi05_libero_{view}_acpd_v2_layer10_lora`；相机键`{view}_image`；不使用teacher推理 |
| 相机 | fixed_dataset tuned相机，与原训练和已有验证一致 |
| 回合 | Spatial/Object/Goal/LIBERO-10各500回合；每任务50回合，共2000回合/模型 |
| 推理 | seed7，224×224，replan5，等待10步；保留视频 |
| 资源 | batch、gpu04、res_gpu04；每模型数组0-3，无数组并发上限；每项1GPU/8CPU/24G/24小时 |
| 汇总 | 每模型1CPU/2G/15分钟；afterok等待该模型四个子任务全部成功 |
| 34K等待 | backview依赖训练148740成功，leftview依赖148741成功；依赖满足前不占GPU |
| 环境 | conda openpi推理、openpi-libero仿真；HF离线、清除代理；命名环境位于`/home/liutong/miniconda3/envs` |

复用`examples/libero/eval_slurm/run_fixed_backview_benchmark_array.sbatch`和
`summarize_fixed_backview_benchmark.sbatch`，显式传入视角配置和相机键，不修改共享脚本。
31K–33K的完成标记、params元数据及`assets/libero_multiview/norm_stats.json`已核对；
34K尚未保存，提交训练成功依赖。预约当前截至2026-10-16 12:48 CST，
Slurm按资源与预约剩余窗口排队，不保证八个模型均能在窗口内启动。
不停止、修改或提前占用现有训练；不复制、移动、覆盖或删除数据与checkpoint。

## 路径

```text
checkpoint:
/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/trajectory_31k_34k/{view}/pi05_libero_{view}_acpd_v2_layer10/pi05_libero_{view}_acpd_v2_layer10_lora_fsdp4_bs64_resume_34k/{step}

evaluation:
/opt/liutong/openpi-5090-evals/acpd-v2-trajectory-31k-34k/{view}/{step}/
```

每模型结果目录下：`logs/`为四套原始日志，`{suite}/videos/`为视频，
`slurm-log/`为Slurm日志，`summary.txt`为最终四套及pooled计数。
以四套各500回合完成标记为完整结果，未完成不报告最终成功率。

## 结果使用

分别与同视角已有30K/35K结果比较；各模型保留原始成功/总数，不合并不同checkpoint的回合。
选择31K–35K中的最高观测点属于探索性选择，不等同于独立测试的最优模型。
尚无新增成功率结论。
