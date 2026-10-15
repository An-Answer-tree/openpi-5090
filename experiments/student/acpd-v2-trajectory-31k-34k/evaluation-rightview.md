# ACPD-v2-Fixed rightview 31K–34K全量验证

## 目的与协议

补齐rightview在30K–35K之间的成功率轨迹，与已有35K的1586/2000（79.30%）比较。
这是探索性checkpoint比较，不修改模型、不关闭contribution或gate注入；
训练来源见[补训协议](protocol.md)，不宣称逐位复现历史35K轨迹。

| 项目 | 固定设置 |
|---|---|
| Checkpoint | 31K/32K/33K/34K，对应30999/31999/32999/33999，均完整保存 |
| 推理配置 | `pi05_libero_rightview_acpd_v2_layer10_lora`；相机键`rightview_image`；不使用teacher推理 |
| 相机 | fixed_dataset tuned相机，与训练和原H9验证一致 |
| 回合 | Spatial/Object/Goal/LIBERO-10各500回合；每任务50回合；2000回合/模型，共8000回合 |
| 推理 | seed7，224×224，replan5，等待10步；保留视频 |
| GPU任务 | batch、gpu04、res_gpu04；每模型数组0-3，无数组并发上限；每项1GPU/8CPU/24G/8小时 |
| 汇总 | 每模型1CPU/2G/15分钟；afterok等待对应四个GPU子任务全部成功 |
| 环境 | conda openpi推理、openpi-libero仿真；HF离线、清除代理 |

复用`examples/libero/eval_slurm/run_fixed_backview_benchmark_array.sbatch`和
`summarize_fixed_backview_benchmark.sbatch`，显式传入rightview配置和相机键。
数组0/1/2/3分别运行Spatial/Object/Goal/LIBERO-10，无额外GPU smoke。
提交前核对四个checkpoint的完成标记、params元数据及norm stats，运行bash语法和Slurm试提交检查。
预约截至2026-10-16 12:48 CST；8小时请求沿用现有验证脚本默认值，启动时间仍由Slurm决定。
不修改现有任务，不移动、复制、覆盖或删除数据及checkpoint。

## 路径

```text
checkpoint:
/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/trajectory_31k_34k/rightview/pi05_libero_rightview_acpd_v2_layer10/pi05_libero_rightview_acpd_v2_layer10_lora_fsdp4_bs64_resume_34k/{30999,31999,32999,33999}

evaluation:
/opt/liutong/openpi-5090-evals/acpd-v2-trajectory-31k-34k/rightview/{step}/
```

每模型目录下：`logs/`为四套原始日志，`{suite}/videos/`为视频，
`slurm-log/`为Slurm日志，`summary.txt`为最终汇总。
已有35K参考：`/opt/liutong/openpi-5090-evals/acpd-v2-multiview-35k/rightview/34999/summary.txt`。

## 判定与记录

每个checkpoint必须完成四套各500回合，核对原始计数与汇总一致后，才报告最终pooled成功率。
保留各套成功/总数，不合并不同checkpoint回合。31K–35K最高观测点属于探索性选择，
不等同于独立测试的最优模型；未计算区间时不作显著性判断。
提交后在本文件记录真实Slurm任务号及检查状态，完成后同步结果台账。
当前尚未提交，尚无新增结果。
