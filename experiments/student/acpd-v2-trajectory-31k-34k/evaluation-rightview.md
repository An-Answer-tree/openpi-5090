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

## 实际提交

协议commit：`69d82bd`，早于Slurm提交。2026-10-15 20:19 CST提交并核对：

| Checkpoint | 四套单卡数组 | 自动汇总 | 提交后状态 |
|---|---|---:|---|
| 31K / 30999 | 149171_[0-3] | 149172 | 四套均已在gpu04启动 |
| 32K / 31999 | 149173_[0-3] | 149174 | Spatial/Object已启动，Goal/LIBERO-10等待资源 |
| 33K / 32999 | 149175_[0-3] | 149176 | 等待gpu04资源/预约窗口 |
| 34K / 33999 | 149177_[0-3] | 149178 | 等待gpu04资源/预约窗口 |

四个汇总分别afterok依赖对应整个数组，未占用GPU。checkpoint三个小型元数据文件逐一解析通过，
bash语法与Slurm test-only检查通过；实际任务的预约、资源、8小时时限和汇总依赖已核对。
20:21核对六个已启动子任务均完成模型加载、建立策略连接并进入首个仿真回合；尚无新增成功率结果。
gpu04此时8卡均已分配，其中新验证6卡，原backview/leftview34K验证各1卡；
MV-SV-KD仍在gpu02运行。未修改任何已有任务；没有扫描大型目录或操作checkpoint。

## 进度快照

2026-10-15 20:53 CST：31K/32K共八个子任务均在gpu04运行，33K/34K等待Priority。
以下仅统计各日志已经结束的回合，不是最终结果；各套当前任务构成不同，不能直接比较checkpoint。

| Checkpoint | Spatial | Object | Goal | LIBERO-10 | 已完成/计划回合 |
|---|---|---|---|---|---|
| 31K | 200/234 | 188/219 | 149/211 | 38/89 | 753/2000 |
| 32K | 223/247 | 196/223 | 164/208 | 23/42 | 720/2000 |

四套数字为成功/已完成回合，尚无完整成功率结论。

## 已完成结果

2026-10-15 23:43 CST核对四套原始日志与`summary.txt`，计数一致。
以下四套各500回合，总体各2000回合；35K为已有完整结果。

| Checkpoint | Spatial | Object | Goal | LIBERO-10 | 总体 |
|---|---|---|---|---|---|
| 31K | 426/500（85.20%） | 445/500（89.00%） | 407/500（81.40%） | 285/500（57.00%） | 1563/2000（78.15%） |
| 32K | 451/500（90.20%） | 443/500（88.60%） | 429/500（85.80%） | 297/500（59.40%） | 1620/2000（81.00%） |
| 33K | 415/500（83.00%） | 434/500（86.80%） | 421/500（84.20%） | 282/500（56.40%） | 1552/2000（77.60%） |
| 35K（已有参考） | 418/500（83.60%） | 450/500（90.00%） | 416/500（83.20%） | 302/500（60.40%） | 1586/2000（79.30%） |

31K–33K的自动汇总149172/149174/149176已生成。32K是目前rightview最高观测点，
比已有35K高1.70点；四套差值依次为+6.60/−1.40/+2.60/−1.00点。
未计算配对区间，不作显著性判断；checkpoint选择是探索性的。
SFT30K为77.00%，与32K差4.00点，但训练进度不同，不能作为同进度方法收益。

证据：上述`evaluation`根路径下`30999/31999/32999/summary.txt`及各自`logs/`；
35K参考路径见“路径”一节。

34K尚未完成：Spatial/Object/Goal为410/452/408成功、各500回合；
LIBERO-10为142/245（临时），合计已完成1745/2000回合。
仅149177_3仍在gpu04运行，149178等待依赖；不报告34K最终总体成功率。
