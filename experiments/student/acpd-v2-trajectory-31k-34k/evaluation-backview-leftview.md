# ACPD-v2-Fixed backview/leftview 31K–34K全量验证

## 目的与固定协议

补齐两个视角在30K–35K之间的成功率轨迹。沿用原H9推理配置，不修改模型，
不关闭任何contribution分支或gate注入。这是探索性checkpoint比较，已完成结果见文末。
训练来源见[补训协议](protocol.md)；不宣称逐位复现历史35K轨迹。

| 项目 | 设置 |
|---|---|
| 模型 | backview、leftview各31K/32K/33K/34K，对应30999/31999/32999/33999 |
| 配置 | `pi05_libero_{view}_acpd_v2_layer10_lora`；相机键`{view}_image`；不使用teacher推理 |
| 相机 | fixed_dataset tuned相机，与原训练和已有验证一致 |
| 回合 | Spatial/Object/Goal/LIBERO-10各500回合；每任务50回合，共2000回合/模型 |
| 推理 | seed7，224×224，replan5，等待10步；保留视频 |
| 资源 | batch、gpu04、res_gpu04；每模型数组0-3，无数组并发上限；每项1GPU/8CPU/24G/12小时 |
| 汇总 | 每模型1CPU/2G/15分钟；afterok等待该模型四个子任务全部成功 |
| 34K等待 | backview依赖训练148740成功，leftview依赖148741成功；依赖满足前不占GPU |
| 环境 | conda openpi推理、openpi-libero仿真；HF离线、清除代理；命名环境位于`/home/liutong/miniconda3/envs` |

复用`examples/libero/eval_slurm/run_fixed_backview_benchmark_array.sbatch`和
`summarize_fixed_backview_benchmark.sbatch`，显式传入视角配置和相机键，不修改共享脚本。
31K–33K的完成标记、params元数据及`assets/libero_multiview/norm_stats.json`已核对；
34K尚未保存，提交训练成功依赖。验证脚本默认8小时，本次设12小时留余量，
同时避免24小时请求过早超出预约剩余窗口。预约当前截至2026-10-16 12:48 CST，
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
完整结果见文末；未完成模型不作成功率结论。

## 实际提交

协议commit：`5c7243b`；排队时限修订commit：`c2aba83`，均早于Slurm提交。
2026-10-15 09:07 CST提交并核对如下；所有汇总任务均afterok依赖对应四套验证成功。

| Student | Checkpoint | 单卡数组 | 自动汇总 | 提交后状态 |
|---|---|---|---:|---|
| backview | 31K / 30999 | 148914_[0-3] | 148915 | 等待gpu04资源/预约窗口 |
| backview | 32K / 31999 | 148916_[0-3] | 148917 | 等待gpu04资源/预约窗口 |
| backview | 33K / 32999 | 148918_[0-3] | 148919 | 等待gpu04资源/预约窗口 |
| backview | 34K / 33999 | 148920_[0-3] | 148921 | afterok等待训练148740 |
| leftview | 31K / 30999 | 148922_[0-3] | 148923 | 等待gpu04资源/预约窗口 |
| leftview | 32K / 31999 | 148924_[0-3] | 148925 | 等待gpu04资源/预约窗口 |
| leftview | 33K / 32999 | 148926_[0-3] | 148927 | 等待gpu04资源/预约窗口 |
| leftview | 34K / 33999 | 148928_[0-3] | 148929 | afterok等待训练148741 |

共32个单卡验证子任务、8个CPU汇总任务，共16000回合；尚未启动，尚无结果。
提交前bash语法与Slurm test-only检查通过；31K–33K三个小型元数据文件逐一解析通过。
提交后核对gpu04预约、每项1GPU/8CPU/24G/12小时、数组0-3及34K训练依赖。
09:07时148740/148741均在约33.7K继续训练，148742仍等待资源，MV-SV-KD142275继续运行；
未修改上述任务，也未新增checkpoint复制或后台轮询任务。

## 已完成结果

2026-10-15 17:33 CST核对，原始日志与自动汇总一致。每套500回合。

| Student / step | Spatial | Object | Goal | LIBERO-10 | Pooled |
|---|---:|---:|---:|---:|---:|
| backview 31K / 30999 | 342/500（68.40%） | 373/500（74.60%） | 285/500（57.00%） | 182/500（36.40%） | 1182/2000（59.10%） |
| backview 32K / 31999 | 338/500（67.60%） | 399/500（79.80%） | 286/500（57.20%） | 150/500（30.00%） | 1173/2000（58.65%） |
| backview 33K / 32999 | 341/500（68.20%） | 374/500（74.80%） | 339/500（67.80%） | 155/500（31.00%） | 1209/2000（60.45%） |
| leftview 31K / 30999 | 415/500（83.00%） | 417/500（83.40%） | 417/500（83.40%） | 371/500（74.20%） | 1620/2000（81.00%） |

证据：`/opt/liutong/openpi-5090-evals/acpd-v2-trajectory-31k-34k/backview/{30999,31999,32999}/summary.txt`，
以及同目录`logs/{spatial,object,goal,libero_10}.log`的`Final success rate`。
leftview31K证据：`/opt/liutong/openpi-5090-evals/acpd-v2-trajectory-31k-34k/leftview/30999/summary.txt`，
以及同目录四套日志。
backview31K/32K/33K分别比历史35K的61.05%低1.95/2.40/0.60点；未计算配对区间，不作显著性判断。

leftview31K比已有35K的1593/2000=79.65%高1.35点（多27次成功）；
Spatial/Object/Goal/LIBERO-10分别高1.40/1.80/1.00/1.20点。
35K参考：`/opt/liutong/openpi-5090-evals/acpd-v2-multiview-35k/leftview/34999/summary.txt`。
这是相同评测协议下的观测差值，尚未计算配对区间；探索性checkpoint选择不等于独立测试的最优模型。

leftview32K/33K及backview34K验证中，leftview34K排队，尚无完整结果。
backview/leftview34K checkpoint均已完整保存，原训练依赖已满足。
