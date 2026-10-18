# MV-SV-KD backview 30K全量验证

## 固定协议

目的：测量第三方方法pi0.5适配版的30K成功率，与相同视角、BS64和步数的
SFT、H9-Fixed、H19比较。本次为既定30K终点验证，不搜索checkpoint。
保持原方法适配边界，不称为原论文完整复现；单训练seed，不推断跨seed稳定性。

| 项目 | 设置 |
|---|---|
| 模型 | 训练142275的29999，含30000个优化步；模型已完整保存 |
| 推理配置 | `pi05_libero_backview_lora`；普通学生LoRA，不加载teacher，无ACPD网络或注入 |
| 视角 | `backview_image`；fixed_dataset tuned相机位姿 |
| 回合 | Spatial/Object/Goal/LIBERO-10各10任务×50回合，共2000回合 |
| 推理 | seed7、224×224、replan5、等待10步，保存视频；沿用SFT/H9验证器 |
| GPU任务 | 数组0–3，每项1GPU/8CPU/24G/8小时，无并发上限；batch自动选节点，排除gpu03 |
| 汇总 | afterok依赖整个数组，1CPU/2G/15分钟，无GPU |
| 环境 | conda openpi推理、openpi-libero仿真，Hugging Face离线，清除代理 |
| 判断 | 四套各500回合且日志与汇总一致才报告最终成功率；未计算配对区间时不作显著性判断 |

复用`run_fixed_backview_benchmark_array.sbatch`及
`summarize_fixed_backview_benchmark.sbatch`，不新增验证代码。
协议先commit再提交；不移动、复制或删除checkpoint，不改其他活动任务或预约。

## 路径

```text
checkpoint:
/opt/liutong/openpi_checkpoints/fixed_dataset/baseline/mv_sv_kd/backview_bs64_30k/pi05_libero_backview_mv_sv_kd/pi05_libero_backview_mv_sv_kd_lora_fsdp4_bs64_30k/29999

evaluation:
/opt/liutong/openpi-5090-evals/mv-sv-kd-backview-bs64-30k/29999
```

验证目录中`logs/{spatial,object,goal,libero_10}.log`为原始日志，
`{suite}/videos/`为视频，`slurm-log/`为调度日志，`summary.txt`为最终汇总。

同进度参考结果：SFT30K为60.45%、H9-Fixed30K为58.00%、H19-FeatureFusion30K
为60.10%；均四套各500回合，证据索引见`EXPERIMENT_RESULTS.md`。
本模型尚无成功率结论。

## 实际提交

协议commit：`9d656ef`。2026-10-18 10:41提交，提交前checkpoint完成标记、
params元数据、norm stats与CPU配置导入检查通过，action_horizon10、action_dim32；
三个复用脚本bash语法及两项Slurm test-only通过。未另申请GPU smoke。

| 任务 | Job | 提交后状态 |
|---|---|---|
| 四套全量验证 | `152012_[0-3]` | 等待Priority；每项1GPU/8CPU/24G/8小时，无并发上限 |
| 自动汇总 | `152013` | afterok等待152012整个数组；1CPU/2G/15分钟 |

Slurm确认未绑定节点或预约，排除gpu03；可自动使用可用的磁性预约。
当时gpu04八张GPU全部分配，验证尚未开始；不能将成功提交视为已开始仿真。
未修改其他活动任务或checkpoint。日志与视频均使用上述独立目录。
