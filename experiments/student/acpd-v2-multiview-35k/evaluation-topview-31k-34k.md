# Topview ACPD-v2 31K–34K 全量验证

本协议在Slurm提交前锁定，复用[30K/35K协议](evaluation-topview-30k-35k.md)
及现有评测脚本，不修改模型、训练或推理代码。

| 项目 | 设置 |
|---|---|
| 目的 | 测量topview ACPD-v2在30K–35K之间的成功率轨迹 |
| 方法 | H9 / ACPD-v2-Fixed，layer10；两路贡献预测及最终hidden的gate注入均保留 |
| 模型 | 31K/32K/33K/34K，对应30999/31999/32999/33999；四套完成标记、params元数据及norm stats已核对 |
| 推理配置 | `pi05_libero_topview_acpd_v2_layer10_lora`，`topview_image`，不加载teacher |
| 相机与回合 | fixed_dataset的`task_operation_cameras.yaml`；Spatial/Object/Goal/LIBERO-10各10任务×50回合，seed7；每模型2000回合，共8000回合 |
| 推理 | 224×224、replan5、等待10步，与既有topview评测一致 |
| 调度 | batch，gpu04，res_gpu04；每模型单卡数组0-3，无并发限制、无旧验证依赖；每项1GPU/8CPU/24G/24小时 |
| 汇总 | 每模型1CPU/2G/15分钟，afterok依赖对应四项；只在2000回合全部完成后生成最终成功率 |
| 环境 | conda openpi与openpi-libero；`CONDA_ENVS_PATH=/home/liutong/miniconda3/envs`；HF/transformers离线，清除代理，HF缓存位于`/opt/liutong/huggingface` |
| 输出 | `/opt/liutong/openpi-5090-evals/acpd-v2-multiview-35k/topview/{30999,31999,32999,33999}/`；logs为逐套日志，各套videos为视频，slurm-log为调度日志，summary.txt为最终汇总 |
| 脚本 | `examples/libero/eval_slurm/run_fixed_backview_benchmark_array.sbatch`与`summarize_fixed_backview_benchmark.sbatch` |

Checkpoint根目录：
`/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/multiview_35k/topview/pi05_libero_topview_acpd_v2_layer10/pi05_libero_topview_acpd_v2_layer10_lora_fsdp4_bs64_35k/`。

这是查看30K/35K临时数据后增加的探索性轨迹测量，不是独立确认实验。
对照为topview SFT BS64 30K（71.55%）；31K–34K与其训练步数不同。
最终记录四套与pooled成功率；比较使用相同任务初始状态的完整逐回合日志。
选择本序列最高点不构成独立测试集或跨训练seed的稳定优势证据。

不移动、复制或删除checkpoint，不修改或停止现有30K/35K验证与MV-SV-KD训练。
尚无验证结论。

## 实际提交

协议commit：`07d84f3`，早于2026-10-14 18:42的Slurm提交。
18:43核对：四组已提交，等待gpu04资源；对应汇总等待afterok依赖。

| Checkpoint | 单卡验证数组 | 自动汇总 | 结果目录编号 |
|---|---|---:|---|
| 31K / 30999 | 148526_[0-3] | 148527 | 30999 |
| 32K / 31999 | 148528_[0-3] | 148529 | 31999 |
| 33K / 32999 | 148530_[0-3] | 148531 | 32999 |
| 34K / 33999 | 148532_[0-3] | 148533 | 33999 |

每组四个单卡任务无并发上限，共16个GPU任务与4个CPU汇总任务。
四组均通过脚本语法、checkpoint完成元数据与Slurm test-only检查。
30K/35K的八个验证任务仍在运行；未修改既有任务或checkpoint。
