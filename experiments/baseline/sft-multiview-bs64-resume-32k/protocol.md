# SFT-View：left/right/topview BS64 30K→32K

## 目的与边界

补齐三个单视角 SFT 的32K基线，与相同视角的 ACPD-v2 32K比较。
本次只提交续训，不提交仿真验证。ACPD的32K是查看轨迹后选择的探索性点；
补齐相同步数基线不能消除checkpoint选择带来的偏差，任务成功率须由后续验证确定。

## 固定配置

| 项目 | 设置 |
|---|---|
| 视角 | leftview、rightview、topview，各一个独立任务 |
| 数据集 | fixed_dataset：`/opt/liutong/libero_multiview_tuned_6view_lerobot` |
| 模型 | 原pi0.5单视角LoRA；action horizon 10；保留原模型配置 |
| 资源 | 每任务4×5090 FSDP、32 CPU、96G内存；时限24小时 |
| 调度 | batch分区自动选择节点；不指定节点、reservation或exclude |
| Batch | physical global BS64，每卡16，无梯度累积 |
| Seed | 42 |
| 恢复 | 原29999完整模型、Adam状态及优化步数；数据从batch30000接续 |
| 学习率 | 原warmup1000、peak2.5e-5、cosine decay30000、末值2.5e-6；不重启warmup |
| 终点 | 优化器step32000 |
| 保存 | save_interval=1000、keep_period=1；保留31K附近和32K，以及全部已有checkpoint |
| 环境 | conda openpi，HF离线，清空代理；HF缓存位于/opt/liutong |

原SFT训练器按零起始循环step触发周期保存：31K附近目录为`31000`
（内含optimizer step31001）；32K最终目录为`31999`（optimizer step32000）。
这是沿用原SFT的周期保存规则，不修改训练代码。checkpoint仍写在各自原run目录中，
不移动、复制或删除已有checkpoint。最终以日志中的恢复步数及新checkpoint完整标记核验。

## 来源与输出

| 视角 | 原run目录；源为其29999，新增31000/31999 |
|---|---|
| leftview | `/opt/liutong/openpi_checkpoints/fixed_dataset/sft/leftview_bs64_30k/pi05_libero_leftview_lora/pi05_libero_leftview_lora_fsdp4_bs64_30k` |
| rightview | `/opt/liutong/openpi_checkpoints/fixed_dataset/sft/rightview_bs64_30k/pi05_libero_rightview_lora/pi05_libero_rightview_lora_fsdp4_bs64_30k` |
| topview | `/opt/liutong/openpi_checkpoints/fixed_dataset/sft/topview_bs64_30k/pi05_libero_topview_lora/pi05_libero_topview_lora_fsdp4_bs64_30k` |

日志：`/opt/liutong/openpi-5090-research/sft-multiview-bs64-resume-32k/slurm-log/%x_%j.out`。

脚本：`scripts/train_slurm/pi05_libero_{leftview,rightview,topview}_lora_fsdp4_bs64_resume_32k.sbatch`。

## 提交前检查

核对三个源checkpoint完成标记及params/train_state元数据、原归一化统计、
CLI配置和shell语法；执行已有数据顺序恢复CPU测试与Slurm试提交。
只检查明确路径和Slurm，不扫描大型数据、checkpoint或日志目录。
