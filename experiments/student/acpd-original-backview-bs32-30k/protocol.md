# 原论文 ACPD：backview 5K→30K

## 目的

接续已完成的原版 ACPD 5K 实验，观察30K训练效果。此处“原版”指
`AAAI_2027_ACPD.pdf`方法部分的动作条件cue选择与预测，不是ACPD-v2真实attention
contribution预测或H19特征融合。长训练尚无结论；5K历史结果为6.50%。

## 方法对应

| 论文组件 | 当前实现 |
|---|---|
| Teacher privileged sequence | agentview+wrist视觉token与同层teacher action hidden，经各自MLP投影后拼接 |
| 动作条件cue选择 | 第6、12层student action hidden作为query；query输入stop-gradient，selector继续训练 |
| Cue预测 | 每层三层MLP从student hidden预测1024维cue；cosine距离匹配，selector与predictor共同训练 |
| 防止cue坍塌 | 对选中cue归一化并乘sqrt(1024)，按batch及action token统计方差；标准差下限1 |
| ACL | 同一噪声、flow time、GT action的teacher/student速度；真实7维动作跨horizon计算Pearson距离 |
| 部署 | 只保留backview学生原动作网络；辅助网络不参与推理，无gate注入 |

`../openpi`的V6.4提供历史selector结构，但该版本对cue target额外stop-gradient；
当前5K实现与PDF采用共同训练selector/predictor及方差正则。续训遵循PDF与现有5K
实现，不混用历史target-detach变体。LoRA、BS32属于5090适配，非PDF全参数BS256设置。

## 固定设置

| 项目 | 设置 |
|---|---|
| 来源 | H3/H5，任务127143，完整checkpoint `4999`，训练完成5000步 |
| Student | pi0.5 LoRA，backview单视角；PaliGemma rank16，action expert rank32；action horizon10，action dim32 |
| Teacher | 冻结agentview+wrist，全量SFT 30K；`/opt/liutong/openpi_checkpoints/fixed_dataset/sft/agentview_wrist/29999/params` |
| 数据 | fixed_dataset，`/opt/liutong/libero_multiview_tuned_6view_lerobot`；teacher/student同帧同步 |
| 损失 | flow + 0.2 × (cue cosine + 0.1 × variance) + 0.5 × ACL |
| Cue网络 | layers6+12；memory1024；projector hidden2048；query stop-gradient |
| 资源 | 4×5090 FSDP；global BS32、累计1；32CPU、96G，时限8天 |
| 调度 | 任务147454；仅gpu04，绑定用户预约res_gpu04；等待资源，不修改已有任务 |
| 优化 | seed42；原AdamW/clip1.0；warmup1000、peak2.5e-5、30K cosine、末期2.5e-6 |
| 接续 | 只读恢复参数、辅助网络、Adam及step；从第5000批重建采样顺序；保持8workers及persistent-worker语义 |
| 保存 | 10K/15K/20K/25K及26K/27K/28K/29K/30K；`keep_period=1`全部保留；来源5K留在原处 |
| 网络 | Hugging Face离线，取消代理；不下载数据或checkpoint |

只改变2→4卡分片与输出位置，BS32不变；浮点归约可能产生数值差异，不承诺逐位相同。
采样顺序接续不表示PyTorch worker内部随机状态恢复；当前图像增强在JAX中按seed与step生成。

## 路径与结果约束

| 内容 | 路径 |
|---|---|
| 源5K | `/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_lora/ablations/layers_5k/pi05_libero_backview_acpd_lora_layers6_12/pi05_libero_backview_acpd_lora_fsdp2_layers6_12_bs32_5k/4999` |
| 续训根目录 | `/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_original/backview_bs32_30k/pi05_libero_backview_acpd_original_layers6_12/pi05_libero_backview_acpd_original_lora_fsdp4_bs32_30k` |
| 目录编号 | `9999/14999/19999/24999/25999/26999/27999/28999/29999` |
| 调度脚本 | `scripts/train_slurm/pi05_libero_backview_acpd_original_fsdp4_bs32_resume_30k.sbatch` |
| 日志 | `/opt/liutong/openpi-5090-research/acpd-original-backview-bs32-30k/slurm-log/` |
| 历史结果 | `experiments/ablation/preliminary/acpd-task-success-5k/analysis.md` |

训练loss仅用于健康及收敛检查。后续成功率使用fixed_dataset相机设置、seed7、四套
各500回合；本次只提交训练，不自动增加验证任务。与H9/H19/SFT BS64比较时标注BS32
差异，不能归为只改变方法的匹配对照。协议先提交git，再提交Slurm。
