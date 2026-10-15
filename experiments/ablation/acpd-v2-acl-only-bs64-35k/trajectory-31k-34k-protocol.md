# ACL-only 30K→34K 轨迹补训

## 目的与边界

从既有ACL-only 30K完整状态补齐31K、32K、33K、34K checkpoint，供后续观察
后期成功率轨迹。本次不是新方法，不根据训练loss判断任务成功率；
不预设某个checkpoint优于已有35K，不宣称逐位复现历史连续训练。
原35K与全部已有checkpoint保留。本次只提交训练，不自动提交验证。

按用户要求先取消三视角SFT续训149303/149304/149305，不自动恢复这些任务，
不调整其他训练、验证、优先级或reservation。

## 固定配置

| 项目 | 配置 |
|---|---|
| Teacher / student | 既有agentview+wrist 30K全量teacher / backview pi0.5 LoRA |
| 数据 | fixed_dataset，`libero_multiview_tuned_6view_lerobot`，沿用原归一化，seed42 |
| 资源 | batch自动选节点；4×5090 FSDP、32CPU、96G、24小时；BS64、累计1 |
| 训练目标 | flow=1.0、ACL=0.5、contribution=0；layer10；关闭注入，与原ACL-only一致 |
| 恢复 | 原29999的完整模型与Adam状态，只读加载；数据从batch30000接续 |
| 学习率 | 沿用warmup1K、peak2.5e-5、30K cosine至2.5e-6；续训不重置，保持2.5e-6 |
| 保存 | 目标34000；save_interval=1000、keep_period=1，保留30999/31999/32999/33999 |
| 网络 | conda openpi，Hugging Face/Transformers离线，清除代理；缓存和临时目录在/opt/liutong |

checkpoint编号是零起始，四个目录分别包含精确31000/32000/33000/34000个优化步。
源checkpoint、teacher与数据集不移动、不复制、不删除；输出使用独立目录。

## 路径

- 脚本：`scripts/train_slurm/pi05_libero_backview_acl_only_fsdp4_bs64_resume_34k.sbatch`
- 源：`/opt/liutong/openpi_checkpoints/fixed_dataset/ablation/acpd_v2_acl_only_bs64_35k/pi05_libero_backview_acl_only_bs64_35k/pi05_libero_backview_acl_only_lora_fsdp4_bs64_35k/29999`
- 输出：`/opt/liutong/openpi_checkpoints/fixed_dataset/ablation/acpd_v2_acl_only_bs64_trajectory_31k_34k/pi05_libero_backview_acl_only_bs64_35k/pi05_libero_backview_acl_only_lora_fsdp4_bs64_resume_34k/{30999,31999,32999,33999}`
- 日志：`/opt/liutong/openpi-5090-research/acpd-v2-acl-only-bs64-trajectory-31k-34k/slurm-log/`

提交前检查脚本语法、实际CLI配置、保存边界、完整状态恢复及数据接续的CPU测试，
并用Slurm test-only确认资源请求可接受；正式启动后再确认GPU恢复、batch30000和有限loss。
没有可用的会话级周期监测工具，本次只核实提交和当次启动状态，不创建系统cron。
