# Topview 暂停与自动续训

更新时间：2026-10-13 10:45 CST。调度协议commit：`987f571`。

| 项目 | 实际设置或状态 |
|---|---|
| 目的 | 暂停topview，优先完成H19 35K全量验证，随后自动续训 |
| 暂停 | 147339于10:44:45停止，日志约21K；最新完整checkpoint为19999（20K），约1K未保存进度需重算 |
| 核对 | 19999与H19 34999的完成标记、params和train_state元数据均完整 |
| H19验证 | 复用147342_[0-3]，四套各500回合、共2000回合；10:44:47全部在gpu04启动，每项1GPU/8CPU/24G，无并发限制 |
| 汇总 | 147343，afterok:147342；全部成功后汇总 |
| 后继训练 | 147628，afterok:147343；gpu04、res_gpu04，4GPU/32CPU/96G/8天，提交后已解除hold，等待依赖 |
| 完整恢复 | 同目录最新完整20K模型、Adam状态与步数；seed42、BS64、累计1、数据从batch20000接续，不重启warmup |
| 配置与保存 | 原loss权重1/0.2/0.5、layer10、30K cosine LR、35K目标均不变；30K及之前每5K、之后每1K，keep_period=1全部保留 |
| 检查 | bash语法、Slurm test-only和实际afterok依赖核对通过；10:48 H19四套均进入仿真回合；后继训练GPU恢复尚未执行 |
| 其他任务 | 原版ACPD147454与MV-SV-KD142275继续运行；没有移动、复制或删除checkpoint与数据 |

续训提交命令：

```bash
sbatch --parsable --hold --reservation=res_gpu04 --nodelist=gpu04 \
  --dependency=afterok:147343 \
  scripts/train_slurm/pi05_libero_topview_acpd_v2_layer10_lora_fsdp4_bs64_35k.sbatch
scancel 147339
scontrol release 147628
```

续训日志：`/opt/liutong/openpi-5090-research/acpd-v2-multiview-35k/slurm-log/pi05-tv-acpdv2-l10-fsdp4-bs64-35k_147628.out`。
Checkpoint仍使用原topview目录，不建立副本。
