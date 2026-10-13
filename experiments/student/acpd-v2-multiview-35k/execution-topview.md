# Topview 暂停与自动续训

更新时间：2026-10-13 11:42 CST。初始调度协议commit：`987f571`。

| 项目 | 实际设置或状态 |
|---|---|
| 目的 | 最初暂停topview优先H19验证；11:38按用户最新要求提前停止H19验证，立即恢复topview训练 |
| 暂停 | 147339于10:44:45停止，日志约21K；最新完整checkpoint为19999（20K），约1K未保存进度需重算 |
| 核对 | 19999与H19 34999的完成标记、params和train_state元数据均完整 |
| H19验证 | 147342_[0-3]于10:44:47启动，11:38取消；已完成1232/2000回合，日志保留，尚无全量结论 |
| 汇总 | 147343同时取消；不汇总为完整2000回合结果 |
| 后继训练 | 复用147628，解除afterok:147343；11:38:26在gpu04启动，res_gpu04，4GPU/32CPU/96G/8天 |
| 完整恢复 | 同目录最新完整20K模型、Adam状态与步数；seed42、BS64、累计1、数据从batch20000接续，不重启warmup |
| 配置与保存 | 原loss权重1/0.2/0.5、layer10、30K cosine LR、35K目标均不变；30K及之前每5K、之后每1K，keep_period=1全部保留 |
| 检查 | 原脚本不变；11:39确认优化器step20000、数据batch20000（epoch3、batch4166）；11:42首步loss=0.0689、监督loss=0.0131，BS64与累计1已由运行指标确认，训练持续推进 |
| 其他任务 | 原版ACPD147454与MV-SV-KD142275继续运行；没有移动、复制或删除checkpoint与数据 |

最初续训提交命令（历史记录）：

```bash
sbatch --parsable --hold --reservation=res_gpu04 --nodelist=gpu04 \
  --dependency=afterok:147343 \
  scripts/train_slurm/pi05_libero_topview_acpd_v2_layer10_lora_fsdp4_bs64_35k.sbatch
scancel 147339
scontrol release 147628
```

续训日志：`/opt/liutong/openpi-5090-research/acpd-v2-multiview-35k/slurm-log/pi05-tv-acpdv2-l10-fsdp4-bs64-35k_147628.out`。
Checkpoint仍使用原topview目录，不建立副本。

11:38最新调度（未重新提交训练）：

```bash
scontrol hold 147628
scontrol update JobId=147628 Dependency=
scancel 147342_0 147342_1 147342_2 147342_3 147343
scontrol release 147628
```

H19部分结果见[`evaluation-35k-partial.md`](../acpd-v2-h19-feature-fusion-30k/evaluation-35k-partial.md)。
