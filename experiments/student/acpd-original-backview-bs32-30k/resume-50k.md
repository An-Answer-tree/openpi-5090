# 原论文 ACPD：30K→50K 续训

本协议在提交训练前锁定。按用户要求停止30K验证，延长原实验观察后期效果；
30K评测未完成，不能据临时成功率判定训练已过峰值或尚未收敛。

| 项目 | 设置 |
|---|---|
| 方法 | 与原[5K→30K协议](protocol.md)一致；layers6+12的cue选择与预测，无ACPD-v2注入 |
| 接续 | 原目录29999恢复student、cue网络、Adam与optimizer step30000；同seed42、8workers，数据从batch30000接续 |
| 训练目标 | optimizer step50000；本次增加20000步 |
| 资源 | gpu04，res_gpu04预约；4×5090 FSDP、global BS32、累计1；32CPU、96G；时限2天 |
| 优化 | 沿用原30K cosine曲线，不重启warmup；30K后LR保持2.5e-6；flow/cue/ACL权重1.0/0.2/0.5，cue内variance权重0.1 |
| 保存 | 沿用25K后每1K规则；新增30999–49999共20个checkpoint，keep_period=1全部保留 |
| checkpoint位置 | 保持原backview_bs32_30k目录及experiment name；名称含30k但目标已延长至50K，不移动、复制或删除既有checkpoint |
| 网络 | 复用conda openpi；Hugging Face离线并清除代理，不下载 |
| 调度脚本 | `scripts/train_slurm/pi05_libero_backview_acpd_original_fsdp4_bs32_resume_50k.sbatch`，复用原脚本，仅覆盖训练终点与来源 |
| 日志 | `/opt/liutong/openpi-5090-research/acpd-original-backview-bs32-50k/slurm-log/` |
| 验证与判断 | 本次只提交训练；成功率仍需同协议闭环验证，不能由训练loss确认峰值提升 |

checkpoint根目录：
`/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_original/backview_bs32_30k/pi05_libero_backview_acpd_original_layers6_12/pi05_libero_backview_acpd_original_lora_fsdp4_bs32_30k/`。

原训练25K增量约15小时37分，末期约3秒/步；20K续训粗估13–18小时，
取2天时限覆盖恢复、编译、保存与服务器波动。预约截至2026-10-16 12:48 CST。

这是探索性延长训练；没有预设50K优于30K，不自动提交额外验证，不改变topview
或MV-SV-KD任务。启动后检查恢复step、数据batch、有限loss及连续训练进度。
