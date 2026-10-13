# MV-SV-KD：更换计算节点并完整续训

目的：在旧版 ACPD 30K 全量验证结束后，将 MV-SV-KD 从 gpu02 改到预约节点 gpu04，
保持原实验配置和输出目录。此操作不是新方法实验。

## 执行条件

| 项目 | 条件 |
|---|---|
| 验证先完成 | 旧版 ACPD 数组 `147939` 四套各500回合完成，汇总 `147944` 成功 |
| 原训练 | `142275`，gpu02，4卡BS64、累计1，目标30K |
| 停止时机 | 在最近完整保存之后停止，避开写入中的 checkpoint；优先等待15K保存，减少重复训练 |
| 完整保存证据 | 对应目录存在 `_CHECKPOINT_METADATA`，日志出现成功 finalize |
| 目标节点 | gpu04，预约 `res_gpu04`，有效期至2026-10-16 12:48 CST |
| 资源 | 4GPU、32CPU、96G，申请2天；提交前再次核对预约剩余时间 |
| 同时运行 | 保留 topview 训练 `147628` 的4卡，不取消或修改它 |
| 输出 | 继续写入原 MV-SV-KD checkpoint 目录，不移动、复制或删除数据和 checkpoint |

2026-10-13 22:38检查：原训练约13.8K，最新完整目录9999；尚未停止或提交续训。
不从10K立即重跑，以免浪费约3.8K已完成步数。

## 续训命令

使用原脚本，仅追加完整恢复选项并指定预约节点；不修改活动任务脚本。
原任务停止并释放资源后执行：

```bash
sbatch --parsable --reservation=res_gpu04 --nodelist=gpu04 --time=2-00:00:00 \
  --job-name=pi05-bv-mvsvkd-r30k-gpu04 \
  scripts/train_slurm/pi05_libero_backview_mv_sv_kd_lora_fsdp4_bs64_30k.sbatch \
  --resume --resume-data-loader
```

`--resume` 从原目录最新完整 checkpoint 恢复模型参数、Adam 状态和优化器步数。
`--resume-data-loader` 按优化器步数恢复确定性采样位置；累计1时二者相等。
噪声和flow时间的随机数由seed42与优化器步数生成，保持原规则。
仍使用BS64、累计1、FSDP4、原teacher、原三项loss权重和原30K cosine LR。
每5K保存，`keep_period=1`，全部保留。

## 完成检查

1. 新日志显示 gpu04、4卡BS64和累计1。
2. 恢复的优化器步数和data batch等于最近完整 checkpoint 的完成步数。
3. 实际训练推进，loss有限；仅加载模型或排队不算成功。
4. Slurm确认gpu04共8卡分配：topview4卡、MV-SV-KD4卡。

实际提交编号、恢复步数与开始时间在执行后记录。尚无新的仿真结论。
