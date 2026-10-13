# H19 30K→35K 续训执行记录

提交时间：2026-10-12 14:54 CST。协议提交：`a6fb25f`。任务：`147300`。

| 项目 | 设置 |
|---|---|
| 状态 | 2026-10-12 15:16:41在gpu04开始；2026-10-13 05:45:08最终checkpoint完成，05:45:16验证afterok依赖满足；续训耗时约14小时29分 |
| 资源 | batch，4×5090，32 CPU，96G，3天，预约`res_gpu04` |
| 训练 | FSDP4 LoRA，physical global BS64，累计1，30K→35K |
| 恢复 | 源checkpoint只读，恢复模型、优化器与步数；seed42、BS64，从batch30000接续 |
| 模型与loss | H19 FeatureFusion、layer10；flow/contribution/ACL=1/0.2/0.5 |
| 学习率 | 沿用30K cosine；30K以后保持2.5e-6，不重启warmup |
| 保存 | 每1K保存，keep_period=1；30999/31999/32999/33999/34999均已完整保存，2026-10-13 09:44核对完成标记、params与train_state元数据 |
| 检查 | 5项恢复测试通过；bash语法、Slurm test-only及保存脚本通过；实际恢复step30000、batch30000，首个训练loss有限 |
| 验证 | 147342_[0-3]于10月13日10:44:47在gpu04全部启动；汇总147343等待四套验证成功；尚无35K结果 |

- 脚本：`scripts/train_slurm/pi05_libero_backview_acpd_v2_feature_fusion_fsdp4_bs64_resume_35k.sbatch`
- 源checkpoint：`/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/feature_fusion_30k/pi05_libero_backview_acpd_v2_feature_fusion/pi05_libero_backview_acpd_v2_feature_fusion_lora_fsdp4_bs64_30k/29999`
- 目标目录：`/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/feature_fusion_35k/pi05_libero_backview_acpd_v2_feature_fusion/pi05_libero_backview_acpd_v2_feature_fusion_lora_fsdp4_bs64_35k`
- 日志：`/opt/liutong/openpi-5090-research/acpd-v2-h19-feature-fusion-35k/slurm-log/pi05-bv-h19-feature-fusion-r35k_147300.out`

原三项35K验证及其汇总继续执行，本次未修改其他任务，也未移动或复制checkpoint。

## 35K自动全量验证

协议提交：`bf943ff`。复用已有数组和汇总脚本，没有修改训练或评测代码。
四个子任务各1 GPU、8 CPU、24G、24小时，无并行数量限制；使用`res_gpu04`。
每套500回合，共2000回合，seed 7；policy为
`pi05_libero_backview_acpd_v2_layer10_feature_fusion_lora`。

| 项目 | 路径或检查结果 |
|---|---|
| Checkpoint | 本次4卡续训目标目录的`34999`；任务开始时检查`_CHECKPOINT_METADATA` |
| 评测结果 | `/opt/liutong/openpi-5090-evals/acpd-v2-h19-feature-fusion/34999/` |
| 日志与视频 | 同目录`slurm-log/`、`logs/`、各suite的`videos/` |
| 汇总 | 同目录`summary.txt`；四套全部成功后运行 |
| 已完成检查 | bash语法、Slurm test-only、CPU读取policy配置通过；实际依赖和资源已核对 |

提交命令：

```bash
sbatch --job-name=eval-bv-h19-35k --reservation=res_gpu04 \
  --dependency=afterok:147300 --time=1-00:00:00 \
  --output=/opt/liutong/openpi-5090-evals/acpd-v2-h19-feature-fusion/34999/slurm-log/%x_%A_%a.out \
  examples/libero/eval_slurm/run_fixed_backview_benchmark_array.sbatch \
  /opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/feature_fusion_35k/pi05_libero_backview_acpd_v2_feature_fusion/pi05_libero_backview_acpd_v2_feature_fusion_lora_fsdp4_bs64_35k/34999 \
  /opt/liutong/openpi-5090-evals/acpd-v2-h19-feature-fusion/34999 \
  pi05_libero_backview_acpd_v2_layer10_feature_fusion_lora backview_image 50

sbatch --job-name=summary-bv-h19-35k --reservation=res_gpu04 \
  --dependency=afterok:147342 \
  --output=/opt/liutong/openpi-5090-evals/acpd-v2-h19-feature-fusion/34999/slurm-log/%x_%j.out \
  examples/libero/eval_slurm/summarize_fixed_backview_benchmark.sbatch \
  /opt/liutong/openpi-5090-evals/acpd-v2-h19-feature-fusion/34999
```

2026-10-13按用户要求暂停topview训练147339，释放4卡后上述验证自动启动。
未重提或重复验证；topview后继任务147628依赖本汇总147343成功后才续训。
10:48四套均完成模型加载并进入仿真回合，近期日志未见异常；尚无最终成功率。
