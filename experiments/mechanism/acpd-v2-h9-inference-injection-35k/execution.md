# 执行记录

提交：2026-10-10 15:24–15:26 CST。协议预先提交：`04aa2b2`。

| 阶段 | Job | 15:26状态 | 资源 |
|---|---|---|---|
| 关闭注入的四套全量验证 | `145280_[0-3]` | 排队，Priority | 每项1 GPU、8 CPU、24G、24小时；不限制数组并发 |
| 自动汇总与配对分析 | `145282` | 等待 `afterok:145280` | 1 CPU、2G、20分钟 |

数组 `0/1/2/3` 分别对应 Spatial/Object/Goal/LIBERO-10，各500回合。
任务使用batch分区、排除gpu03，不绑定节点。运行全程关闭联网并清除代理。
结果尚未产生，尚无结论；不修改已有训练或验证任务。

## 提交命令

```bash
sbatch --parsable --job-name=h9-injection-off-35k --time=1-00:00:00 \
  examples/libero/eval_slurm/run_fixed_backview_benchmark_array.sbatch \
  /opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/batch_scaling_30k/pi05_libero_backview_acpd_v2_layer10/pi05_libero_backview_acpd_v2_lora_fsdp4_bs64_30k/34999 \
  /opt/liutong/openpi-5090-evals/acpd-v2-h9-inference-ablation/34999 \
  pi05_libero_backview_acpd_v2_layer10_loss_only_lora \
  backview_image 50

sbatch --parsable --dependency=afterok:145280 \
  experiments/mechanism/acpd-v2-h9-inference-injection-35k/analyze.sbatch
```

## 检查与证据

| 检查 | 结果 |
|---|---|
| checkpoint完整标记 | `_CHECKPOINT_METADATA` 存在 |
| 模型配置差异 | 仅 `exact_contribution_injection` 从True变False |
| 配置与训练/部署一致性测试 | 2项通过 |
| 配对分析测试 | 2项通过 |
| 原35K对照日志 | 40任务×50回合完整；Spatial/Object/Goal/10成功数366/369/318/168，共1221/2000 |
| Shell语法和提交检查 | `bash -n`、`git diff --cached --check`、Slurm test-only通过；正式资源和依赖已核实 |

| 输出 | 路径 |
|---|---|
| 关闭注入日志、逐套视频、summary.txt | `/opt/liutong/openpi-5090-evals/acpd-v2-h9-inference-ablation/34999/` |
| Slurm日志 | `/opt/liutong/openpi-5090-evals/slurm-log/h9-injection-off-35k_145280_*.out`；`summary-h9-injection-off-35k_145282.out` |
| 完整注入对照 | `/opt/liutong/openpi-5090-evals/acpd-v2-training-trajectory/final-hidden/34999/` |
| 自动配对分析 | `experiments/mechanism/acpd-v2-h9-inference-injection-35k/results/paired_analysis.json` |

配对差值方向为完整注入减关闭注入。当前没有会话定时监测工具；后处理由Slurm
依赖执行，不表示代理会持续监测或自动更新台账。
