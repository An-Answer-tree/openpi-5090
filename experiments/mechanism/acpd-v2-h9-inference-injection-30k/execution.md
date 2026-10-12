# H9 30K 关闭推理注入：执行记录

2026-10-12提交；协议提交`1d956a0`。使用现有验证器，不修改模型或训练代码。

| 任务 | Job | 提交时状态 | 资源 |
|---|---|---|---|
| 四套全量验证 | 147346_[0-3] | Priority排队 | 每项1 GPU、8 CPU、24G、24小时；无数组并发限制 |
| 汇总与配对分析 | 147347 | 等待afterok:147346 | 1 CPU、2G、20分钟 |

使用`res_gpu04`预约；排除gpu03。评测全程关闭联网、清除代理；checkpoint只读。
数组0/1/2/3对应Spatial/Object/Goal/LIBERO-10，各500回合。

| 检查 | 结果 |
|---|---|
| Checkpoint | 29999完整标记存在 |
| 模型配置 | 与完整H9仅`exact_contribution_injection`不同 |
| 对照完整性 | 原30K日志四套均有10任务×50回合，共1160/2000成功 |
| 提交 | bash语法、Slurm test-only通过；实际资源、预约与汇总依赖已核对 |

提交命令：

```bash
sbatch --parsable --job-name=h9-injection-off-30k --reservation=res_gpu04 \
  --time=1-00:00:00 \
  --output=/opt/liutong/openpi-5090-evals/acpd-v2-h9-inference-ablation/29999/slurm-log/%x_%A_%a.out \
  examples/libero/eval_slurm/run_fixed_backview_benchmark_array.sbatch \
  /opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/batch_scaling_30k/pi05_libero_backview_acpd_v2_layer10/pi05_libero_backview_acpd_v2_lora_fsdp4_bs64_30k/29999 \
  /opt/liutong/openpi-5090-evals/acpd-v2-h9-inference-ablation/29999 \
  pi05_libero_backview_acpd_v2_layer10_loss_only_lora backview_image 50

sbatch --parsable --reservation=res_gpu04 --dependency=afterok:147346 \
  experiments/mechanism/acpd-v2-h9-inference-injection-30k/analyze.sbatch
```

日志、视频、summary.txt在
`/opt/liutong/openpi-5090-evals/acpd-v2-h9-inference-ablation/29999/`；配对结果在
`experiments/mechanism/acpd-v2-h9-inference-injection-30k/results/paired_analysis.json`。
尚无30K关闭注入成功率结论。

18:39确认Spatial/Object/Goal三个子任务在gpu04运行，日志已有连续仿真
回合及Success记录；模型加载和环境启动正常。LIBERO-10等待Priority，
汇总等待四套全部成功。尚无完整成功率结果。
