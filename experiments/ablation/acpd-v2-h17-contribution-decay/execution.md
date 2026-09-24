# H17 执行记录

提交时间：2026-09-24 11:03 CST。尚无训练或验证结论。

| 任务 | Job | 状态 | 依赖/资源 |
|---|---:|---|---|
| 10K→20K训练 | 134422 | Priority排队 | batch，4 GPU、32 CPU、96G，5天 |
| 20K全量验证 | 134423_[0-3] | 等待训练成功 | 每子任务1 GPU、8 CPU、24G，12小时；四套共2000回合 |
| 配对分析 | 134424 | 等待两组验证成功 | afterok:134423:134008；1 CPU、2G |

协议提交 `0401d01`；实现提交 `5036842`。复用原模型、注入、teacher、数据、seed
和30K LR，唯一干预为 contribution 权重。源checkpoint只读恢复，不产生副本。

## 提交前核验

- CPU 16项测试通过，覆盖原蒸馏行为、JIT调度边界和实际`compute_gradients`中的权重缩放。
- 小模型checkpoint验证指定步数、完整Adam状态及下一次更新一致；源文件大小与mtime不变。
- 完整命令经过tyro解析；4 GPU、physical BS64、累积1、目标20K、原30K LR和keep_period=1通过检查。
- 训练和分析脚本通过`bash -n`；Python lint及`git diff --check`通过。
- 源`9999`存在完整标记、params、train_state和assets。尚未执行真实4卡训练，运行状态以Slurm为准。

## 输出检索

| 内容 | 路径 |
|---|---|
| 训练launcher | `scripts/train_slurm/pi05_libero_backview_acpd_v2_h17_decay_fsdp4_bs64_20k.sbatch` |
| 配对分析launcher | `examples/libero/eval_slurm/analyze_h17_contribution_decay.sbatch` |
| 训练日志 | `/opt/liutong/openpi-5090-research/acpd-v2-h17-contribution-decay/slurm-log/pi05-bv-h17-decay-bs64-20k_134422.out` |
| 新checkpoint | `/opt/liutong/openpi_checkpoints/fixed_dataset/ablation/acpd_v2_h17_contribution_decay/pi05_libero_backview_acpd_v2_layer10/pi05_libero_backview_acpd_v2_h17_decay_bs64_20k/{14999,19999}` |
| 验证logs/videos | `/opt/liutong/openpi-5090-evals/acpd-v2-h17-contribution-decay/19999/` |
| 配对结果 | 上述验证目录内`h17_vs_h9.json/.md`、`h17_vs_sft.json/.md` |

启动时检查日志中的optimizer step=10000、dataloader batch=10000，首条实际权重0.2。
后续日志应在12500附近权重0.125、15000之后权重0.05；每100步日志为区间平均值。
运行结束后需核对两个完整checkpoint及2000回合，再解释配对差值与置信区间。
