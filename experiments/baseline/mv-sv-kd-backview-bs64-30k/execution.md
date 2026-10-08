# 执行记录

更新：2026-10-08 23:13 CST。训练与仿真尚无结果。

| 项目 | 记录 |
|---|---|
| 协议 commit | `0d7c8f6`，先于实验提交 |
| 实现 commit | `ff4d1e8` |
| 训练脚本 | `scripts/train_slurm/pi05_libero_backview_mv_sv_kd_lora_fsdp4_bs64_30k.sbatch` |
| CPU 检查 | `train_distill_test.py`：26 passed；ruff、bash语法、CLI覆盖与diff检查通过 |
| 已检查内容 | 特征loss数值、教师梯度截断、三项loss对学生梯度、无ACPD参数、普通LoRA参数shape兼容、旧分支测试 |
| 尚未验证 | 真实GPU反向传播、4卡BS64峰值显存、仿真成功率 |
| Smoke job | `142274`，debug01，1GPU/8CPU/32G/1小时；Priority排队 |
| Smoke 设置 | 真实base/teacher/fixed_dataset；BS1、FSDP1、2步，不保存checkpoint |
| 正式训练 job | `142275`，batch，4GPU/32CPU/96G/8天；等待`afterok:142274` |
| 正式训练设置 | backview，BS64，累计1，30K，loss权重GT/输出/特征=1/0.5/0.1；详见协议 |
| 保存策略 | 每5K保存，keep_period=1；保留4999/9999/14999/19999/24999/29999 |
| 部署配置 | `pi05_libero_backview_lora`，无新增部署网络 |
| 原任务 | 未取消、改资源或改优先级；数据集与teacher checkpoint未移动或复制 |

## 提交命令

```bash
sbatch --partition=debug01 --gres=gpu:1 --cpus-per-task=8 --mem=32G \
  --time=01:00:00 --job-name=smoke-pi05-bv-mvsvkd \
  scripts/train_slurm/pi05_libero_backview_mv_sv_kd_lora_fsdp4_bs64_30k.sbatch \
  --name=pi05_libero_backview_mv_sv_kd_smoke --exp-name=single_gpu_two_steps \
  --checkpoint-base-dir=/opt/liutong/openpi-5090-research/mv-sv-kd-backview-bs64-30k/smoke \
  --batch-size=1 --fsdp-devices=1 --num-train-steps=2 --log-interval=1 \
  --save-interval=100 --no-save-final-checkpoint

sbatch --dependency=afterok:142274 \
  scripts/train_slurm/pi05_libero_backview_mv_sv_kd_lora_fsdp4_bs64_30k.sbatch
```

## 输出路径

以下为配置路径，排队期间尚未产生训练日志或checkpoint。

| 内容 | 路径 |
|---|---|
| 正式checkpoint目录 | `/opt/liutong/openpi_checkpoints/fixed_dataset/baseline/mv_sv_kd/backview_bs64_30k/pi05_libero_backview_mv_sv_kd/pi05_libero_backview_mv_sv_kd_lora_fsdp4_bs64_30k` |
| 正式日志 | `/opt/liutong/openpi-5090-research/mv-sv-kd-backview-bs64-30k/slurm-log/pi05-bv-mvsvkd-fsdp4-bs64-30k_142275.out` |
| Smoke日志 | `/opt/liutong/openpi-5090-research/mv-sv-kd-backview-bs64-30k/slurm-log/smoke-pi05-bv-mvsvkd_142274.out` |
