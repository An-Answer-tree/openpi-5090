# H17-Decay 15K 中间点验证

## 目的

测量 contribution 权重从 0.2 衰减到 0.05 后的中间效果，区分收益出现于
15K 还是 15K 到 20K 的后续训练。

## 锁定协议

- checkpoint：H17-Decay `14999`，只读，不复制或移动。
- 对照：同源 H9-Fixed 15K；主实验仍为 H17 20K 对 H9 20K。
- 评测：LIBERO spatial、object、goal、libero_10，各 500 回合，共 2000 回合。
- 相机：fixed_dataset 的固定相机位姿；student 使用 backview、layer 10 配置。
- 资源：单卡数组，最多 2 个子任务并行；每个子任务 1 GPU、8 CPU、24G。
- 预设判据：仅报告 pooled 成功率和四套 suite；不以 loss 代替任务指标。

本结果属于预注册主实验之外的中间点补充测量。若 15K 未改善，不能否定
20K 可能改善；若 15K 改善，也只能说明衰减后的中间点有任务收益。

## 路径

- checkpoint：`/opt/liutong/openpi_checkpoints/fixed_dataset/ablation/acpd_v2_h17_contribution_decay/pi05_libero_backview_acpd_v2_layer10/pi05_libero_backview_acpd_v2_h17_decay_bs64_20k/14999`
- 输出：`/opt/liutong/openpi-5090-evals/acpd-v2-h17-contribution-decay/14999/`
