# 执行状态

2026-09-28：协议已锁定，单卡只读探针和 `debug01` Slurm 脚本已完成静态检查、
CLI 检查及纯数值单元测试（3 passed）。两个 checkpoint、资产目录和 fixed_dataset
路径均已确认存在。

10:34 CST 的检查曾发现 GPU0 上有非 Slurm 进程，因此没有立即提交。
13:57 CST 复核时，Slurm 作业分别占用 GPU0/GPU2，GPU1 空闲；Linux available
内存约 316GiB（先前误将 `FreeMem` 当成可用内存）。`sbatch --test-only` 通过后，
提交只读诊断 `136591`，已在 `debug01` GPU1 运行并开始恢复 SFT checkpoint。
14:05 CST 完成，Slurm 状态 `COMPLETED`、退出码 `0:0`、用时 7 分 49 秒。
日志：`slurm-log/probe-bv-flow-time-5k_136591.out`；原始结果：
`results/136591/summary.json`、`results/136591/paired_errors.npz`。
14:19 CST，复核任务 `136595` 也完成，退出码 `0:0`；原始结果：
`results/136595/summary.json`、`results/136595/paired_errors.npz`。同样本全时间点
复核未复现首轮低时间段差异，因此不启动时间重采样训练。
未修改已有训练、排队任务、checkpoint 或数据集。
