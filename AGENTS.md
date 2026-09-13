# Research Coding Agent

- 作为科研实验代码 agent，在功能正确的前提下最小化实现，避免无关重构和冗余校验。
- Python 使用 Google Python Style，代码保持简单、可读、适合开源。
- 修改后运行最小必要检查。只提交本任务文件并推送当前分支；网络连接失败时使用 `http://127.0.0.1:7877` 重试。
- 不移动数据集或 checkpoint。不得递归扫描大型数据、checkpoint、日志或视频目录；优先读取明确路径和查询 Slurm。

## 实验记录

- 开始实验规划、提交、监测、分析或结果汇报前，先读取 `EXPERIMENT_RESULTS.md`、`research-state.yaml` 和 `findings.md`。
- 实验产生最终结果后，同步更新 `EXPERIMENT_RESULTS.md`。只记录实际运行的配置、指标、结论和证据路径。
- 使用简洁中文和表格。未完成或未验证的结果必须明确标记，不得由训练 loss 推断任务成功率。
