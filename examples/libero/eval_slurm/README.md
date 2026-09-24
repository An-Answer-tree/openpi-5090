# LIBERO 验证脚本

当前 H9 与 SFT 轨迹的四套正式验证使用 `run_fixed_backview_benchmark_array.sbatch`：
每个 array 子任务验证一套 benchmark，四套各 500 回合。其调用
`run_fixed_backview_suite.bash` 启动 policy server 和仿真，并将日志、视频写到传入的
`EVAL_DIR`。汇总任务在四个子任务成功后运行。

`pi05_*.sbatch` 是各次实验使用的固定 checkpoint wrapper，保留用于复现历史提交；
其中的旧 50K/60K H9 wrapper 不代表任务仍在运行。当前任务号和结果路径以
`research-state.yaml` 与 `EXPERIMENT_RESULTS.md` 为准。
