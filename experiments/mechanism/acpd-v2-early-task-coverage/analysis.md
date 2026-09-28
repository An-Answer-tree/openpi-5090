# ACPD-v2 早期逐任务收益：探索性结果

复用匹配的 H9-Fixed 与 backview SFT 的 5K、10K、15K 正式验证日志。
每点四套、40任务、每任务50回合；逐任务差值为 `H9成功率-SFT成功率`。
脚本核对了同一步两模型的 episode 键，以及三个训练点的任务与 episode 集合。
未重新训练或仿真。重算的四套及 pooled 差值与原配对分析一致。

| 套件 | 5K 正/负/零任务数 | 10K 正/负/零 | 15K 正/负/零 |
|---|---:|---:|---:|
| Spatial | 9/1/0 | 9/1/0 | 5/3/2 |
| Object | 10/0/0 | 7/2/1 | 7/2/1 |
| Goal | 6/4/0 | 8/2/0 | 7/2/1 |
| LIBERO-10 | 3/1/6 | 8/1/1 | 6/3/1 |
| **全部40任务** | **28/6/6** | **32/6/2** | **25/10/5** |

10K 的 pooled 差值为 `+14.20` 点，32/40任务为正，四套均至少7/10任务为正。
因此该点的收益不只由少数任务造成。到15K，10K时为正的32个任务中仅20个仍为正，
12个变为零或负；40任务的10K与15K差值 Spearman 相关为 `0.123`。
例如 Goal 的“push the plate to the front of the stove”从 `+46` 点变为 `-32` 点。
这说明当前观测到的逐任务收益排序并不稳定，不能直接按10K结果制定固定任务路由。

本分析已知套件汇总后才开展，且只有一个训练seed、每任务50回合；逐任务计数和
相关性是探索性描述，没有进行40任务的多重比较校正。这里不能判断变化来自注入、
其他训练路径，还是有限回合的波动。推理期同checkpoint注入消融见
`../acpd-v2-h9-inference-injection-10k/`。

证据：`results/task_coverage.json`；输入日志目录记录于
`experiments/baseline/sft-backview-bs64-5k/results/h9_vs_h12_paired_analysis.json` 与
`experiments/student/acpd-v2-h9-early-trajectory/results/h9_vs_sft_{10k,15k}_paired_analysis.json`。
