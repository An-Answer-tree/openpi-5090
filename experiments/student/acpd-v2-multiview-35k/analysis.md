# 多视角 ACPD-v2-Fixed 35K 验证

更新时间：2026-10-12。固定训练与评测配置见 [protocol.md](protocol.md)。
每套500回合，四套共2000回合，使用fixed_dataset相机位姿、seed 7。

| 套件 | Leftview ACPD 35K | Leftview SFT 30K | 差值（百分点） | 配对95%区间（百分点） |
|---|---:|---:|---:|---:|
| Spatial | 408/500，81.60% | 84.00% | -2.40 | [-6.80,+2.00] |
| Object | 408/500，81.60% | 79.60% | +2.00 | [-2.80,+6.80] |
| Goal | 412/500，82.40% | 79.40% | +3.00 | [-1.40,+7.60] |
| LIBERO-10 | 365/500，73.00% | 71.60% | +1.40 | [-3.80,+6.60] |
| Pooled | 1593/2000，79.65% | 1573/2000，78.65% | +1.00 | [-1.35,+3.35] |

Leftview点估计高于SFT，但配对区间跨0，未检测到确定优势。ACPD训练35K、
SFT训练30K，不能将此比较写为同训练进度的收益。区间来自相同任务初始状态的
分任务配对bootstrap（10,000次，seed 42），只反映本次评测回合的不确定性，
不覆盖训练seed变化。

| 套件 | Rightview ACPD 35K | Rightview SFT 30K | 差值（百分点） | 配对95%区间（百分点） |
|---|---:|---:|---:|---:|
| Spatial | 418/500，83.60% | 85.00% | -1.40 | [-5.20,+2.60] |
| Object | 450/500，90.00% | 84.80% | +5.20 | [+1.60,+8.80] |
| Goal | 416/500，83.20% | 84.00% | -0.80 | [-4.80,+3.20] |
| LIBERO-10 | 302/500，60.40% | 54.20% | +6.20 | [+1.20,+11.20] |
| Pooled | 1586/2000，79.30% | 1540/2000，77.00% | +2.30 | [+0.20,+4.40] |

Rightview已完成全量评测。本次pooled配对区间高于0，Object与LIBERO-10为正，
Spatial与Goal区间跨0。此比较为单训练seed、不同训练步数的探索性结果，
不能归因为同进度的蒸馏收益。配对方法与Leftview一致。

| 证据 | 路径 |
|---|---|
| Leftview最终汇总 | `/opt/liutong/openpi-5090-evals/acpd-v2-multiview-35k/leftview/34999/summary.txt` |
| Leftview SFT对照 | `/opt/liutong/openpi-5090-evals/sft-leftview-bs64-30k/29999/summary.txt` |
| Leftview配对原始指标 | [leftview_35k_vs_sft_30k_paired_analysis.json](results/leftview_35k_vs_sft_30k_paired_analysis.json) |
| Rightview最终汇总 | `/opt/liutong/openpi-5090-evals/acpd-v2-multiview-35k/rightview/34999/summary.txt` |
| Rightview SFT对照 | `/opt/liutong/openpi-5090-evals/sft-rightview-bs64-30k/29999/summary.txt` |
| Rightview配对原始指标 | [rightview_35k_vs_sft_30k_paired_analysis.json](results/rightview_35k_vs_sft_30k_paired_analysis.json) |
| Rightview日志 | `/opt/liutong/openpi-5090-evals/acpd-v2-multiview-35k/rightview/34999/logs/{spatial,object,goal,libero_10}.log` |
