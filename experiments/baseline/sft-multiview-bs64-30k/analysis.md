# H14：多视角 SFT BS64 30K 结果

| 视角 | Spatial | Object | Goal | LIBERO-10 | Pooled | 状态 |
|---|---:|---:|---:|---:|---:|---|
| backview | 70.00% | 69.80% | 68.20% | 33.80% | 60.45% | 完成 |
| topview | 81.00% | 80.60% | 79.20% | 45.40% | 71.55% | 完成 |
| leftview | 84.00% | 79.60% | 79.40% | 71.60% | 78.65% | 完成 |
| rightview | 85.00% | 84.80% | 84.00% | 54.20% | 77.00% | 完成 |

每个结果包含四套 benchmark、共 2,000 episodes。四个视角的 pooled 排序为
leftview `78.65%`、rightview `77.00%`、topview `71.55%`、backview `60.45%`。
backview 比次低的 topview 低 `11.10` 点，支持遮挡和相机位姿显著影响单视角策略，
也说明后续蒸馏必须使用 view-matched SFT 对照。
