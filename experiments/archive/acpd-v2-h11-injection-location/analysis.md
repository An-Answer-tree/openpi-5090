# H11 结果：注入位置

H11 与 H9-scale-b 使用相同的 teacher、student view、layer 10 target、loss
权重、seed、FSDP4、physical BS64 和 step 4,999 checkpoint。H11 将 predictor
query 和注入位置从最终 hidden 前移到 layer 10 attention 后、FFN 前。

| 方法 | Spatial | Object | Goal | LIBERO-10 | Pooled |
|---|---:|---:|---:|---:|---:|
| H9-scale-b：最终 hidden 注入 | 25.40% | 32.40% | 30.00% | 4.80% | 23.15% |
| H11：layer 10 attention 后注入 | 14.60% | 40.60% | 27.00% | 2.60% | 21.20% |
| H11 - H9 | -10.80 | +8.20 | -3.00 | -2.20 | -1.95 |

10,000 次 task-stratified paired bootstrap 得到 pooled 差值 95% CI
`[-4.10, +0.20]` 个百分点。H11 未达到预注册的 `+1.5` 点和正置信区间下界，
因此 H11 假设不成立。该变化提高 Object，但显著降低 Spatial；没有证据说明
同层注入整体优于 H9 的最终 hidden 注入。

原始配对结果位于 `results/h11_vs_h9_paired_analysis.json`。四套验证日志位于
`/opt/liutong/openpi-5090-evals/acpd-v2-injection-location-5k/`。
