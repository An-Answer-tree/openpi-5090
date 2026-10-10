# H9 35K 推理期注入消融

预注册日期：2026-10-10。仅验证已有 checkpoint，不训练、不修改或移动权重。

## 目的与做法

判断已训练 H9 35K 的闭环成功率是否依赖预测 contribution 的推理期注入。
复用完整注入的已有全量验证，只新增关闭注入的验证。

| 项目 | 固定设置 |
|---|---|
| 模型 | ACPD-v2-Fixed（H9），backview，layer 10，BS64 训练所得 `34999` checkpoint |
| 对照 | `pi05_libero_backview_acpd_v2_layer10_lora`，原验证 pooled `61.05%`（1221/2000） |
| 干预 | 用 `pi05_libero_backview_acpd_v2_layer10_loss_only_lora` 加载同一 checkpoint，仅关闭两路预测 contribution 的 residual 注入 |
| 参数 | 参数结构不变，checkpoint 只读；保留训练所得主干、预测器、gate 和动作输出层权重 |
| 环境 | fixed_dataset 的 tuned 相机位姿；student 仅使用 backview；LIBERO Spatial/Object/Goal/10 |
| 回合 | 每套10个任务，每任务50回合，共2000回合；seed 7，相同任务初始状态 |
| 推理 | 图像224，replan steps 5，wait steps 10，flow sampling steps 10；沿用原动作归一化 |
| 资源 | 单卡数组 `0-3`，不限制数组并发；每项8 CPU、24G、24小时；排除 gpu03；batch分区 |
| 汇总 | 四项全部成功后，1 CPU、2G自动汇总及配对分析 |

checkpoint：
`/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/batch_scaling_30k/pi05_libero_backview_acpd_v2_layer10/pi05_libero_backview_acpd_v2_lora_fsdp4_bs64_30k/34999`。

完整注入对照：
`/opt/liutong/openpi-5090-evals/acpd-v2-training-trajectory/final-hidden/34999/`。

关闭注入的日志与视频：
`/opt/liutong/openpi-5090-evals/acpd-v2-h9-inference-ablation/34999/`。

## 指标与判读

主指标为完整注入减关闭注入的 pooled 成功率差值。报告四套成功/总数、各套差值
及95%配对区间。按固定任务内的相同初始状态配对重采样10000次，bootstrap seed 42。
使用已有 `scripts/distillation_acpd/analyze_paired_eval.py`；关闭注入作为 baseline，
完整注入作为 treatment。评测不中途按临时成功率停止。

差值为正且区间下界大于零，支持该 checkpoint 的推理期注入有正向作用。
区间跨零表示本实验未检测到总体差异，不等于注入无效。
若关闭注入更好，只说明该 checkpoint 的当前推理注入降低成功率。

这是预先指定干预的机制实验；35K是在已有轨迹中选出的观测最佳点，结论限定于该点。
训练使用注入、推理关闭注入会改变模型的输入分布。本实验不能分离训练期注入
与蒸馏 loss 的作用，不能代替从头训练的无注入消融，也不单独证明提高最终上限。
