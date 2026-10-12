# H9 30K 推理期注入消融

预注册日期：2026-10-12。复用已训练模型，仅关闭推理期注入。

| 项目 | 设置 |
|---|---|
| 目的 | 测量H9 30K成功率是否依赖推理期预测贡献注入 |
| 模型 | ACPD-v2-Fixed，backview、layer 10，BS64训练所得`29999` |
| 干预 | 同一权重，仅将`exact_contribution_injection`从True改为False |
| 配置 | `pi05_libero_backview_acpd_v2_layer10_loss_only_lora`；名称含loss_only，但本实验没有重新训练 |
| 对照 | 复用完整注入30K全量验证：Spatial/Object/Goal/10为64.0/73.0/61.8/33.2%，pooled 58.00% |
| 环境 | fixed_dataset tuned相机位姿，student仅看backview |
| 回合 | 四套各10任务，每任务50回合，共2000回合，seed 7 |
| 推理 | 图像224，replan 5，wait 10，flow sampling 10；沿用checkpoint动作归一化 |
| 资源 | batch单卡数组0–3，无并发数量限制；每项8 CPU、24G、24小时；排除gpu03；使用res_gpu04 |
| 后处理 | 四套全部成功后，1 CPU、2G、20分钟自动汇总并计算配对区间 |

Checkpoint只读，不复制或移动：
`/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/batch_scaling_30k/pi05_libero_backview_acpd_v2_layer10/pi05_libero_backview_acpd_v2_lora_fsdp4_bs64_30k/29999`。

完整注入对照：
`/opt/liutong/openpi-5090-evals/acpd-v2-training-trajectory/final-hidden/29999/`。

关闭注入结果、日志和视频：
`/opt/liutong/openpi-5090-evals/acpd-v2-h9-inference-ablation/29999/`。

主指标为开启减关闭的pooled成功率差值；固定任务内按相同初始状态配对
bootstrap 10000次，seed42，报告四套及pooled的95%区间。不中途按临时
成功率停止。区间跨0表示未检测到差异，不表示两者等效。

这是已有模型的推理干预，不能替代从头训练的无注入消融，也不能判断训练期
注入与蒸馏loss各自的作用。此前35K验证已完成，结果独立保留；30K尚无结论。
