# H9 10K 推理期注入消融

预注册日期：2026-09-29。只验证已有 checkpoint，不训练、不移动模型或数据。

## 问题

H9 10K 比匹配 SFT 高 14.20 个百分点；H13 从头训练时关闭注入，5K 比 H9
低 2.55 点。但 H13 同时改变了训练轨迹，不能回答 H9 部署时是否依赖注入。

## 配对实验

| 项目 | 固定设置 |
|---|---|
| 模型 | H9-Fixed backview BS64 layer 10 的同一个 9999 checkpoint |
| 对照 | 原配置 `pi05_libero_backview_acpd_v2_layer10_lora`，已完成 2,000 回合验证 |
| 干预 | 仅改用现有 `pi05_libero_backview_acpd_v2_layer10_loss_only_lora` 配置加载同一 checkpoint；推理时关闭 residual 注入 |
| 数据与环境 | fixed_dataset 相机位姿；LIBERO 四套；每套 500 回合；seed 7；相同任务初始状态 |
| 参数 | 两配置参数树相同；不改 checkpoint、模型权重或其他推理参数 |
| 主指标 | 对照成功率减干预成功率；40 个任务、2,000 回合的 task-stratified paired bootstrap 95% 区间 |

对照日志：`/opt/liutong/openpi-5090-evals/h9-early-trajectory-full/9999/logs/`。
checkpoint：`/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/trajectory_recovery_20k/pi05_libero_backview_acpd_v2_layer10/pi05_libero_backview_acpd_v2_lora_fsdp4_bs64_20k_keep_all/9999`。
干预结果写入：`/opt/liutong/openpi-5090-evals/acpd-v2-h9-inference-ablation/9999/`。

## 判读

若差值为正且配对区间下界大于零，支持“已训练 H9 的推理输出依赖注入”。
若区间跨零，则本实验未检测到总体依赖；不等于注入从未影响训练。
同时报告四套各自差值，避免 pooled 结果掩盖方向相反的套件。
本实验不分离预测 agentview 与 wrist 的各自贡献，也不能证明注入一定提高最终上限。
