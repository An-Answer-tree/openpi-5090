# Backview SFT BS64 30K--60K 训练轨迹

## 目的

将匹配的 backview SFT 从 step 29,999 精确续训到 60K，作为 H9 ACPD-v2
30K 后训练轨迹的公平对照，判断两种方法的最佳 checkpoint 是否仅由训练步数决定。

## 训练控制

除训练终点外，保持原 BS64 SFT 不变：4 GPU、physical global BS64、梯度累积 1、
seed 42、backview 单视角和 LoRA 配置。恢复完整模型、optimizer state 和确定性
dataloader 位置。

原 cosine schedule 在 30K 到达 `2.5e-6`。30K--60K 保持该末值，不重新启动
warmup。继续在原 checkpoint 目录写入，并以 `save_interval=5000`、
`keep_period=1` 保留 35K、40K、45K、50K、55K、60K checkpoint。

## 验证

使用与 H9 相同的 backview 相机、seed 7 和每套 500 episodes 协议验证 35K、
40K、50K、60K。分别报告四套成功率和 2,000 episodes pooled success，并在
相同步数上计算 `H9 - SFT` 的配对差值。35K 为探索性比较；40K、50K、60K用于
定位两种方法各自的最佳训练步数。不进行第二组 seed 复测。
