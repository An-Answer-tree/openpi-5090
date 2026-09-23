# H9 ACPD-v2 30K--60K 训练轨迹

## 目的

从 H9-scale-b 的 step 29,999 精确续训到 60K，判断成功率在 30K 后继续提高、
进入平台期或下降，并在固定 10K 间隔中定位最佳 checkpoint。

## 训练控制

除训练终点外，保持 H9-scale-b 不变：4 GPU、physical global BS64、梯度累积 1、
seed 42、layer 10 exact contribution、loss 权重 `1.0/0.2/0.5`。恢复完整模型、
optimizer state 和确定性 dataloader；随机噪声继续由 optimizer step 生成。

原 cosine schedule 在 30K 已到达 `2.5e-6`。30K--60K 保持该末值，不重启
warmup，也不把学习率重新升高。继续在原 checkpoint 目录写入，并以
`save_interval=5000`、`keep_period=1` 保留 `34999/39999/44999/49999/54999/59999`。

## 验证

训练成功结束后，用同一 backview 相机、seed 7 和每套 500 episodes，只验证
40K、50K、60K。每个 checkpoint 报告四套成功率和 2,000 episodes pooled
success，并将三者中的最高值作为本次训练轨迹的最佳结果。不进行第二组 seed 复测。

训练产生完整 35K checkpoint 后，新增一次探索性提前验证。35K 使用完全相同的
相机、seed、四套 benchmark 和 2,000 episodes 协议，直接与 30K 比较，用于判断
30K 后是否已出现继续提高的信号。该结果不替代原定的 40K、50K、60K 验证。
