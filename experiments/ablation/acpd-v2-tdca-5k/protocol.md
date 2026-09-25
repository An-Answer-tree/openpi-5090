# ACPD-v2-TDCA：任务驱动贡献适配 5K 协议

## 研究问题

ACPD-v2 的 contribution predictor 当前通过 contribution loss 学习 teacher 的
attention contribution；预测结果注入 student hidden 时使用 `stop_gradient`，因此
flow/action loss 不能直接调整 predictor。TDCA 只解除这条训练梯度截断，检验预测的
teacher contribution 是否需要同时适配 student 的动作目标。

## 预注册改动

| 项目 | ACPD-v2-TDCA | ACPD-v2-Fixed 对照 |
|---|---|---|
| student | backview pi0.5 LoRA，layer 10 | 相同 |
| teacher | agentview+wrist SFT checkpoint `29999` | 相同 |
| 数据 | fixed_dataset，seed 42，确定性采样 | 相同 |
| 训练目标 | flow=1.0，contribution=0.2，ACL=0.5 | 相同 |
| 注入 | final hidden residual；prediction 保留 action 梯度 | prediction 对注入路径 stop-gradient |
| 训练资源 | 4×5090 FSDP，physical global BS64，累积1 | 相同 |
| 优化 | warmup 1K，peak LR 2.5e-5，30K cosine schedule | 相同 |
| 终点 | 5K；保存 `4999` | 已有 `4999` |

teacher target 仍然 stop-gradient；只有 student predicted contribution 经过注入时
接收 flow loss 梯度。推理路径不改变。除该梯度路径外不修改模型、数据和 loss 权重。

## 预测与判据

如果 predictor 只会拟合 teacher、但不适合 student 动作，TDCA 应在相同 5K checkpoint
上提高 pooled success；允许 contribution cosine 小幅下降，但不能明显恶化 flow loss。
如果 cosine 提高而成功率不提高，说明 predictor 容量或拟合不是主要瓶颈；如果 flow
loss 下降、成功率不变，则动作训练仍未转化为仿真收益。

- 四套 LIBERO suite 各 500 回合，共 2,000 回合；seed 7，固定相机位姿。
- 主比较：TDCA 减 ACPD-v2-Fixed 的 pooled success 及 task-stratified paired
  bootstrap 95% CI。
- 同时记录 supervised loss、contribution cosine、gate 和实际梯度范数。
- 只有 pooled 差值的 95% CI 下界大于 0，才称为 5K 任务成功率支持 TDCA；否则报告
  无证据或负结果。单个 suite 的提高不能替代 pooled 判据。

本实验只判断 5K 的早期机制，不声称 TDCA 提高长期上限；若结果为正，后续另行预注册
20K/30K 公平实验。
