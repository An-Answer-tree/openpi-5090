# H13：ACPD-v2 注入消融

## 问题

H9-scale-b 同时使用 exact contribution 蒸馏损失和推理时 residual 注入。H13
只关闭 residual 注入，检验 H9 的收益是否需要显式注入，还是蒸馏损失作为辅助
监督已经足够。

## 假设

在保留 exact contribution 蒸馏损失时，显式注入仍能提高 5K checkpoint 的
任务成功率。

## 严格对照

| 项目 | H9-scale-b（控制） | H13（实验） |
|---|---|---|
| Teacher | agentview+wrist，step 29,999 | 相同 |
| Student | backview-only pi0.5 LoRA | 相同 |
| Teacher target | layer 10 两路 exact contribution | 相同 |
| 损失权重 | supervised `1.0`，contribution `0.2`，ACL `0.5` | 相同 |
| Predictor | 从 layer 10 student hidden 分别预测两路 contribution | 相同 |
| 注入 | 两路预测求和，经零初始化 scalar gate 加到最终 hidden | 关闭 |
| 训练 | FSDP4，physical global BS64，accumulation 1 | 相同 |
| 优化 | seed 42；1K warmup；30K cosine schedule | 相同 |
| 筛选点 | step 4,999 | step 4,999 |

唯一实验变量是预测 contribution 是否注入 student。H13 仍训练并保存 predictor；
推理时也关闭注入。训练和验证不得移动 dataset、teacher 或 checkpoint。

## 运行顺序

1. 增加独立的 injection 开关，并验证 H9 默认行为不变、H13 仍创建 predictor。
2. 运行最小单元测试和 4 GPU、BS64、2-step smoke test。
3. smoke 通过后训练 H13 到 5K，不继续到 30K。
4. 在 H9-scale-b 相同的四套 LIBERO 初始状态上验证，共 2,000 episodes。
5. 按 task 分层，对 H9-scale-b 减 H13 做配对 bootstrap。

## 判据

记 `delta = H9-scale-b pooled success - H13 pooled success`：

- `delta >= 1.5` 个百分点且配对 95% CI 下界大于 `0`：支持显式注入有效。
- `delta` 的配对 95% CI 完全落在 `[-1.5, +1.5]`：在当前精度下认为
  loss-only 足够。
- 其余情况：证据不足；不根据训练 loss、contribution cosine 或 gate 下结论。

四个 benchmark 的成功率同时报告，用于判断 pooled 差异是否由单一 suite 驱动。

## 验证资源

验证使用 2 GPU，每张 GPU 串行运行两个 benchmark。每套仍使用固定初始状态、
seed 7 和每任务 50 次，共 500 episodes；四套合计 2,000 episodes。该设置只减少
并行度，不改变 H9 与 H13 的比较协议。
