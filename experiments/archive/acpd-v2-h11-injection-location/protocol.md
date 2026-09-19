# H11：ACPD-v2 注入位置

## 问题

ACPD-v2 当前用第 10 层 block 输出预测 teacher attention contribution，再加到
action expert 最终 hidden。H11 检验：在第 10 层 attention 输出上完成预测并立即
注入，是否比最终层融合更有效。

## 假设

将预测 contribution 加在第 10 层 attention 残差之后、FFN 之前，使后续
网络继续处理特权信息，可提高 5K checkpoint 的任务成功率。

## 严格对照

| 项目 | H9-scale-b（控制） | H11（实验） |
|---|---|---|
| Teacher target | layer 10 agentview、wrist exact contribution | 相同 |
| Predictor | 两路独立预测后求和 | 相同 |
| Predictor query | layer 10 block 输出 | layer 10 attention 输出、注入之前 |
| 注入 | 最终 hidden、action head 之前 | layer 10 attention 后、FFN 前 |
| Gate | 单个零初始化标量，`tanh` | 相同 |
| 训练 | FSDP4，physical global BS64，accumulation 1 | 相同 |
| 优化 | seed 42；1K warmup；`2.5e-5` 到 `2.5e-6` cosine | 相同 |
| 预算 | 30K steps；每 5K 保存 | 相同 |

H11 同时前移 predictor query 和注入位置。该 query 是注入时能取得的同层状态；
若继续使用 layer 10 block 输出，就必须先完成 FFN 再回到 FFN 前注入，形成因果
循环或需要第二次 student forward。

## 实现约束

- 单次 student forward；不复制或移动 dataset、teacher、student checkpoint。
- 只对 action tokens 注入，不改变 prefix 或 time token。
- contribution 在注入路径上 stop-gradient；predictor 仅由 ACPD prediction loss 更新。
- 训练和推理使用相同注入路径。
- 保留 H9 最终 hidden 注入行为，避免改变在途控制组。

## 运行顺序

1. 单元测试：零 gate 等价、注入位置、predictor 梯度隔离、训练/推理配置一致。
2. 4 GPU、BS64、2-step smoke test；有限 loss、非零 gate/predictor/LoRA 梯度且无 OOM 才通过。
3. smoke 成功后提交 4 GPU、BS64、30K 正式训练。
4. 分别验证 H9-scale-b 与 H11 的 step 4,999；每个模型四套共 2,000 episodes。

## 判据

主要指标是 H11 相对 H9-scale-b 的 pooled success 差值。H11 需同时满足：

- pooled success 提升至少 `1.5` 个百分点；
- 按 task 分层的配对 bootstrap 95% CI 下界大于 `0`。

不满足则不支持 H11。supervised loss、contribution cosine、gate 和显存只用于
训练健康与机制分析，不替代任务成功率结论。
