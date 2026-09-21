# H15：ACPD-v2 持续蒸馏

## 问题

ACPD-v2 在 5K 比匹配 SFT 高 `9.30` 点，但在 30K 低 `2.45` 点。H15 检验：
保留后期 teacher 信号，同时降低其相对强度，能否把早期优势转化为更高的 30K 上限？

## 假设

固定 `0.2/0.5` 权重使后期 contribution 与 ACL 标量和约为 supervised loss 的
`3.29` 倍，而 contribution cosine 在 5K 后改善有限。若辅助目标后期过强，则退火到
非零下限应恢复主任务优化，同时继续约束 privileged contribution predictor。

## 设置

- 起点：H9-scale-b 的完整 `4999` checkpoint，包括模型、优化器、step 和数据顺序。
- 资源：4×RTX 5090，FSDP LoRA，physical global BS64，无梯度累积。
- 模型：layer 10 exact contribution，final-hidden gated injection 保持开启。
- 主损失权重：全程 `1.0`。
- 0--5K：沿用 H9，contribution `0.2`，ACL `0.5`。
- 5K--15K：两项辅助权重分别余弦退火至 `0.02` 和 `0.1`。
- 15K--30K：保持 `0.02/0.1`，不把 teacher 信号降为零。
- 学习率：沿用原 30K cosine，不重启 warmup。
- 数据：seed 42，并从逻辑 batch 5,000 继续相同采样顺序。
- 保存：每 5K 保存，`keep_period=1`。

权重下限根据 H9 30K 的标量比例预注册：贡献项缩小10倍后约为 supervised 的
`0.26` 倍，ACL 缩小5倍后约为 `0.14` 倍，两项合计约 `0.40` 倍。标量比例不代表
梯度比例，因此该设置是待实验检验的工程假设。

## 评价

主要终点固定为 30K，并与匹配 SFT 30K 的 `60.45%`、固定权重 H9 的 `58.00%`
比较。每个模型使用相同四套 LIBERO、每套500回合，并报告 task-stratified paired
bootstrap 95% CI。

若 H15 相对 SFT 的 pooled 差值大于0且95% CI下界大于0，则支持持续蒸馏提高模型
上限。10K/20K 仅作为训练轨迹诊断，不用于事后改变主要终点。

若 H15 未超过 SFT，则下一步测试 agentview/wrist 分离、按 action token 条件化的
动态 gate；不继续扫描静态固定权重。
