# 5090 训练基础设施与 SFT 基线

## 训练能力

以下均为 pi0.5 LoRA 的实测配置。BS 表示全局 batch size。

| 训练类型 | 最少可运行 GPU | 4 卡无梯度累积的已验证 BS | 实测依据 |
|---|---:|---:|---|
| SFT | 1 | 32 | LoRA 单卡可运行；4 卡 BS32 已稳定完成 60K steps。 |
| 原始 ACPD 蒸馏 | 2 | 32 | 2 卡 BS32 已完成多组 2K/5K 实验；4 卡 BS32 已通过训练测试。 |

当前 ACPD-v2 比原始 ACPD 多计算真实 attention contribution。为降低单次计算压力，采用 4 卡、micro BS8、梯度累积 4 次，effective BS 仍为 32。

实测训练时间：SFT 4 卡 BS32 训练 30K 约 16 小时；原始 ACPD 2 卡 BS32 训练 5K 约 4.5 至 9 小时。ACPD-v2 尚未完成，不报告最终用时。

## BS32 的 SFT 结果

共同设置：4 卡 FSDP LoRA、global BS32、cosine 学习率。每个 checkpoint 在四套 LIBERO benchmark 上共验证 2,000 episodes。

| Checkpoint | Pooled success |
|---:|---:|
| 30K | 46.20% |
| 40K | 49.20% |
| **50K** | **57.05%** |
| 60K | 54.25% |

结论：BS32 已能稳定支持 SFT 和蒸馏实验。SFT 在 50K 达到最高成功率，继续训练到 60K 没有带来提升。现有结果证明 BS32 足够，但没有进行 batch size 对照实验，因此不声称 BS32 是最优 batch size。
