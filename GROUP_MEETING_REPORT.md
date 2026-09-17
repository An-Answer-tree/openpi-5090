# ACPD 蒸馏实验组会报告

## 1. 训练环境与 SFT 基线

### 1.1 计算资源与 Batch Size

以下均为 pi0.5 LoRA 的实测配置。BS 表示全局 batch size。

| 训练类型 | 最少可运行 GPU | 4 卡无梯度累积的已验证 BS | 实测依据 |
|---|---:|---:|---|
| SFT | 1 | 32 | LoRA 单卡可运行；4 卡 BS32 已稳定完成 60K steps。 |
| ACPD 蒸馏 | 2 | 32 | 2 卡 BS32 已完成多组 2K/5K 实验；4 卡 BS32 已通过训练测试。 |

实测训练时间：SFT 4 卡 BS32 训练 30K 约 16 小时；ACPD 蒸馏 2 卡 BS32 训练 5K 约 4.5 至 9 小时。

### 1.2 SFT 训练步数对比

共同设置：4 卡 FSDP LoRA、global BS32、cosine 学习率。每个 checkpoint 在四套 LIBERO benchmark 上共验证 2,000 episodes。

| Checkpoint | Pooled success |
|---:|---:|
| 30K | 46.20% |
| 40K | 49.20% |
| **50K** | **57.05%** |
| 60K | 54.25% |

结论：BS32 已能稳定支持 SFT 和蒸馏实验。SFT 在 50K 达到最高成功率，继续训练到 60K 没有带来提升。现有结果证明 BS32 足够，但没有进行 batch size 对照实验，因此不声称 BS32 是最优 batch size。

## 2. 原始 ACPD 诊断

### 2.1 训练 Loss 能否选择方案

| 问题 | 实验 | 结果 |
|---|---|---|
| 是否需要 layer 6+12 | 比较 layer 6、12、6+12，训练 5K | layer 6 loss 最低，但差异均小于 1%。 |
| Cue 或 ACL 是否加快收敛 | 比较 Flow、Cue、ACL、Full，训练 2K | 相对 Flow 的 loss 差异均小于 1%。 |

结论：双层 ACPD 没有表现出优势；训练 loss 不能判断 Cue 和 ACL 是否提高任务成功率。

### 2.2 成功率提升来自哪个组件

共同设置：backview student、BS32、5K、四套 LIBERO 共 2,000 episodes。

| 方法 | Pooled success | 结论 |
|---|---:|---|
| Flow-only | 4.45% | 基线 |
| ACL-only | 6.35% | 比 Flow 高 1.90 点，95% CI `[0.75, 3.10]`。 |
| Full ACPD | 6.50% | 比 ACL-only 高 0.15 点，95% CI `[-1.15, 1.40]`。 |

结论：Full ACPD 优于 Flow-only，但提升主要由 ACL 解释。

## 3. 特权 Cue 改进

### 3.1 为什么修改 Cue

原 Cue 可能通过共享的动作输入完成预测，不能证明学生学到了特权视觉信息。改进后，学生预测 teacher 中 agentview 和 wrist 对 action token 的真实 attention contribution，并使用打乱视角 K/V 的结果作为负对照。该方案记为 ACPD-v2。

### 3.2 哪一层最适合蒸馏

在 held-out episodes 上扫描 action expert 的 layer 6--12。评价指标为正确视角与打乱视角的 cosine gap，并同时要求 explained variance 大于 0。

| 结果 | 数值 |
|---|---:|
| layer 6 | gap `0.0997`，explained variance `-0.1468`，不可用 |
| layer 10 | gap `0.3992`，explained variance `0.3556`，最优 |
| 次优 layer 11 | gap `0.3508` |

结论：特权 attention contribution 可以从 backview student 中恢复；layer 10 比次优层高 `0.0485`，因此用于后续任务验证。该结果尚不能证明任务成功率提高。

## 4. ACPD-v2 任务成功率验证

当前实验使用 layer 10、4 卡 FSDP、physical BS32、无梯度累积，训练 5K 后验证四套 LIBERO 共 2,000 episodes。训练正在运行，任务成功率尚无结论。

通过标准：相对 ACL-only 提升至少 1.5 个百分点，且配对 bootstrap 95% CI 下界大于 0。

## 5. 阶段性结论

1. 5090 已能稳定完成 pi0.5 LoRA SFT 和 ACPD 蒸馏，BS32 足够使用。
2. cosine SFT 的最佳 checkpoint 是 50K，pooled success 为 57.05%。
3. 原始 ACPD 的 5K 提升主要来自 ACL。
4. layer 10 的特权 attention contribution 最容易恢复，其任务效果仍在验证。
