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

## 3. ACPD-v2 方法

原 ACPD 的 cue 由 selector 和 predictor 共同学习，并且 selector 可以读取与 student 共享带噪动作和 flow time 的 teacher action tokens。ACPD-v2 将这一目标替换为 teacher attention 中可直接计算的固定视觉贡献。

### 3.1 问题定义

Teacher 使用特权视角 `agentview+wrist`，student 只使用 `backview`。两者接收相同的语言、机器人状态、带噪动作 $x_t$ 和 flow time $t$。Teacher 参数冻结，student 使用 LoRA 训练。

对于真实动作 $a$ 和噪声 $\epsilon$：

$$
x_t=t\epsilon+(1-t)a, \qquad u_t=\epsilon-a.
$$

Student 首先学习标准 flow matching：

$$
\mathcal{L}_{\mathrm{FM}}=\mathbb{E}\left[\lVert v_S(x_t,t)-u_t\rVert_2^2\right].
$$

### 3.2 提取 Teacher 的特权视觉贡献

在 action expert 的第 $l$ 层，对每个 action query，分别提取 agentview 和 wrist 对 attention residual 的贡献：

$$
C_{l,i}^v=G_{l,i}\odot W_l^O\left(\sum_{j\in v}P_l(i,j)V_l(j)\right),
\qquad v\in\{\mathrm{agent},\mathrm{wrist}\}.
$$

$i$ 表示 action token。$P_l$ 是 teacher 对全部有效 image、language 和 action key 计算的完整 softmax；视角内部不重新归一化。$W_l^O$ 和 $G_l$ 分别是 action expert 原有的 attention output projection 和 AdaRMS residual gate。因此，$C_l^v$ 是该视角在 teacher 原始 attention 计算中产生的真实加性 residual，而不是额外学习出的 cue。

### 3.3 Student 预测并使用特权贡献

Student 在同一层得到 backview action hidden state $h_l^S$。线性 predictor 分别预测两个不可见视角的贡献：

$$
(\hat C_l^{\mathrm{agent}},\hat C_l^{\mathrm{wrist}})=f_\phi^l(h_l^S).
$$

Predictor 使用按视角归一化的 MSE 训练：

$$
\mathcal{L}_{\mathrm{contrib}}
=\frac{1}{2}\sum_v
\frac{\mathbb{E}\lVert \hat C_l^v-\operatorname{sg}(C_l^v)\rVert_2^2}
{\mathbb{E}\lVert C_l^v\rVert_2^2+\varepsilon}.
$$

预测出的两个 residual 相加后，通过一个可学习标量 gate 注入 student 最终的 action representation：

$$
\tilde z_S=z_S+\tanh(\gamma)\operatorname{sg}
\left(\hat C_l^{\mathrm{agent}}+\hat C_l^{\mathrm{wrist}}\right).
$$

$\gamma$ 从 0 初始化，由 policy loss 学习是否以及多大程度使用预测贡献。停止梯度保证 predictor 只学习重建 teacher contribution，而不会通过 policy loss 寻找捷径。推理时不需要 teacher 或特权视角，student 仅根据 backview 生成并使用预测贡献。

### 3.4 完整训练目标

ACPD-v2 保留 ACL，使 student 与 teacher 的动作流方向保持一致：

$$
\mathcal{L}_{\mathrm{ACL}}=1-\rho(v_S,v_T),
$$

其中 $\rho$ 是每个样本在 action horizon 和 7 个有效动作维度上的 Pearson correlation。完整目标为：

$$
\mathcal{L}=\mathcal{L}_{\mathrm{FM}}
+0.2\mathcal{L}_{\mathrm{contrib}}
+0.5\mathcal{L}_{\mathrm{ACL}}.
$$

ACPD-v2 不再学习原来的 teacher cue selector，也不再使用原 Cue 的 variance regularizer。

## 4. 蒸馏层选择

### 4.1 实验设计

固定 teacher 和 Flow-only student，在 episode 级 held-out 数据上测试 layer 6--12。每层、每个 teacher 视角训练独立线性 probe，将 student 的 $h_l^S$ 映射为 teacher 的 $C_l^v$。每个 probe 使用 3 个初始化 seed、500 steps；验证集包含 256 个样本。

该实验只回答“student 是否能从 backview 恢复该层的特权贡献”，不训练 policy，也不测任务成功率。

### 4.2 指标计算

为排除共享动作和时间输入形成的捷径，将目标视角的 K/V 在 batch 内错配，同时保持 action query、带噪动作、flow time 和其他 key 不变，重新计算 teacher contribution $C_{l,\mathrm{shuffle}}^v$。

| 指标 | 计算 | 含义 |
|---|---|---|
| Correct cosine | $\cos(\hat C_l^v,C_l^v)$ | 预测与正确 teacher contribution 的方向一致性。 |
| Shuffled cosine | $\cos(\hat C_l^v,C_{l,\mathrm{shuffle}}^v)$ | 预测与错误视角 contribution 的方向一致性。 |
| Cosine gap | Correct cosine $-$ Shuffled cosine | gap 越大，预测越能恢复当前样本的特权视觉贡献，而不是只复现共享动作输入。 |
| Explained variance | $1-\mathrm{MSE}(\hat C,C)/\mathrm{MSE}(\bar C,C)$ | 是否优于始终预测验证集均值；大于 0 才说明恢复了样本差异。 |
| Hard gap | teacher 相对 student 的 flow error 优势最大的 25% 样本上的 cosine gap | 检查困难样本上的可恢复性。 |

Cosine gap 和 hard gap 的 95% CI 通过 episode bootstrap 计算。可用层需同时满足：overall gap $\geq0.10$、hard gap $\geq0.05$、两者 CI 下界大于 0、explained variance 大于 0。最佳层还需比次优层高至少 0.02。

### 4.3 层扫描结果

| Layer | Overall gap | Hard gap | Explained variance | 结论 |
|---:|---:|---:|---:|---|
| 6 | 0.0997 | 0.0706 | -0.1468 | 不可用 |
| 7 | 0.3261 | 0.2776 | 0.2833 | 通过 |
| 8 | 0.2858 | 0.2373 | 0.3420 | 通过 |
| 9 | 0.3417 | 0.3194 | 0.2822 | 通过 |
| **10** | **0.3992** | **0.3779** | **0.3556** | **选中** |
| 11 | 0.3508 | 0.3045 | 0.3113 | 通过 |
| 12 | 0.2367 | 0.2080 | 0.0903 | 通过 |

Layer 10 的 overall gap 比次优 layer 11 高 0.0485，超过预设的 0.02 门槛；其 hard gap 和 explained variance 也为所有候选层最高。因此，后续 policy 实验使用 layer 10。该结论只证明 layer 10 最容易恢复，不代表它已经提高任务成功率。

## 5. ACPD-v2 任务成功率验证

当前实验使用 layer 10、4 卡 FSDP、physical BS32、无梯度累积，训练 5K 后验证四套 LIBERO 共 2,000 episodes。训练正在运行，任务成功率尚无结论。

通过标准：相对 ACL-only 提升至少 1.5 个百分点，且配对 bootstrap 95% CI 下界大于 0。

## 6. 阶段性结论

1. 5090 已能稳定完成 pi0.5 LoRA SFT 和 ACPD 蒸馏，BS32 足够使用。
2. cosine SFT 的最佳 checkpoint 是 50K，pooled success 为 57.05%。
3. 原始 ACPD 的 5K 提升主要来自 ACL。
4. layer 10 的特权 attention contribution 最容易恢复，其任务效果仍在验证。
