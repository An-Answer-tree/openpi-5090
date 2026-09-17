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

| Checkpoint | 平均成功率 |
|---:|---:|
| 30K | 46.20% |
| 40K | 49.20% |
| **50K** | **57.05%** |
| 60K | 54.25% |

结论：BS32 已能稳定支持 SFT 和蒸馏实验。SFT 在 50K 达到最高成功率，继续训练到 60K 没有带来提升。

## 2. 原始 ACPD 诊断

### 2.1 训练 Loss 能否选择方案

| 问题 | 实验 | 结果 |
|---|---|---|
| 是否需要 layer 6+12 | 比较 layer 6、12、6+12，训练 5K | layer 6 loss 最低，但差异均小于 1%。 |
| Cue 或 ACL 是否加快收敛 | 比较 Flow、Cue、ACL、Full，训练 2K | 相对 Flow 的 loss 差异均小于 1%。 |

结论：双层 ACPD 没有表现出优势；训练 loss 不能判断 Cue 和 ACL 是否提高任务成功率。

### 2.2 成功率提升来自哪个组件

共同设置：backview student、BS32、5K、四套 LIBERO 共 2,000 episodes。

| 方法 | 平均成功率 | 结论 |
|---|---:|---|
| Flow-only | 4.45% | 基线 |
| ACL-only | 6.35% | 比 Flow-only 高 1.90 个百分点；差值的 95% 置信区间为 `[+0.75, +3.10]` 个百分点，整个区间大于 0，说明 ACL 的提升可信。 |
| Full ACPD | 6.50% | 比 ACL-only 高 0.15 个百分点；差值的 95% 置信区间为 `[-1.15, +1.40]` 个百分点，区间跨过 0，不能证明原 Cue 带来额外提升。 |

95% 置信区间表示配对 bootstrap 对“两个方法平均成功率差值”给出的不确定性范围。区间全部大于 0，表示有正向提升证据；区间跨过 0，表示现有结果无法排除两种方法没有差异。

结论：Full ACPD 优于 Flow-only，但提升主要由 ACL 解释。

## 3. ACPD-v2 训练目标

ACPD-v2 的训练目标是：让只观察 backview 的 student 恢复 teacher 中 agentview 和 wrist 对动作计算的视觉贡献，并利用这些贡献改善动作生成。

### 3.1 Flow Matching 目标

Teacher 使用特权视角 `agentview+wrist`，student 只使用 `backview`。两者接收相同的语言、机器人状态、带噪动作 $x_t$ 和 flow time $t$。Teacher 参数冻结，student 使用 LoRA 训练。

对于真实动作 $a$ 和噪声 $\epsilon$：

$$
x_t=t\epsilon+(1-t)a, \qquad u_t=\epsilon-a.
$$

Student 首先学习标准 flow matching：

$$
\mathcal{L}_{\mathrm{FM}}=\mathbb{E}\left[\lVert v_S(x_t,t)-u_t\rVert_2^2\right].
$$

### 3.2 特权视觉贡献目标

在 action expert 的第 $l$ 层，每个 action token 都会通过 attention 从图像 token 中读取信息。对于一个视角，将 action token 分配给该视角各图像 token 的 attention 权重乘以对应的 value 并求和，再经过该层原有的输出投影和门控，就得到这个视角实际加到 action token 上的信息向量。本文将该向量定义为该视角的视觉贡献：

$$
C_{l,i}^v=G_{l,i}\odot W_l^O\left(\sum_{j\in v}P_l(i,j)V_l(j)\right),
\qquad v\in\{\mathrm{agent},\mathrm{wrist}\}.
$$

$i$ 表示第 $i$ 个 action token，$j$ 表示图像 token。$P_l(i,j)$ 是 action token $i$ 对图像 token $j$ 的 attention 权重，$V_l(j)$ 是图像 token $j$ 携带的信息。求和项表示 action token 从视角 $v$ 读取到的信息；$W_l^O$ 将其映射到 action hidden state 的空间，$G_{l,i}$ 决定该信息实际加入 action token 的强度。

因此，$C_{l,i}^v$ 不是 attention 分数，而是与 action hidden state 同维度、在 teacher 前向计算中实际加入 action token 的向量。训练 student 恢复该向量，就是让 student 根据 backview 推断 teacher 从 agentview 和 wrist 获得的动作相关信息。$P_l$ 仍由全部有效 token 共同计算 softmax，只在求和时按视角拆分，因此两个视角的贡献可以相加并还原其在原 attention 输出中的对应部分。

### 3.3 特权贡献预测与注入

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
2. cosine SFT 的最佳 checkpoint 是 50K，平均成功率为 57.05%。
3. 原始 ACPD 的 5K 提升主要来自 ACL。
4. layer 10 的特权 attention contribution 最容易恢复，其任务效果仍在验证。
