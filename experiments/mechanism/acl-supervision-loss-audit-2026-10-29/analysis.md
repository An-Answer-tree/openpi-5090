# ACL、PCD 训练损失与梯度方向核验

日期：2026-10-29。类型：已有日志与机制诊断的探索性再分析。
没有运行新训练、GPU 评测或梯度计算。

## 主要结论

| 问题 | 实际证据 | 可支持的结论 |
|---|---|---|
| ACL-only 是否加快 flow loss 收敛 | 与 SFT 对齐 349 个日志点；全程均值 0.0241201 对 0.0241146，差 +0.0226%；相关系数 0.99836 | 拟合轨迹几乎相同，未观察到更快收敛 |
| ACL 目标是否仍在优化 | 首末窗口 ACL loss 从 0.05705 降至 0.02533，下降 55.60%；对应平均相关性从 0.94295 增至 0.97467 | 训练中仍在改善教师—学生的相对速度一致性；没有 SFT 对应指标，不能确定相对 SFT 的额外改善 |
| ACL 与 flow 梯度是否相容 | 完整 ACPD 早、晚检查点上的梯度 cosine 中位数为 0.92084、0.90120；各 200 批均未出现反向 | 在所测共享 LoRA 参数和批次上，ACL 提供与真实动作监督同向的更新 |
| PCD 是否只是重复 flow 的梯度 | cosine 中位数为 0.16993、0.09566；加权范数比为 1.23863、1.17537 | PCD 提供明显不同、主要接近正交的参数约束，不能把它视为简单重复 flow，也不能仅据此判断其任务收益 |
| 合成后是否反向 | 加权辅助梯度与 flow 的 cosine 为 0.62794、0.59199；总训练梯度为 0.85072、0.83801 | 所测辅助目标的组合仍保持正向，不支持将其概括为整体梯度冲突 |

## ACL-only 与 SFT 的训练 loss

比较相同 backview、初始化、训练 seed、BS64、LoRA 和学习率设置。
将 SFT 的三个连续训练日志合并，再按记录的 optimizer step 对齐至 34,900。
排除初始化的 step 0，共 349 个共同日志点。

| 日志步数窗口 | 点数 | SFT flow MSE | ACL-only flow MSE | 相对差异 | ACL loss | 1 − ACL loss |
|---|---:|---:|---:|---:|---:|---:|
| 0–5K | 49 | 0.037453 | 0.037394 | −0.158% | 0.057053 | 0.942947 |
| 5–10K | 50 | 0.027380 | 0.027498 | +0.431% | 0.041016 | 0.958984 |
| 10–15K | 50 | 0.024764 | 0.024812 | +0.194% | 0.036678 | 0.963322 |
| 15–20K | 50 | 0.022464 | 0.022382 | −0.365% | 0.032670 | 0.967330 |
| 20–25K | 50 | 0.020312 | 0.020340 | +0.138% | 0.029332 | 0.970668 |
| 25–30K | 50 | 0.018728 | 0.018758 | +0.160% | 0.026786 | 0.973214 |
| 30–35K | 50 | 0.017968 | 0.017922 | −0.256% | 0.025330 | 0.974670 |

各窗口 flow MSE 差异绝对值均小于 0.44%。该现象适合说明 ACL 的作用
不是明显降低训练期 flow MSE 或加快该目标的拟合，不能据此宣称提高泛化。
这些是训练批次日志，数值保留四位小数；不是冻结模型后的 held-out 统一重测。

`1 − ACL loss` 来自实现中的平均 Pearson correlation。教师和学生共享带噪动作及
flow time，且 SFT 没有记录该相关性。因此高相关性本身不是独立的动作精度证明。

## 完整 ACPD 的梯度方向

使用已完成的 H15a 诊断：H9/ACPD-v2 的两个 BS64 训练检查点（5K、30K），
各在相同的 200 个 BS32 诊断批次上计算梯度，不更新参数。
比较范围为教师与学生监督共同影响的学生 LoRA 参数；不包括 predictor 和 gate。
PCD 使用历史记录名 `contribution`，权重为 0.2；ACL 权重为 0.5；flow 权重为 1。

下表均为逐批指标的中位数，反向比例除外。范数比已包含上述损失权重。

| 梯度项 | 与 flow 的 cosine：5K | 30K | 范数 / flow：5K | 30K | 反向比例：5K / 30K |
|---|---:|---:|---:|---:|---:|
| ACL | 0.92084 | 0.90120 | 0.84397 | 0.85132 | 0/200 / 0/200 |
| PCD | 0.16993 | 0.09566 | 1.23863 | 1.17537 | 4/200 / 5/200 |
| 加权辅助合成：PCD + ACL | 0.62794 | 0.59199 | 1.62586 | 1.50108 | 0/200 / 0/200 |

新增代数再分析：

- 在每批上将 flow 与辅助项相加，得到总训练梯度与 flow 的 cosine 中位数为
  **0.85072 / 0.83801**。
- 由两项范数与和的范数反推出 PCD–ACL cosine，中位数为 **0.16870 / 0.09656**。
- 梯度投影的线性恒等式最大误差分别为 `3.85e-7 / 3.20e-7`，通过数值核验。

上述新增值先逐批计算，再取中位数，没有将各指标中位数代入公式替代逐批计算。
总训练梯度由 `flow + 0.2 PCD + 0.5 ACL` 构成，仍限于同一共享 LoRA 参数子集。

## 对论文叙述的价值

最合适的角度是**优化方向的相容性与差异**：ACL 沿着与动作监督接近的方向更新；
PCD 施加不同方向的视觉表征约束；二者组合后，辅助更新仍与 flow 正向对齐。
PCD 接近正交不等于无效，ACL 高度同向也不等于新增信息已经得到独立证明。
这与两种监督承担不同角色的设计动机一致，闭环收益由任务成功率及融合消融支持。

适合作为后续写作参考的英文表述：

> Gradient diagnostics show that ACL closely aligns with the ground-truth flow gradient, while PCD provides a largely orthogonal update. Their combined auxiliary gradient remains positively aligned with flow supervision. This pattern is consistent with using action correlation and privileged-cue prediction as complementary training objectives.

这段诊断文字尚未加入论文。若采纳，应同时交代共享参数子集和诊断批次的设置。

## 证据与复现

- 复现：`python experiments/mechanism/acl-supervision-loss-audit-2026-10-29/analyze.py`
- [逐点 loss 数据](results/matched_losses.csv)、[汇总与来源](results/summary.json)。
- 原梯度诊断：[协议](../acpd-v2-h15a-gradient-conflict/protocol.md)、
  [分析](../acpd-v2-h15a-gradient-conflict/analysis.md)。
- 原始梯度数据：`/opt/liutong/openpi-5090-research/acpd-v2-gradient-conflict/{h9-5k,h9-30k}/{summary.json,per_batch.jsonl}`。
- 训练日志的精确路径均保存在汇总 JSON；脚本仅打开指定文件，不扫描目录。
