# Research Findings

## 研究问题

如何利用 teacher 的 `agentview+wrist` 特权视觉信息，提高只看弱视角的 pi0.5 student？

正式 BS64 原始方法称为 **ACPD-v2-Fixed（固定权重）**，历史简称 H9-Fixed；从其
10K状态分支的权重消融称为 **ACPD-v2-Decay**，历史简称 H17-Decay；无注入消融称为
**ACPD-v2-NoInjection**，历史简称 H13-LossOnly。
下文历史 H9 BS32 单独注明。恢复与延长训练属于 H9-Fixed，不是新方法。

## 当前认识

| 问题 | 证据 | 结论 |
|---|---|---|
| 原始 Cue 是否有效 | Full ACPD 仅比 ACL-only 高 `0.15` 点，95% CI `[-1.15, 1.40]` | 没有检测到 Cue 的额外收益。 |
| 特权视觉贡献能否恢复 | layer 10 overall gap `0.3992`，比次优层高 `0.0485` | backview 表征可以预测 teacher 的真实 attention contribution。 |
| 两路贡献的恢复质量是否一致 | layer 10 agentview/wrist explained variance 为 `0.4722/0.2389`，贡献/总attention范数比为 `0.3122/0.5945` | wrist 目标更大但较难恢复；这是分路融合的设计动机，不是任务收益或显著层间差异的证明。 |
| layer 10 是否已被显著证明优于 layer 11 | H7.1 点估计差值 `0.0485`，但没有保存逐样本层间差值；分别的 bootstrap 区间不能替代配对检验 | layer 10 是预注册规则下的工程选择，层间显著性仍未验证；复核约需 6 小时、2 GPU，当前不优先。 |
| ACPD-v2 是否有正向任务信号 | BS32 5K 相对 ACL-only 提升 `5.00` 点，95% CI `[3.60, 6.45]` | 单 seed、5K 筛选支持 ACPD-v2。 |
| ACPD-v2 是否优于公平 SFT | BS64 5K 为 `23.15%` 对 `13.85%`；30K 为 `58.00%` 对 `60.45%` | ACPD-v2 有早期优势，但没有保持到 30K。 |
| ACPD-v2 在 10K 是否仍有优势 | ACPD-v2 `37.45%`，SFT `23.25%`；差值 `+14.20` 点，95% CI `[+11.90, +16.60]` | 10K 仍有明确早期优势。 |
| ACPD-v2 是否有早期步数效率 | 5K H9 `23.15%`对10K SFT `23.25%`，配对差值`-0.10`点、区间`[-2.20,+2.05]`点；历史训练用时约13小时55分对6小时35分 | 只在约23%成功率水平观察到少用一半优化步和样本呈现量的接近表现；这些运行未体现GPU时间优势，另两组跨step比较也未满足±3点范围。 |
| ACPD-v2 是否提高5K–30K检查点序列平均表现 | 六点梯形平均 H9 `45.995%`、SFT `40.860%`；差值`+5.135`点、配对回合重采样区间`[+4.080,+6.210]`点；10K–30K仍高`3.481`点 | 探索性再分析支持按步数衡量的过程收益；H9中段由同配置任务重建，非单次连续轨迹；30K终点仍低于SFT，历史运行也未体现GPU时间优势。 |
| ACPD-v2 在 20K 是否仍有优势 | ACPD-v2 `49.60%`，SFT `47.20%`；差值 `+2.40` 点，95% CI `[-0.15, +5.00]` | pooled 优势已不再可检测。Object 高 `9.60` 点、Goal 低 `8.20` 点；不能把套件差异归因于单一机制。 |
| 20K套件差异是否由少数任务驱动 | Object 20K有8/10任务为正、30K仅5/10；Spatial 20K与30K逐任务差值Spearman `-0.881` | 任务效应随训练阶段改变。不能直接用20K任务排序确定30K的固定路由；同seed探索性分析。 |
| 10K的早期优势是否仅由少数任务驱动 | 10K有32/40任务正差，固定任务回合重采样区间[27,34]；四套均至少7/10任务为正。到15K原32个正差任务仅20个保持，10K/15K逐任务差值Spearman `0.123` | 10K多数任务受益在现有回合重采样下较稳健，但逐任务排序随训练变化；同seed探索性分析，不能固定路由或推断注入因果作用。 |
| ACPD-v2 在 25K 是否仍有优势 | ACPD-v2 `55.75%`，SFT `55.80%`；差值 `-0.05` 点，95% CI `[-2.50, +2.45]` | 25K 已无可检测优势。 |
| 匹配 SFT 在 20K 的水平 | BS64 backview SFT 20K pooled success 为 `47.20%` | H9 20K 配对验证已完成。 |
| ACPD-v2 延长训练是否有效 | H9-scale-b 从 5K `23.15%` 提高到 30K `58.00%`，但匹配 SFT 30K 为 `60.45%` | 延长训练提高了绝对成功率，尚未证明最终优于 SFT。 |
| H9 是否在 25K 早于 30K 达峰 | 25K 为 `55.75%`，30K 为 `58.00%`；差值 `-2.25` 点，95% CI `[-4.70, +0.20]` | 不支持 25K 已过峰值。 |
| H9 在 30K 后是否继续提高 | 35K 为 `61.05%`，相对 30K 提升 `3.05` 点，95% CI `[+0.60, +5.50]`；40K 为 `59.45%`，相对35K差值`-1.60`点，95% CI `[-4.05, +0.80]` | 检测到30K--35K提高；尚不能确认35K后下降。 |
| H9 35K 是否优于同进度 SFT | H9 `61.05%`，SFT `57.15%`；探索性配对差值 `+3.90` 点，95% CI `[+1.45, +6.35]` | 35K 同进度优势成立；SFT 35K 低于自身30K，尚不能证明H9提高最终上限。 |
| SFT 35K 与自身30K比较 | `57.15%` 对 `60.45%`；配对差值 `-3.30` 点，95% CI `[-5.75, -0.85]` | 35K低于30K；后期轨迹仍非单调。 |
| SFT 60K 与自身30K比较 | `62.05%` 对 `60.45%`；配对差值 `+1.60` 点，95% CI `[-0.95, +4.05]` | 60K是已测SFT最高点，但未检测到确定的 pooled 增益。 |
| H9最高点是否超越SFT最高点 | H9 35K `61.05%` 对SFT 60K `62.05%`；探索性配对差值 `-1.00` 点，95% CI `[-3.45, +1.45]` | 不能证明ACPD-v2提高最终上限。 |
| 显式 residual 注入是否有效 | H9 为 `23.15%`，H13 loss-only 为 `20.60%`；差值 `+2.55` 点，配对 95% CI `[+0.40, +4.70]` | 达到预注册判据，支持显式注入。 |
| 放开预测贡献的动作梯度是否有效 | TDCA 5K 为 `21.35%`，H9-Fixed 为 `23.15%`；配对差值 `-1.80` 点，95% CI `[-3.90, +0.35]` | 未检测到 5K pooled 提升；不能据此断言总体更差。 |
| ACPD-v2 是否降低训练 flow MSE | 299 个对齐点与 SFT 的相关系数为 `0.9985`，全程平均 MSE 几乎相同 | 5K 成功率增益不是更低训练 MSE 的结果；内部表示改变是待验证的机制解释。 |
| 5K 优势是否对应特定 flow 时间段误差下降 | 首轮512样本整体差值近0；同样本全时间点复核128样本，t=0.1 差值 `+0.00283`、95% CI 跨0，整体差值 `+0.00341`、95% CI 跨0 | 首轮低时间段信号未复现；没有证据支持用时间重采样解释5K成功率优势。 |
| 辅助目标在后期是否仍有数值权重 | 首末窗口中 contribution/ACL 占总目标约 `61.39/16.69%` 与 `60.29/16.52%` | 辅助目标未自然消失；loss 占比不能判断其对成功率的因果作用。 |
| 后期辅助梯度是否更冲突 | 正式 BS32 中组合冲突率在 5K/30K 均为 `0%`，cosine 中位数为 `0.6279/0.5920` | 不支持后期辅助梯度冲突解释，不优先运行梯度投影。 |
| Contribution 是否可能形成后期约束 | 30K时其加权梯度范数为flow的`1.175`倍，cosine仅`0.0957`；25K--40K预测cosine从`0.7972`升至`0.8026` | 近正交梯度是否限制后期成功率仍是假设；由H17-Decay测试。 |
| 注入支路是否接收任务梯度 | predicted residual使用`stop_gradient`；flow不能经该支路更新predictor，能更新gate；共享主干仍接收flow梯度 | 注入缺少直接任务梯度是代码事实，是否导致后期优势消失尚无结论。 |
| H9推理注入是否改善局部动作MSE | 同checkpoint离线ON−OFF：10K `+0.00000834`、30K `+0.00000261`，按episode配对区间均跨0；注入速度变化比均值为`0.290%/0.187%` | 注入只造成小幅速度改变，未检测到局部MSE改善；不能从离线结果推断闭环成功率，等待10K同checkpoint仿真消融。 |
| gate作用是否在样本间异质 | 10K约54.12% episode受益、45.88%变差；30K约52.06%受益、47.94%变差，平均差值接近0 | 支持条件化 gate 的机制动机，但尚未证明 student 输入能预测正负方向；动态 gate 仍需 held-out 对照。 |
| gate可靠性是否存在可判别信号 | oracle correction cosine 与 `on-off` MSE 的 episode correlation 为10K `-0.8943`、30K `-0.8791`；cosine≥0时82.5%--90.3% episode受益，cosine<0时仅2.7%--3.7%受益 | 这是由MSE展开式得到的 oracle sanity check，不是独立预测证据；需学习不使用真实 target 的可观测可靠性估计器。 |
| gate幅度是否足以作为可靠性代理 | 注入改变量大小与`on-off` MSE的episode correlation为10K `0.0158`、30K `-0.0951` | 贡献/速度改变量大小不能单独决定 gate；应估计方向可靠性或两路一致性。 |
| H9训练中的teacher动作目标是否可靠 | 训练日志七个记录点的`teacher_better_ratio`为`0.9958--1.0000`，且`teacher_task_loss`始终低于`student_task_loss` | 提供动作级 teacher 信号有效的正向机制证据；统计来自训练 batch，未与SFT配对，也不能证明注入的因果收益。 |
| teacher信号强但后期成功率优势消失说明什么 | H9窗口中student/teacher task-loss比从`10.89×`降至`5.49×`，而H9相对SFT的成功率优势在25K--30K消失 | teacher目标质量不是唯一瓶颈；应优先改进特权信息到最终动作的转换接口，不能只增加teacher或contribution loss权重。 |
| 单视角 baseline 是否受视角影响 | left/right/top/backview 30K pooled 分别为 `78.65/77.00/71.55/60.45%` | 视角差异大；ACPD 必须使用相同 student 视角的 SFT 对照。 |

## 文献约束

Lopez-Paz et al. (2015) 将 privileged information 表述为只在训练阶段可见的
额外描述，并讨论了“先重建特权描述、再拼回 student”的直接方案可能比直接
蒸馏 teacher 输出更困难。Xiao et al. (NeurIPS 2024) 进一步形式化了部分可观测
环境中的边界：相同 student 观测可能对应不同真实状态和不同 expert 动作，直接
模仿不可辨识的 teacher 信息可能严格次优。它们与本项目的 H7/H18 结果相容：
contribution 可以被预测，不代表预测 contribution 在动作接口中有独立收益。
因此后续改进应优先检验与真实动作行为相关、且能由 student 观测解释的 teacher
信号，而不是无条件增加内部 contribution predictor 的容量。文献笔记：
`literature/privileged-distillation.md`。

## 方法判断

H17-Decay在15K完成Spatial/Object，成功率为41.80%/59.00%，比H9-Fixed低
7.60/5.40点。其余评测根据已观察结果提前停止；这是探索性部分结果，不支持
10K–15K权重0.2→0.05的中期收益，不证明所有调度无效。20K主比较仍未完成。
证据：`experiments/ablation/acpd-v2-h17-contribution-decay/analysis-15k.md`。

ACPD-v2 使用 teacher 真实 Q/K/V、完整 attention softmax、输出投影和 residual gate，
分别提取 agentview 与 wrist 对 action token 的贡献。Student 在 layer 10 预测两个
贡献向量，训练时使用 contribution loss，并在最终 hidden 上进行门控注入。

H7/H7.1 证明该目标可以从 backview 恢复；H9 与 H12 在 5K、10K 检测到 ACPD-v2
早期优势，但 30K 时 ACPD-v2 为 `58.00%`，匹配 SFT 为 `60.45%`，差值
`-2.45` 点，95% CI `[-5.00, 0.00]`。H13 表明 5K 时显式注入带来 `2.55`
点 pooled 增益。训练 loss 只用于健康检查，最终判断使用 benchmark 成功率。

匹配 loss 分析显示，SFT 与 ACPD-v2 的监督 flow MSE 轨迹几乎重合，但 ACPD-v2
在 5K 已有明显成功率优势。Contribution cosine 在 5K 达到 `0.7157`，之后缓慢
升至 30K 的 `0.7999`；gate 在约 12K 达峰后下降。25K 成功率为 `55.75%`，
低于 30K 的 `58.00%`。匹配 SFT 在 25K 为 `55.80%`，与 ACPD-v2 的
`55.75%` 无差异，因此早期增益在 25K 已消失；训练 loss、cosine 和 gate 都不能
替代任务成功率验证。

H15a 正式 BS32 诊断没有观察到 30K 组合冲突率上升：5K 与 30K 均为 `0%`，组合
cosine 中位数仍为正。该结果否定了优先测试 conflict-aware 梯度投影的依据。H9 35K
相对自身30K提高，且在探索性的同进度比较中高于SFT 35K；但SFT 35K低于自身30K，
因此尚未证明H9提高最终上限。H9 40K相对35K的下降区间跨0。20K pooled相对SFT
仅高2.40点且区间跨0，但Object明显受益、Goal明显下降；不能直接把套件差异归因于
contribution注入。H9续训在45K checkpoint完整后按用户决定停止。SFT 40K/50K/60K
已验证，分别为60.55%/61.30%/62.05%；H9最高观测点35K的61.05%未超过
SFT最高观测点60K的62.05%。

H9-Fixed 的 contribution 梯度后期仍较强且接近正交，注入支路又对预测向量使用
`stop_gradient`。主干表征仍随任务训练变化，所以不能把这一代码事实解释为预测向量
完全无法适应任务，也不能由gate下降推断实际注入量必然下降。

H9训练日志还显示，teacher action target 在七个记录点的 task loss 均低于 student，
`teacher_better_ratio` 为 `0.9958--1.0000`。这说明 teacher 的动作级信号在当前训练
batch 上具有稳定的任务相关性，是继续测试动作方向或动作特征融合的依据；但它不是
SFT对照、闭环验证或因果证据，不能单独解释 H9 的成功率变化。详见
`experiments/mechanism/acpd-v2-teacher-action-diagnostic/analysis.md`。
当前优先检验的假设是：固定contribution权重在后期形成不利约束。
H17-Decay只改变该权重；它不能同时证明不可恢复信息、LoRA容量或stop-gradient
就是唯一原因。学习率、loss数值及gate趋势均不能替代成功率验证。

## 工程约束

- 最终论文对照除 teacher 外统一使用 4 GPU、physical global BS64。
- 蒸馏 checkpoint 使用零起始目录编号；需要保留每 5K 时设置 `keep_period=1`。
- 精选 checkpoint 只在完整写入后建立硬链接，不移动或删除原 checkpoint。
- 活动任务的 checkpoint、数据集、脚本和输出路径不得修改。
- teacher 特征依赖 noisy action 和 flow time，不能预计算而不改变方法。
- 早期 supervised loss 不能代替任务成功率筛选 ACPD 组件。
- 只取教师速度修正中与真实 flow 误差平行的部分，在平方误差下等价于对
  原 flow 梯度做样本加权；不能把这种做法的收益归因于教师的额外信息。

## 待回答问题

- 20K 的 Object 正向和 Goal 负向差异是否来自任务条件下的 contribution 使用，
  还是训练随机性与基础模型差异？
- 已训练 H9 在推理时关闭注入，会否降低相同任务初始状态上的成功率？
- 公平 BS64 ACL-only 训练到 30K 后，ACPD-v2 的增益是否仍然成立？
- 10K 后将 contribution 权重平滑降至非零下限能否解除后期约束？H17-Decay
  在15K已测两套低于H9，20K全量验证按用户决定取消，主比较尚无结论。
- 为 detached contribution 增加只接收 flow 梯度的小型 adapter，能否提高注入的后期收益？

H18-ActionReadout 将该问题拆成冻结 H9 30K 的低成本接口诊断：hidden-only、
hidden-layer10 和 hidden-contribution 三个小型动作修正头共用冻结前向，仅比较 episode-held-out
flow MSE。它不更新 LoRA，不加载 teacher，不是闭环成功率实验。

H18 已完成：hidden-contribution 相对冻结 H9 的 episode mean flow MSE 下降 `0.813%`，
但按冻结 H9 归一化的配对95% CI 为 `[-1.804%, +0.031%]`；相对 hidden-layer10 对照反而变差 `0.029%`，
且区间跨0。该结果不支持直接增加动作修正 head，也不支持立即增加 contribution predictor
复杂度。它只说明当前 contribution 接口没有可检测的独立动作收益，不否定 H7/H9 已观察到的
contribution 可恢复性或 H9 早期仿真收益。详见
`experiments/mechanism/acpd-v2-h18-action-readout/analysis.md`。

同 checkpoint 离线注入诊断在10K/30K均未检测到7维flow MSE改善。它使“小gate
未必改变动作”的疑问具体化：速度确实改变，但样本级改变量均值不足0.3%。
该结果不能判定闭环增益是否来自训练期辅助目标；10K推理关闭注入的正式仿真仍待完成。

进一步复用该诊断的逐样本四分支误差：10K 的 wrist 单路−agentview 单路
MSE 为 `+0.00002726`，配对区间 `[+0.00000761,+0.00004680]`；30K 差值
`+0.00001012` 的区间跨零。两点的完整注入相对 agentview 单路和两路交互项
区间也均跨零。该探索性结果只提示10K时很小的局部差异，不能支持稳定的
分路gate或wrist有害的结论；见
`experiments/mechanism/acpd-v2-gate-effect/view-branch-analysis.md`。
