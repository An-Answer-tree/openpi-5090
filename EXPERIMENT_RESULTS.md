# 实验结果台账

更新时间：2026-10-08（CST）

本文件只记录实际运行的配置、指标、结论和证据。详细协议与分析见
[`experiments/README.md`](experiments/README.md)。工程故障不作为实验结果。

## 当前结论

本文正式 BS64 结果中的 H9 统一称为 **ACPD-v2-Fixed（Attention Contribution
Privileged Distillation，固定权重）**。新实验称为 **ACPD-v2-Decay**；两者关系如下。

| 规范名称 | 历史简称 | Contribution 权重 | ACL 权重 | 区别 |
|---|---|---|---:|---|
| ACPD-v2-Fixed | H9-Fixed / H9-scale-b | 全程 0.2 | 0.5 | 原始 ACPD-v2 |
| ACPD-v2-Decay | H17-Decay | 10K–15K：0.2→0.05；之后 0.05 | 0.5 | 只改变 contribution 权重，训练至20K |
| ACPD-v2-NoInjection | H13-LossOnly | 全程 0.2 | 0.5 | 去除 residual 注入，保留两个蒸馏 loss |
| ACPD-v2-TDCA | TaskAdapt（新规范名） | 全程 0.2 | 0.5 | 训练时允许动作损失更新 contribution predictor |
| ACPD-v2-FeatureFusion | H19 | 全程 0.2 | 0.5 | 预测两路贡献与最终 action hidden 融合，替代 scalar gate 注入 |

上述实验均为 backview、layer 10；“Fixed/Decay”指 contribution loss 权重，
不是学习率。H9-Fixed 与 H17-Decay 均沿用原 30K cosine LR。
H9-recovery、mid-trajectory、trajectory 分别是 H9-Fixed 的轨迹恢复、验证和续训，
不是新的方法。早期 BS32 的 H9 保留原编号，并明确标注 BS32。

规范命名总表见 [`experiments/README.md`](experiments/README.md#学术命名)。

| 问题 | 结论 | 证据 |
|---|---|---|
| 4×5090 能否训练 pi0.5 LoRA | 可以。SFT 与 ACPD-v2 均已稳定运行 physical global BS64。 | H2、H9-scale-b |
| 早期 BS32 SFT 最佳点 | cosine 50K，四套 pooled success `57.05%`；不能代替 BS64 公平基线。 | 4×500 episodes |
| 当前 backview BS64 SFT 最高观测点 | 60K `62.05%`；相对自身30K `+1.60` 点，配对95% CI `[-0.95, +4.05]`，未检测到确定提升。 | 四套各500回合 |
| 原始 ACPD 的 Cue 是否有效 | 没有可靠证据。Full 仅比 ACL-only 高 `0.15` 点，95% CI `[-1.15, 1.40]`。 | H5、H8 |
| 哪层 exact contribution 最可恢复 | layer 10；overall gap `0.3992`，比次优 layer 11 高 `0.0485`。 | H7/H7.1 |
| Layer 10 两路视觉贡献是否同样易恢复 | agentview/wrist explained variance 为 `0.4722/0.2389`，贡献范数比为 `0.3122/0.5945`；仅描述性证据，不代表任一路的任务收益。 | H7.1 原始指标 |
| ACPD-v2 是否优于匹配 BS64 SFT | 5K 时提升 `9.30` 点；30K 时为 `58.00%` 对 `60.45%`，差值 `-2.45` 点，95% CI `[-5.00, 0.00]`。早期优势未保持到 30K。 | H9-scale-b、H12 |
| ACPD-v2-FeatureFusion 是否可替代 H9 | H19在5K/10K/25K比H9低`1.95/0.90/0.40`点，30K高`2.10`点；30K配对95%区间`[-0.40,+4.60]`点，相对SFT为`-0.35`点、区间`[-2.95,+2.20]`点。当前保留H9主方案，H19为候选，未证明稳定优势或提高上限。 | [H9/H19比较](experiments/student/acpd-v2-h19-feature-fusion-30k/analysis.md)；四套各500回合 |
| H19 是否改善训练loss与贡献预测 | 与H9对齐299点，监督loss相关系数`0.999840`，各5K窗口均值差异不足`0.17%`；25–30K贡献cosine为`0.796822/0.797180`（H19/H9）。没有更快收敛或更好贡献预测的证据。 | [相同步数loss比较](experiments/student/acpd-v2-h19-feature-fusion-30k/analysis.md#训练-loss) |
| ACPD-v2 的早期优势在 10K 是否仍存在 | 存在。H9 为 `37.45%`，SFT 为 `23.25%`；差值 `+14.20` 点，配对 95% CI `[+11.90, +16.60]`。 | 相同 2,000 episodes |
| 10K 优势是否覆盖四个 benchmark suite | 是。Spatial/Object/Goal/LIBERO-10 分别高 `+18.2/+12.8/+18.4/+7.4` 点，四个 suite 的配对 95% CI 均严格高于 0：`[+12.6,+23.6]`、`[+7.2,+18.2]`、`[+13.8,+22.8]`、`[+4.6,+10.2]`。 | H9/SFT 10K 全量验证 |
| 15K 优势是否仍覆盖四个 benchmark suite | 四套点估计均为正：`+5.8/+6.4/+6.2/+4.4` 点；Object 和 Goal 的区间严格高于 0，Spatial 与 LIBERO-10 下界为 0。 | H9/SFT 15K 全量验证 |
| ACPD-v2 能否少用训练步数达到早期成功率 | H9 5K `23.15%` 对 SFT 10K `23.25%`，差值 `-0.10` 点、配对95%区间 `[-2.20,+2.05]` 点，落入探索性±3点范围；其余两组不满足。历史训练用时约13小时55分对6小时35分，未体现GPU时间优势。 | 三组跨step配对验证与训练日志 |
| ACPD-v2 的检查点序列平均成功率是否更高 | 5K–30K 六点梯形平均为 H9 `45.995%`、SFT `40.860%`，差值 `+5.135` 点，配对回合重采样95%区间 `[+4.080,+6.210]` 点；10K–30K差值仍为`+3.481`点。H9中段为同配置重建，不是单次连续运行；探索性，不代表最终上限或GPU时间收益。 | `experiments/mechanism/acpd-v2-learning-curve-area/analysis.md` |
| 5K–30K 的过程优势是否覆盖四个 suite | 梯形面积差均为正：Spatial `+6.56` 点 `[+4.26,+8.84]`、Object `+8.56` 点 `[+6.38,+10.76]`、Goal `+2.44` 点 `[+0.42,+4.48]`、LIBERO-10 `+2.98` 点 `[+1.10,+4.86]`。 | 六个 checkpoint 的配对回合重采样；支持过程平均优势，不代表最终上限 |
| ACPD-v2 在 15K 是否仍优于 SFT | H9-Fixed 为 `46.60%`，SFT 为 `40.90%`；差值 `+5.70` 点，配对 95% CI `[+3.25, +8.15]`。 | 相同 2,000 episodes |
| ACPD-v2 的早期优势何时消失 | 20K 时差值 `+2.40` 点，95% CI `[-0.15, +5.00]`，已无可检测 pooled 优势；25K 为 `-0.05` 点，95% CI `[-2.50, +2.45]`。 | 相同 2,000 episodes |
| ACPD-v2 在20K的任务差异 | Object 比 SFT 高 `9.60` 点，Goal 低 `8.20` 点；两项配对区间均不跨0。pooled 的 `+2.40` 点区间跨0。 | H9/SFT 20K 配对分析 |
| 20K差异是否集中在少数任务 | Object 20K 有8/10任务为正，30K为5/10；Spatial 20K与30K逐任务差值Spearman `-0.881`。方向随训练变化，固定任务路由缺乏依据。 | 20/25/30K逐任务日志分析 |
| 10K早期优势是否覆盖多数任务 | 10K有32/40任务高于匹配SFT，回合重采样区间[27,34]；四套均至少7/10任务为正。15K原32个正差任务仅20个仍为正。 | 5/10/15K逐任务日志分析；固定任务回合重采样 |
| ACPD-v2 从 5K 继续训练是否有效 | 同一 H9-scale-b 训练在 30K 达到 `58.00%`，比 5K 高 `34.85` 点；匹配 SFT 30K 为 `60.45%`，尚未证明最终优势。 | 4×500 episodes |
| H9 25K 是否早于 30K 达峰 | 不支持。25K 为 `55.75%`，30K 为 `58.00%`；25K-30K 为 `-2.25` 点，95% CI `[-4.70, +0.20]`。 | 相同 2,000 episodes |
| H9 在 30K 后是否继续提高 | 35K 为 `61.05%`，比 30K 高 `3.05` 点，配对 95% CI `[+0.60, +5.50]`。但相对 SFT 30K 仅高 `0.60` 点，95% CI `[-1.90, +3.05]`，尚不能证明最终优于 SFT。 | 相同 2,000 episodes |
| H9 与同进度 SFT 在 35K 的比较 | H9 `61.05%`，SFT `57.15%`；探索性配对差值 `+3.90` 点，95% CI `[+1.45, +6.35]`。SFT 35K 比自身 30K 低 `3.30` 点，不能据此断言 H9 提高了最终上限。 | 相同 2,000 episodes |
| H9 的最佳观测点是否超过 SFT 的最佳观测点 | H9 35K `61.05%` 对 SFT 60K `62.05%`；探索性配对差值 `-1.00` 点，95% CI `[-3.45, +1.45]`。未证明 ACPD-v2 提高最终上限。 | 相同 2,000 episodes；不同训练进度 |
| H9 40K 是否继续高于 35K | 40K `59.45%`，35K `61.05%`；探索性配对差值 `-1.60` 点，95% CI `[-4.05, +0.80]`，未检测到确定的下降。 | 相同 2,000 episodes |
| 显式 contribution 注入是否有效 | 有正向证据。H9 为 `23.15%`，H13 loss-only 为 `20.60%`；差值 `+2.55` 点，配对 95% CI `[+0.40, +4.70]`。 | H13 |
| 放开 contribution predictor 的动作梯度能否提高 5K 成功率 | TDCA 为 `21.35%`，H9-Fixed 为 `23.15%`；差值 `-1.80` 点，配对 95% CI `[-3.90, +0.35]`。未检测到提高。 | TDCA 5K，2,000 episodes |
| ACPD-v2 是否通过降低训练 MSE 获益 | 没有该证据。与 SFT 对齐的 299 个监督 loss 点相关系数为 `0.9985`，全程均值几乎相同；5K 成功率增益不能由更低训练 MSE 解释。 | H9/H12 loss 对齐 |
| 辅助目标在 30K 前是否自然消失 | 没有。加权 contribution/ACL 从首个到末个窗口下降 `53.42/53.04%`，但在总目标中的窗口占比保持约 `59--61%/16--17%`。 | H9 0--30K loss |
| 后期是否出现明显辅助梯度冲突 | 不支持。正式 BS32 中组合冲突率在 5K/30K 均为 `0%`，cosine 中位数为 `0.6279/0.5920`。 | H15a，200 个成对 batches |
| Contribution 后期梯度是什么方向 | 30K 加权梯度/flow范数比中位数`1.175`，cosine中位数`0.0957`，接近正交；是否限制任务成功率尚无结论。 | H15a；工作假设见`findings.md` |
| H9 的小 gate 是否实际改善离线动作预测 | 同checkpoint完整注入−关闭注入的7维flow MSE：10K为`+0.00000834`、30K为`+0.00000261`，配对区间均跨0；样本级速度变化比均值为`0.290%/0.187%` | 预测速度确有小幅改变，但未检测到离线MSE改善；不能替代同checkpoint闭环验证。 |
| H9 两路预测贡献是否存在稳定的局部误差差异 | wrist单路−agentview单路 MSE：10K `+0.00002726`，95% CI `[+0.00000761,+0.00004680]`；30K `+0.00001012`，区间跨0 | 探索性分路分析；10K有微小差异，30K未复现，两路交互未检出；不能据此确定分路gate有闭环收益。 |
| H9 gate 作用是否存在 episode 异质性 | 复用10K/30K逐样本 `on-off` MSE，按194个 episode 统计正负方向 | 无新Slurm任务 | 完成；探索性 | 10K约52.06% episode受益、47.94%变差；30K约54.12%受益、45.88%变差；支持条件化 gate 假设但不证明动态 gate 有效 |
| H9 gate 理想选择上限 | 每个 episode 用真实 flow target 选择关闭或完整注入的较小 MSE | 无新Slurm任务 | 完成；oracle诊断 | 10K理想 MSE 比关闭注入低 `0.00005999`，95%区间 `[-0.00007270,-0.00004780]`；30K低 `0.00004081`，区间 `[-0.00004926,-0.00003281]`。存在小的条件化局部收益上限，但不可直接部署 |
| H9 gate 可靠性 oracle 诊断 | 复用 gate probe 的真实 flow target，按 oracle correction cosine 分组 | 无新Slurm任务 | 完成；探索性 | episode correlation 为10K `-0.8791`、30K `-0.8943`；该相关性由MSE展开式预期得到，只能作为 oracle sanity check，不能证明可部署动态 gate |
| H9 gate 幅度代理诊断 | 复用 gate probe 的速度改变量大小与 `on-off` MSE | 无新Slurm任务 | 完成；探索性 | episode correlation 仅为10K `-0.0951`、30K `0.0158`；不支持仅按贡献范数调 gate |

## 正式 BS64 实验

除 teacher 外，最终论文训练统一使用 4 GPU、physical global BS64、无梯度累积。

| ID | 目的 | 实际配置 | Job | 状态 | 结果 |
|---|---|---|---:|---|---|
| Teacher | 提供 agentview+wrist 特权信息 | 全量 SFT，30K | 历史任务 | 完成 | checkpoint `29999` |
| H12 | backview 单视角 baseline | LoRA，BS64，5K 后确定性续训至 60K | 130285/130491/130762/132791；35K验证134992/134995；后期验证134998 | 60K训练与40K/50K/60K验证完成 | 5K `13.85%`；20K `47.20%`；25K `55.80%`；30K `60.45%`；35K `57.15%`；40K `60.55%`；50K `61.30%`；60K `62.05%` |
| H14-top | topview baseline | LoRA，BS64，30K，每 5K 保存 | 130669/130890 | 完成 | 30K pooled `71.55%` |
| H14-left | leftview baseline | LoRA，BS64，30K，每 5K 保存 | 130670/130891 | 完成 | 30K pooled `78.65%` |
| H14-right | rightview baseline | LoRA，BS64，30K，每 5K 保存 | 130671/130892 | 完成 | 30K pooled `77.00%` |
| H9-Fixed | 固定权重 ACPD-v2 对照 | layer 10，BS64，contribution=0.2，ACL=0.5 | 129728/130773 | 30K完成；恢复与续训见下表 | 5K `23.15%`；10K `37.45%`；15K `46.60%`；20K `49.60%`；25K `55.75%`；30K `58.00%`；35K `61.05%`；40K `59.45%` |
| ACPD-v2-Fixed leftview | 检验 H9-Fixed 在 leftview student 上的效果 | layer 10，4卡 BS64，累计1，loss=`1.0/0.2/0.5`，35K，每5K保留 | 训练139443；5K验证139971/139972 | 训练中；5K四套验证完成 | 5K：Spatial `25.40%`，Object `62.60%`，Goal `49.60%`，LIBERO-10 `34.20%`，Pooled `42.95%`（859/2000）；35K尚无结论 |
| ACPD-v2-Fixed rightview | 检验 H9-Fixed 在 rightview student 上的效果 | 同上，仅改变 student 视角 | 训练139444；5K验证139973/139974 | 训练中；5K四套验证完成 | 5K：Spatial `59.40%`，Object `65.60%`，Goal `41.60%`，LIBERO-10 `23.80%`，Pooled `47.60%`（952/2000）；35K尚无结论 |
| ACPD-v2-Fixed topview | 检验 H9-Fixed 在 topview student 上的效果 | 同上，仅改变 student 视角 | 训练139445；5K验证139975/139976 | 训练中；5K四套验证完成 | 5K：Spatial `34.00%`，Object `30.60%`，Goal `38.80%`，LIBERO-10 `7.80%`，Pooled `27.80%`（556/2000）；35K尚无结论 |
| H19-FeatureFusion | 检验预测视觉贡献在动作输出前的特征融合能否提高最终成功率；观察 30K 后短期轨迹 | backview，layer 10，从 pi0.5 base 开始；4卡 FSDP LoRA，BS64，flow/contribution/ACL=1/0.2/0.5；0–30K每5K保存；从 `29999` 完全续训至35K，每1K保存且全部保留 | smoke 136339；正式136345；35K续训141714（替换139791）；5K验证138746/138748；10K验证138747/138749；25K验证139046/139049；30K验证139418/139419 | 30K checkpoint 与 5K/10K/25K/30K 四套验证完成；35K续训暂停（JobHeldUser），等待用户绑定reservation后释放 | 已有结果：5K `21.20%`，10K `36.55%`，25K `55.35%`，30K `60.10%`；35K 尚无成功率结论 |
| H13-LossOnly | 判断 contribution 注入是否有效 | H9-Fixed 去除 residual 注入，BS64，5K | 130599/130774 | 完成 | pooled `20.60%`；H9-Fixed 高 `2.55` 点，配对 95% CI `[+0.40, +4.70]` |
| ACPD-v2-TDCA | 检验动作损失是否能改善 contribution 注入适配 | backview，layer 10，BS64，flow=1.0、contribution=0.2、ACL=0.5；仅开放 predictor 的动作梯度，训练至5K | 135943；验证136221；汇总136222 | 5K 训练及四套全量验证完成 | `21.35%`；比 H9-Fixed 低 `1.80` 点，95% CI `[-3.90, +0.35]`；不支持提升 |
| ACL-only BS64 | 检验 contribution 学习和注入在 ACL 之外的增益 | backview，layer 10，BS64，flow=1.0、ACL=0.5、contribution=0、关闭注入；训练至35K | 135724；25K验证137607/137608；30K验证135725、分析135729；35K验证135726、分析135728 | 35K训练及25K/30K/35K四套验证完成 | 25K `52.90%`；30K `54.35%`；35K `59.35%`。相对匹配SFT差值分别 `-6.10/-6.10/+2.20` 点；30K为负，35K区间跨0，未证明稳定提升 |
| H17-Decay | 检验后期 contribution 权重是否过强 | H9-Fixed 10K完整状态→20K；BS64；10K–15K权重0.2→0.05，ACL=0.5 | 134422/136119/134423/134424 | 15K仅Spatial/Object完成；根据部分结果取消20K验证及配对分析 | 15K两套41.80%/59.00%，比H9低7.60/5.40点；不支持该调度的中期收益；20K无最终结论 |

### H9-Fixed 轨迹任务

| 阶段（原标签） | 作用 | Job | 状态/已得结论 |
|---|---|---|---|
| 恢复（H9-recovery） | 重建被清理的早期 checkpoint，提供 H17 的同源10K起点与20K对照 | 130704/132198/132398 | 10K、15K、20K checkpoint 完整 |
| 中期验证（H9-mid-trajectory） | 测量10K/15K/20K/25K优势变化 | 132731/132732/134008/134009/132082/132084 | 10K相对SFT +14.20点；15K +5.70点；20K +2.40点（区间跨0）；25K -0.05点 |
| 延长训练（H9-trajectory） | 原计划30K→60K；45K checkpoint完整后停训，验证35K/40K | 132390/132720/132724/134371/134372；取消135001/135002 | 35K为61.05%；40K为59.45%；45K验证已取消，尚无结论；50K/60K未训练 |

H17-Decay 的主比较为同源 **H9-Fixed recovery 20K**，次比较为 SFT BS64 20K。
H17 15K只有Spatial/Object部分结果，无四套pooled；详见
`experiments/ablation/acpd-v2-h17-contribution-decay/analysis-15k.md`。
20K主比较尚无结论，现有H9数值不属于H17。

## 当前机制实验

| ID | 目的 | 实际配置 | Job | 状态 | 结论 |
|---|---|---|---:|---|---|
| H15a | 判断 ACPD-v2 后期是否存在辅助梯度干扰 | H9 5K/30K；每点200个相同BS32 batch；不更新参数 | smoke 132308；快速 132311；正式 132250 | 完成 | 5K/30K 组合冲突率均为 `0%`；不支持后期梯度冲突解释 |
| H18-ActionReadout | 检查冻结 H9 的 contribution 是否能被小型动作修正头有效使用 | H9 BS64 30K 冻结；hidden-only、hidden-layer10、hidden-contribution 三组；单卡500步、BS8 | smoke 136129；正式136130 | 完成；无仿真 | contribution 相对冻结 H9 MSE 下降 `0.813%`，95% CI `[-1.804%, +0.031%]`；相对 layer10 对照变差 `0.029%`，未通过1%筛选 |
| H18-ActionReadout-Oracle | 判断不同状态是否需要不同动作读出接口 | 复用 H18 的 1,024 个样本、194 个 episode；逐 episode 用真实 flow target 选择四种读出头中的最优者；CPU bootstrap 2,000次 | 无新 Slurm 任务 | 完成；不可部署上限 | episode-level 理想选择相对最佳固定 hidden-layer10 降低 MSE `2.131%`，95% CI `[-2.264%, -1.492%]`；支持条件路由假设，不是仿真成功率结论 |
| H18-ActionReadout-Oracle-Holdout | 检查 H18 oracle 是否能在未参与选择的样本上复现 | 每个 episode 交替一半样本选头、另一半测试，交换两半；179 个 episode；CPU bootstrap 2,000次 | 无新 Slurm 任务 | 完成；未通过 | holdout oracle 相对固定 layer10 变差 `0.027%`，MSE差值95% CI `[-0.00047134,+0.00055144]`；原 `2.131%` 主要是同样本选择上限 |
| H18-ObservableRouter | 检验 Student 可见特征能否预测动作读出分支 | Job 136935 提取 H18 同一验证划分的特征；CPU ridge 路由，episode 奇偶两折，正则 `1e-2` | 136935 | 完成；未通过 | 相对固定 hidden-layer10 变差 `0.127%`，MSE差值95% CI `[-0.00029520, +0.00053006]`；简单 Student-only 路由不能恢复 oracle 上限 |
| H18-ActionReadout-Ensemble | 检查读出头误差互补是否可由固定平均利用 | 复用 136935 特征；50% hidden-only + 50% hidden-contribution；另做两折 alpha 网格探索 | 无新 Slurm 任务 | 完成；未通过 | 固定平均相对 layer10 变差 `0.036%`，MSE差值95% CI `[-0.00029722,+0.00036291]`；不提交正式融合验证 |
| Flow-time 5K | 检查早期优势是否对应特定时间段的 flow 误差 | 冻结 SFT/H9 5K；首轮512样本，复核128样本且每样本5个 t；单卡只读 | 136591、136595 | 完成；无仿真 | 首轮整体无差异；首轮 t=0.1 信号在同样本复核中未复现（差值 `+0.00283`，95% CI 跨0）；不支持启动时间重采样训练 |
| H9 10K 推理期注入消融 | 同一个 H9 checkpoint 仅在推理时关闭 contribution 注入，判断部署时是否依赖该分支 | 四套各500回合；与原H9结果配对 | 136866、136867 | 完成 | 完整注入`37.45%`，关闭注入`36.05%`；差值`+1.40`点，配对95% CI `[-0.95,+3.70]`点；未检测到总体依赖 |
| H9 可观测 gate 可靠性 | 检验 student-only 视角一致性特征能否预测 episode 是否应注入 | H9 10K/30K，各128个 BS8 batch、194个 episode，两折 episode-held-out AUROC | 136904_0/1；136974/136904 | 完成；探索性 | 10K两折 AUROC：cosine `0.5600/0.4906`、分歧度 `0.5795/0.4962`、不平衡度 `0.5769/0.5098`；30K：`0.5268/0.5387`、`0.5128/0.5241`、`0.5357/0.5041`，均未达到预注册`0.60`，不支持当前动态 gate |
| H9 Agentview-only 注入 10K | 检验离线 probe 中 agentview-only 优于 wrist-only 的方向能否复现到闭环 | H9 10K `9999`；推理时保留 agentview、屏蔽 wrist；四套各500回合 | 136992；汇总136993 | 停止；无结论 | 训练使用双分支、推理屏蔽一支，存在分布不一致；不作为独立 contribution 因果证据 |
| ACPD-v2 注入离线动作影响 | 固定 H9 10K/30K，比较完整/关闭/分路注入的7维 flow MSE 与速度改变量 | 各128个BS8 batch；单卡冻结前向；按episode重采样 | smoke 136878；正式136879_[0-1%1] | 两点完成 | 完整注入−关闭MSE为`+0.00000834/+0.00000261`，两点区间均跨0；无离线改善证据，待闭环验证 |
| 注入分路差异 | 复用 H9 10K/30K 四分支逐样本误差，比较 agentview、wrist 与两路交互 | 各1,024样本、194 episode；CPU配对重采样，无新仿真 | 无新Slurm任务 | 完成；探索性 | 10K wrist−agent MSE `+0.00002726`，30K区间跨0；未检测到稳定分路差异或交互，不能证明闭环收益 |
| H9 teacher action signal 训练诊断 | 检查 teacher 动作目标是否持续优于 student 当前动作 | 复用 H9 训练日志的 0/5K/10K/15K/20K/25K/29.9K 记录，比较同一 batch 的 task loss | 无新Slurm任务 | 完成；训练诊断 | `teacher_better_ratio` 为 `0.9958--1.0000`，teacher task loss 始终低于 student；支持动作级 teacher 信号有效，但不是 benchmark 或因果结论 |
| Teacher-Action-MSE 5K（取消） | 直接拟合 teacher 动作的候选消融 | 4×5090、BS64、teacher-action MSE 权重0.1；smoke 136967 完成，正式训练及验证链已取消 | 136914、136916_[0-3]、136917、136918 | 取消；无科研结果 | GT flow/action 已是直接监督目标，未继续增加有误差的 teacher 动作 MSE；不能据此给出正负实验结论 |
| teacher信号与后期成功率关系 | 复用 H9 5K 窗口 task loss 与正式 benchmark 对照 | 无新Slurm任务 | 完成；探索性诊断 | student/teacher task-loss 比从 `10.89×` 降至 `5.49×`，但 H9 25K--30K 相对 SFT 优势消失；提示动作接口是主要待检验瓶颈 |
| ACPD-v2 动作方向诊断 | 冻结 H9/SFT 5K、10K，比较相对真实 flow target 的中心化 7D action cosine | 136895 smoke；136897 正式数组 | 完成；探索性混合结果 | 5K H9-SFT cosine `-0.0006888`，CI跨0；10K `+0.0007579`，CI跨0；仅10K `t=0.7` 局部为正，不能形成 pooled 一致方向改善结论 |
| ACPD-v2 同样本时间动作方向复核（取消） | 排除不同 flow 时间点使用不同样本的混杂，复核 5K/10K 的五个时间点 | 136987；每 checkpoint 128个 BS8 batch，同一样本遍历五个 t | 取消；未完成 | 无结果，不作正负结论 |
| 5K–30K 检查点序列面积 | 比较 H9 与匹配 SFT 六点平均成功率 | 5K/25K/30K为原H9训练，10K/15K/20K为同配置重建；40任务×50回合，10,000次配对回合重采样 | 无新Slurm任务 | 完成；探索性 | H9高`5.135`点，区间`[+4.080,+6.210]`点；30/40任务面积差为正；不是单次连续轨迹或GPU时间收益 |
| 中期逐任务差异 | 检查20/25/30K的H9−SFT差异是否稳定 | 只读三组全量验证日志，每任务50回合 | 无新Slurm任务 | 完成；探索性 | Object 20K 8/10任务为正，30K仅5/10；Goal三点均负的任务有4个；不能由套件均值推断普遍收益 |
| 早期逐任务收益覆盖 | 检查10K优势是否广泛、10K到15K任务收益是否稳定 | 只读5/10/15K全量验证日志；每任务50回合、2,000次配对重采样 | 无新Slurm任务 | 完成；探索性 | 10K有32/40任务为正，重采样区间[27,34]；15K其中仅20个保持，不支持直接使用固定任务路由 |
| 早期步数效率 | 比较H9比SFT少训练5K步时的成功率 | 只读5K对10K、10K对15K、15K对20K正式验证日志；各2,000回合配对 | 无新Slurm任务 | 完成；探索性 | 5K H9与10K SFT差值-0.10点，区间[-2.20,+2.05]点；另两组未达到±3点范围，不支持全程固定领先 |

## 探索性快速轨迹

每个 checkpoint 使用四套共 400 episodes，仅用于选择正式验证点，不作为论文最终数值。

| 模型 | Step | Spatial | Object | Goal | LIBERO-10 | Pooled | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| SFT backview BS64 | 10K | 21.00% | 30.00% | 28.00% | 3.00% | 20.50% | 完成 |
| SFT backview BS64 | 15K | 34.00% | 58.00% | 53.00% | 12.00% | 39.25% | 完成 |

H9-Fixed 的快速验证在运行前升级为正式2,000回合，因此没有400回合结果。

## 前期筛选结果

下表用于记录方法形成过程，不作为最终 BS64 公平对照。

| ID | 设置 | 关键结果 | 结论 |
|---|---|---|---|
| H1/H2 | ACPD LoRA FSDP4，BS32 | 单卡峰值约 `17.3 GiB` | 4×5090 训练可行 |
| H3 | layer 6、12、6+12，BS32，5K | loss 差异小于 1% | 不支持必须使用 6+12 层 |
| H4 | Flow、Cue、ACL、Full，BS32，2K | supervised loss 差异小于 1% | loss 不能筛选组件 |
| H5 | Full 对 Flow-only，BS32，5K | `6.50%` 对 `4.45%`，差值 `+2.05` 点 | Full 有正向信号 |
| H7/H7.1 | layer 6-12 contribution probe | layer 10 overall gap `0.3992` | 选择 layer 10 |
| H8 | ACL-only，BS32，5K | `6.35%`；Full 仅高 `0.15` 点 | H5 增益主要由 ACL 解释 |
| H9 | ACPD-v2 layer 10，BS32，5K | `11.35%`，相对 H8 `+5.00` 点 | ACPD-v2 有正向信号 |

## 验证结果

每个 benchmark 500 episodes；pooled 为四套共 2,000 episodes。

| 模型 | Spatial | Object | Goal | LIBERO-10 | Pooled |
|---|---:|---:|---:|---:|---:|
| SFT-2GPU BS16 30K | 9.40% | 20.60% | 20.80% | 2.80% | 13.40% |
| SFT-4GPU BS32 30K | 11.20% | 32.40% | 34.20% | 2.40% | 20.05% |
| SFT-cosine BS32 30K | 47.80% | 59.40% | 57.60% | 20.00% | 46.20% |
| SFT-cosine BS32 40K | 60.40% | 56.60% | 58.80% | 21.00% | 49.20% |
| SFT-cosine BS32 50K | 63.20% | 72.20% | 66.80% | 26.00% | 57.05% |
| SFT-cosine BS32 60K | 63.40% | 68.60% | 59.40% | 25.60% | 54.25% |
| Flow-only BS32 5K | 1.80% | 7.60% | 8.40% | 0.00% | 4.45% |
| ACL-only BS32 5K | 2.00% | 15.40% | 7.60% | 0.40% | 6.35% |
| Full ACPD BS32 5K | 2.80% | 10.80% | 12.20% | 0.20% | 6.50% |
| H9 前期 ACPD-v2 BS32 5K | 5.20% | 24.20% | 15.60% | 0.40% | 11.35% |
| SFT backview BS64 5K | 9.40% | 18.40% | 24.80% | 2.80% | 13.85% |
| H9-Fixed backview BS64 5K | 25.40% | 32.40% | 30.00% | 4.80% | 23.15% |
| H19-FeatureFusion backview BS64 5K | 20.00% | 33.00% | 29.20% | 2.60% | 21.20% |
| SFT backview BS64 10K | 23.40% | 37.80% | 29.20% | 2.60% | 23.25% |
| H9-Fixed backview BS64 10K | 41.60% | 50.60% | 47.60% | 10.00% | 37.45% |
| H19-FeatureFusion backview BS64 10K | 42.60% | 52.80% | 42.60% | 8.20% | 36.55% |
| SFT backview BS64 15K | 43.60% | 58.00% | 47.40% | 14.60% | 40.90% |
| H9-Fixed backview BS64 15K | 49.40% | 64.40% | 53.60% | 19.00% | 46.60% |
| H9-Fixed backview BS64 20K | 55.20% | 72.20% | 46.20% | 24.80% | 49.60% |
| H9-Fixed backview BS64 25K | 63.20% | 69.40% | 62.00% | 28.40% | 55.75% |
| H19-FeatureFusion backview BS64 25K | 65.60% | 67.00% | 63.60% | 25.20% | 55.35% |
| H9-Fixed backview BS64 30K | 64.00% | 73.00% | 61.80% | 33.20% | 58.00% |
| H9-Fixed backview BS64 35K | 73.20% | 73.80% | 63.60% | 33.60% | 61.05% |
| H9-Fixed backview BS64 40K | 68.20% | 76.00% | 60.00% | 33.60% | 59.45% |
| ACL-only backview BS64 25K | 63.40% | 62.40% | 61.80% | 24.00% | 52.90% |
| ACL-only backview BS64 30K | 61.20% | 71.00% | 60.00% | 25.20% | 54.35% |
| ACL-only backview BS64 35K | 67.00% | 75.20% | 63.60% | 31.60% | 59.35% |
| H13-LossOnly backview BS64 5K | 19.40% | 35.60% | 25.00% | 2.40% | 20.60% |
| ACPD-v2-TDCA backview BS64 5K | 18.20% | 35.00% | 28.40% | 3.80% | 21.35% |
| SFT backview BS64 20K | 50.60% | 62.60% | 54.40% | 21.20% | 47.20% |
| SFT backview BS64 25K | 64.00% | 64.00% | 65.60% | 29.60% | 55.80% |
| SFT backview BS64 30K | 70.00% | 69.80% | 68.20% | 33.80% | 60.45% |
| SFT backview BS64 35K | 65.80% | 68.40% | 64.60% | 29.80% | 57.15% |
| SFT backview BS64 40K | 70.00% | 73.20% | 69.40% | 29.60% | 60.55% |
| SFT backview BS64 50K | 70.20% | 74.60% | 70.80% | 29.60% | 61.30% |
| SFT backview BS64 60K | 73.20% | 78.00% | 67.60% | 29.40% | 62.05% |
| SFT topview BS64 30K | 81.00% | 80.60% | 79.20% | 45.40% | 71.55% |
| SFT leftview BS64 30K | 84.00% | 79.60% | 79.40% | 71.60% | 78.65% |
| SFT rightview BS64 30K | 85.00% | 84.80% | 84.00% | 54.20% | 77.00% |

## 精选 Checkpoint

精选目录使用文件级硬链接，不复制模型数据。原 checkpoint 本次不移动、不删除。

根目录：`/opt/liutong/openpi_checkpoints/fixed_dataset/curated`

| 分类 | 模型 | 已归档 | 状态 |
|---|---|---|---|
| teacher | agentview+wrist | `29999` | 完整 |
| baseline | backview BS64 | `4999` | 5K--60K checkpoint 完整；20K--60K 已测各点验证完成 |
| baseline | topview BS64 | - | 5K--30K checkpoint 和 30K 验证完整 |
| baseline | leftview BS64 | - | 5K--30K checkpoint 和 30K 验证完整 |
| baseline | rightview BS64 | - | 5K--30K checkpoint 和 30K 验证完整 |
| student | backview ACPD-v2 layer 10 BS64 | `24999` | 恢复轨迹至20K、延长轨迹至45K的 checkpoint 完整；40K验证完成，45K验证已取消 |
| ablation | backview loss-only BS64 | - | 5K checkpoint 和验证完整；之后补至 30K |
| ablation | backview ACL-only BS64 | - | 25K/30K/35K checkpoint 与全量验证完整 |

H11 不进入精选 checkpoint 目录。旧 BS16/BS32 checkpoint 暂不删除，也不进入正式索引。

## 证据路径

| 内容 | 路径 |
|---|---|
| 实验目录索引 | `experiments/README.md` |
| H7/H7.1 原始指标 | `/opt/liutong/openpi-5090-research/acpd-exact-attention-probe/results/` |
| H9 训练日志 | `slurm-log/pi05-bv-acpdv2-l10-pbs32-5k_129710.out` |
| H9-scale-b 训练日志 | `slurm-log/pi05-bv-acpdv2-l10-fsdp4-bs64-30k_129728.out` |
| H12 0--5K 训练日志 | `slurm-log/pi05-bv-sft-bs64-5k_130285.out` |
| H12 5K--30K 训练日志 | `slurm-log/pi05-bv-sft-bs64-r30k_130762.out` |
| H12 20K 验证 | `/opt/liutong/openpi-5090-evals/sft-backview-bs64-20k/20000/summary.txt` |
| H9早期轨迹快速验证 | `/opt/liutong/openpi-5090-evals/h9-early-trajectory-quick/` |
| H9/SFT 10K 正式验证 | `/opt/liutong/openpi-5090-evals/h9-early-trajectory-full/` |
| H9 15K 正式验证 | `/opt/liutong/openpi-5090-evals/h9-early-trajectory-full/14999/summary.txt` |
| H9/SFT 15K 配对分析 | `experiments/student/acpd-v2-h9-early-trajectory/results/h9_vs_sft_15k_paired_analysis.json` |
| H9/SFT 20K 配对分析 | `experiments/student/acpd-v2-h9-early-trajectory/results/h9_vs_sft_20k_paired_analysis.json` |
| H9 20K 验证 | `/opt/liutong/openpi-5090-evals/acpd-v2-training-trajectory/final-hidden/19999/summary.txt` |
| SFT 35K 验证 | `/opt/liutong/openpi-5090-evals/sft-backview-bs64-training-trajectory/35000/summary.txt`；job 134992/134995 |
| SFT 35K/30K 配对分析 | `experiments/baseline/sft-backview-bs64-trajectory-60k/results/sft_35k_vs_30k_paired_analysis.json` |
| SFT 40K/50K/60K 验证 | `/opt/liutong/openpi-5090-evals/sft-backview-bs64-training-trajectory/{40000,50000,59999}/summary.txt` |
| SFT 40K/50K/60K 对30K配对分析 | `experiments/baseline/sft-backview-bs64-trajectory-60k/results/sft_{40k,50k,60k}_vs_30k_paired_analysis.json` |
| H9 35K/SFT 60K配对分析 | `experiments/student/acpd-v2-h9-trajectory-60k/results/h9_35k_vs_sft_60k_paired_analysis.json` |
| H9/SFT 35K 配对分析 | `experiments/student/acpd-v2-h9-trajectory-60k/results/h9_35k_vs_sft_35k_paired_analysis.json` |
| H9 40K/35K 配对分析 | `experiments/student/acpd-v2-h9-trajectory-60k/results/h9_40k_vs_h9_35k_paired_analysis.json` |
| H9/SFT 10K 配对分析 | `experiments/student/acpd-v2-h9-early-trajectory/results/h9_vs_sft_10k_paired_analysis.json` |
| H9/H12 loss 对齐指标 | `experiments/student/acpd-v2-h9-batch-scaling-30k/results/loss_comparison.json` |
| H9/H12 loss 对齐图 | `artifacts/pi05_backview_sft_vs_acpdv2_loss.png` |
| H9/H12 30K 配对分析 | `experiments/baseline/sft-backview-bs64-5k/results/h9_vs_h12_30k_paired_analysis.json` |
| H9/H12 25K 配对分析 | `experiments/baseline/sft-backview-bs64-5k/results/h9_vs_h12_25k_paired_analysis.json` |
| H12 30K 验证 | `/opt/liutong/openpi-5090-evals/sft-backview-bs64-30k/29999/summary.txt` |
| H9-scale-b 30K 验证 | `/opt/liutong/openpi-5090-evals/acpd-v2-training-trajectory/final-hidden/29999/summary.txt` |
| H9-scale-b 25K 验证 | `/opt/liutong/openpi-5090-evals/slurm-log/sum-bv-h9-25k_132084.out` |
| H9 25K/30K 成对分析 | `experiments/student/acpd-v2-h9-20k-30k-trajectory/results/h9_25k_vs_30k_paired_analysis.json` |
| H9 25K 协议 | `experiments/student/acpd-v2-h9-20k-30k-trajectory/protocol.md` |
| H9 30K--60K 协议 | `experiments/student/acpd-v2-h9-trajectory-60k/protocol.md` |
| H9 35K 验证 | `/opt/liutong/openpi-5090-evals/acpd-v2-training-trajectory/final-hidden/34999/summary.txt` |
| H9 40K 验证 | `/opt/liutong/openpi-5090-evals/acpd-v2-training-trajectory/final-hidden/39999/summary.txt` |
| H9 45K checkpoint（验证已取消） | `/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/batch_scaling_30k/pi05_libero_backview_acpd_v2_layer10/pi05_libero_backview_acpd_v2_lora_fsdp4_bs64_30k/44999` |
| H9 35K/30K 配对分析 | `experiments/student/acpd-v2-h9-trajectory-60k/results/h9_35k_vs_h9_30k_paired_analysis.json` |
| H9 35K/SFT 30K 配对分析 | `experiments/student/acpd-v2-h9-trajectory-60k/results/h9_35k_vs_sft_30k_paired_analysis.json` |
| H15a 梯度诊断协议 | `experiments/mechanism/acpd-v2-h15a-gradient-conflict/protocol.md` |
| H18 ActionReadout 协议与分析 | `experiments/mechanism/acpd-v2-h18-action-readout/`；原始结果 `/opt/liutong/openpi-5090-research/acpd-v2-h18-action-readout/results/136130/summary.json` |
| H18 动作读出条件路由上限 | `experiments/mechanism/acpd-v2-h18-action-readout/oracle-routing-protocol.md`、`oracle-routing-analysis.md`；结果 `/opt/liutong/openpi-5090-research/acpd-v2-h18-action-readout/results/136130/oracle_routing.json` |
| H18 oracle episode 内交叉复核 | `experiments/mechanism/acpd-v2-h18-action-readout/oracle-holdout-protocol.md`、`oracle-holdout-analysis.md`；结果 `/opt/liutong/openpi-5090-research/acpd-v2-h18-action-readout/results/136130/oracle_holdout.json` |
| H18 可观测动作读出路由 | `experiments/mechanism/acpd-v2-h18-action-readout/observable-routing-protocol.md`、`observable-routing-analysis.md`；结果 `/opt/liutong/openpi-5090-research/acpd-v2-h18-action-readout/results/136935/observable_router.json` |
| H18 动作读出固定融合 | `experiments/mechanism/acpd-v2-h18-action-readout/ensemble-protocol.md`、`ensemble-analysis.md`；结果 `/opt/liutong/openpi-5090-research/acpd-v2-h18-action-readout/results/136935/ensemble.json` |
| H19-FeatureFusion 协议 | `experiments/student/acpd-v2-h19-feature-fusion-30k/protocol.md` |
| H19-FeatureFusion checkpoint | `/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/feature_fusion_30k/pi05_libero_backview_acpd_v2_feature_fusion/pi05_libero_backview_acpd_v2_feature_fusion_lora_fsdp4_bs64_30k/{4999,9999,14999,19999,24999,29999}/` |
| H19-FeatureFusion 训练日志 | `/opt/liutong/openpi-5090-research/acpd-v2-h19-feature-fusion/slurm-log/pi05-bv-acpdv2-feature-fusion-bs64-30k_136345.out` |
| H19-FeatureFusion 验证 | 有效5K/10K：`/opt/liutong/openpi-5090-evals/acpd-v2-h19-feature-fusion/{4999,9999}/`，任务 `138746/138748`、`138747/138749`；25K：`/opt/liutong/openpi-5090-evals/acpd-v2-h19-feature-fusion/24999-rerun-50-per-task/`（139046/139049）；30K：`/opt/liutong/openpi-5090-evals/acpd-v2-h19-feature-fusion/29999-rerun-50-per-task/`（139418/139419），pooled `60.10%` |
| H19-FeatureFusion 35K续训 | 从完整 H19 30K `29999` 恢复模型、优化器和 dataloader 状态，目标 `35000`；每1K保存，`keep_period=1`保留`30999/31999/32999/33999/34999`；不覆盖原30K目录 | 141714，4×5090，BS64，梯度累计1，当前 `PENDING (JobHeldUser)`，等待绑定reservation后释放；旧139791已取消 | 源 checkpoint：`/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/feature_fusion_30k/pi05_libero_backview_acpd_v2_feature_fusion/pi05_libero_backview_acpd_v2_feature_fusion_lora_fsdp4_bs64_30k/29999`；目标根目录：`/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/feature_fusion_35k`；尚无35K验证结果 |
| ACPD-v2 多视角 35K 协议 | `experiments/student/acpd-v2-multiview-35k/protocol.md` |
| ACPD-v2 多视角 35K 训练脚本 | `scripts/train_slurm/pi05_libero_{leftview,rightview,topview}_acpd_v2_layer10_lora_fsdp4_bs64_35k.sbatch` |
| ACPD-v2 多视角 35K checkpoint 根目录 | `/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/multiview_35k/{leftview,rightview,topview}/`（训练中，尚无最终 checkpoint 结论） |
| ACPD-v2 多视角 5K 全量验证 | `/opt/liutong/openpi-5090-evals/acpd-v2-multiview-35k/{leftview,rightview,topview}/4999/`；四套各500回合，`summary.txt`为最终汇总，`logs/`为逐套日志，逐套`videos/`为视频；Slurm日志见同根`slurm-log/` |
| H15a 快速诊断协议 | `experiments/mechanism/acpd-v2-h15a-gradient-conflict/fast_protocol.md` |
| H15a 快速诊断分析 | `experiments/mechanism/acpd-v2-h15a-gradient-conflict/analysis.md` |
| H15a 结果目录 | `/opt/liutong/openpi-5090-research/acpd-v2-gradient-conflict/` |
| ACPD-v2 loss问题定位 | `experiments/student/acpd-v2-h9-batch-scaling-30k/analysis.md` |
| H12 分析 | `experiments/baseline/sft-backview-bs64-5k/analysis.md` |
| H13 分析 | `experiments/ablation/acpd-v2-h13-injection-ablation/analysis.md` |
| H17 协议与任务记录 | `experiments/ablation/acpd-v2-h17-contribution-decay/` 内 `protocol.md`、`execution.md` |
| H17 checkpoint | `/opt/liutong/openpi_checkpoints/fixed_dataset/ablation/acpd_v2_h17_contribution_decay/pi05_libero_backview_acpd_v2_layer10/pi05_libero_backview_acpd_v2_h17_decay_bs64_20k` |
| H17 验证与配对分析 | `/opt/liutong/openpi-5090-evals/acpd-v2-h17-contribution-decay/19999/` |
| ACL-only BS64 协议 | `experiments/ablation/acpd-v2-acl-only-bs64-35k/protocol.md` |
| ACL-only BS64 checkpoint | `/opt/liutong/openpi_checkpoints/fixed_dataset/ablation/acpd_v2_acl_only_bs64_35k/pi05_libero_backview_acl_only_bs64_35k/pi05_libero_backview_acl_only_lora_fsdp4_bs64_35k/{24999,29999,34999}/` |
| ACL-only BS64 验证与分析 | `/opt/liutong/openpi-5090-evals/acpd-v2-acl-only-bs64-35k/{24999,29999,34999}/`；分析日志 `135729`、`135728` |
| ACPD-v2-TDCA 协议 | `experiments/ablation/acpd-v2-tdca-5k/protocol.md` |
| ACPD-v2-TDCA checkpoint | `/opt/liutong/openpi_checkpoints/fixed_dataset/ablation/acpd_v2_tdca_5k/pi05_libero_backview_acpd_v2_tdca/pi05_libero_backview_acpd_v2_tdca_lora_fsdp4_bs64_5k/4999` |
| ACPD-v2-TDCA 验证 | `/opt/liutong/openpi-5090-evals/acpd-v2-tdca-5k/4999/summary.txt` |
| ACPD-v2-TDCA 与 H9 5K 配对分析 | `experiments/ablation/acpd-v2-tdca-5k/results/tdca_vs_h9_5k_paired_analysis.json` |
| Flow-time 5K 诊断协议、结果与日志 | `experiments/mechanism/acpd-v2-flow-time-5k/`；结果 `results/136591/summary.json`、`results/136595/summary.json` |
| H9 10K 推理期注入消融 | `experiments/mechanism/acpd-v2-h9-inference-injection-10k/analysis.md`；配对结果 `results/paired_analysis.json`；验证 `/opt/liutong/openpi-5090-evals/acpd-v2-h9-inference-ablation/9999/` |
| ACPD-v2 注入离线动作影响 | `experiments/mechanism/acpd-v2-gate-effect/analysis.md`；原始结果 `/opt/liutong/openpi-5090-research/acpd-v2-gate-effect/results/{136880,136879}/` |
| ACPD-v2逐任务差异 | `experiments/mechanism/acpd-v2-task-heterogeneity/analysis.md`；原始三组日志路径见该文 |
| ACPD-v2早期逐任务收益 | `experiments/mechanism/acpd-v2-early-task-coverage/analysis.md`；逐任务原始计数见同目录 `results/task_coverage.json` |
| ACPD-v2早期步数效率 | `experiments/mechanism/acpd-v2-step-efficiency/analysis.md`；三组成对结果见同目录 `results/` |
| ACPD-v2训练轨迹面积 | `experiments/mechanism/acpd-v2-learning-curve-area/analysis.md`；原始计算结果见同目录 `results/learning_curve_area.json` |
| H14 多视角结果 | `experiments/baseline/sft-multiview-bs64-30k/analysis.md` |
| SFT 验证目录 | `/opt/liutong/openpi-5090-evals/` |
| ACPD 验证目录 | `/opt/liutong/openpi-5090-evals/acpd-*` |
