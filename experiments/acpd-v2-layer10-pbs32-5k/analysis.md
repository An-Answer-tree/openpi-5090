# H9 分析

## 结果

| 方法 | Spatial | Object | Goal | LIBERO-10 | Pooled |
|---|---:|---:|---:|---:|---:|
| H8 ACL-only | 2.00% | 15.40% | 7.60% | 0.40% | 6.35% |
| H9 ACPD-v2 | 5.20% | 24.20% | 15.60% | 0.40% | 11.35% |

H9 相对 H8 的 pooled 差值为 `+5.00` 个百分点。对相同 2,000 个 episode
进行 10,000 次 task-stratified paired bootstrap，95% CI 为
`[+3.60, +6.45]` 个百分点。

## 结论

H9 同时超过预注册的 `+1.5` 点门槛，且置信区间下界大于 0，因此支持
layer-10 exact contribution 的 5K 部署效果。该结论来自单 seed、单 student
视角，仍需重复 seed 确认。

原始验证结果位于
`/opt/liutong/openpi-5090-evals/acpd-v2-task-success-5k/layer10-exact-contribution-pbs32/4999`，
配对分析位于 `results/h9_vs_h8_paired_analysis.json`。

## 训练过程检查

| 指标 | step 0 | step 4,900 | 判断 |
|---|---:|---:|---|
| supervised loss | 0.0892 | 0.0311 | 稳定下降，约下降 65% |
| student task loss | 0.3959 | 0.1407 | 稳定下降，约下降 64% |
| exact contribution loss | 1.0003 | 0.4063 | 目标可被持续拟合 |
| exact contribution cosine | 0.0057 | 0.6980 | 预测方向从无相关变为明显相关 |
| teacher-better ratio | 1.0000 | 0.9987 | teacher 在训练样本上持续优于 student |
| exact contribution gate | 0.0000 | 0.0048 | 融合门保持很小，注入路径较保守 |
| 单卡显存峰值 | - | 17,406 MiB | 4 卡 FSDP、BS32 运行稳定 |

训练共记录 5,000 个 optimizer steps，checkpoint 4,999 的 `params` 和 `assets`
均成功保存；日志未出现 NaN、OOM 或 traceback。`learning_rate=0.0000` 是统一以四位
小数格式输出的结果，不能解释为实际学习率为零。

## 设计判断

H9 同时支持 ACPD-v2 的三个必要环节：真实 teacher Q/K/V attention contribution
可以作为固定监督目标，backview student 能从自身观测恢复该目标，且把两个视角的预测
贡献融合到最终 hidden 后能提高任务成功率。相对同一批配对 episode 的 H8 ACL-only，
pooled 提升 5.00 个百分点，95% CI 为 [+3.60, +6.45]；Spatial、Object、Goal
分别提升 3.20、8.80、8.00 个百分点，LIBERO-10 没有提升。

因此，当前判断是“设计合理并有正向证据”，但不是“机制已完全证明”。H9 中 gate 最终
仅为 0.0048，且 predictor 的注入在反向传播中被 stop-gradient；任务增益可能同时来自
贡献预测辅助损失对 LoRA 表征的约束，而不一定全部来自推理时残差的直接幅度。还需要
H11 的同层注入对照、同 BS64 的 SFT 对照以及重复 seed，才能分离这两种解释并检验长期
任务效果。`weighted_acpd_loss / supervised_loss` 在末步约为 2.68，说明权重 0.2
在当前损失尺度下并不等价于“弱约束”，后续权重校准应在注入结构固定后进行。
