# ACPD-v2 改进路线

本文件是实验规划，不记录成功率。成功率只写入 `EXPERIMENT_RESULTS.md`。

## 当前问题

ACPD-v2-Fixed 在 backview、BS64 上的 pooled success 为：5K `23.15%`、10K
`37.45%`、25K `55.75%`、30K `58.00%`。匹配 SFT 为 5K `13.85%`、10K
`23.25%`、25K `55.80%`、30K `60.45%`。当前可靠结论是：ACPD-v2 加快早期学习，
尚未证明提高最终上限。

## 三个可检验假设

| 假设 | 依据 | 最小检验 |
|---|---|---|
| A. 后期固定 contribution 监督形成约束 | 30K 时加权 contribution 约占总 loss 60%；H15a 未发现明显负梯度冲突 | `ACPD-v2-Decay`，只降低 10K 后权重 |
| B. 注入分支没有被任务目标适配 | residual 注入对 predicted contribution 使用 `stop_gradient`；flow loss 主要更新 gate | 1 卡短跑比较 detached 与 task-gradient-connected residual |
| C. 部分 teacher contribution 对 backview 不可可靠恢复 | layer 10 只有平均可恢复性证据 | 1 卡短跑加入 contribution-energy/confidence 加权 |

## 执行顺序（低成本优先）

### 0. CPU 诊断

从已有 H9 日志统计 contribution target power、cosine、gate 和 flow loss 的分位数，
检查低能量样本是否主导 normalized contribution loss。不产生成功率结论，不改变活动任务。

### 1. 单卡机制 smoke

固定 seed、fixed_dataset、backview、layer 10、LoRA，global micro-BS16 或 32，
每个候选运行 300--500 optimizer steps。只保留两个候选：

1. `ACPD-v2-Decay` 短跑，核对权重调度和梯度稳定性；
2. `ACPD-v2-TaskAdapt` 短跑：teacher target 仍 stop-gradient，但取消 predicted
   residual 到 flow loss 的梯度截断，推理接口不变。

记录 flow/contribution/ACL loss、contribution cosine、gate、分支梯度范数和 NaN。
若 flow loss 比 Fixed 高超过 10%，或 contribution cosine 下降超过 0.05，停止该候选。
smoke 只筛选实现，不声明方法有效。

### 2. 单卡短训练筛选

通过 smoke 的候选最多两个，各运行 2K--5K steps；每个 checkpoint 做探索性评测：
四套 suite、每任务 10 回合，共 400 回合。该结果只排序，不作为最终论文数值。
保留早期表现不低于 Fixed、且相对匹配 SFT 仍为正的候选；小于 3 个百分点的差异不扩展。

### 3. 正式确认

只把一个候选扩展为 4 卡 BS64，训练至 30K/35K 并进行每点 2,000 回合全量验证。
主比较必须同时包含 `SFT-View(BV)` 和 `ACPD-v2-Fixed`，使用相同 checkpoint、seed、
相机位姿和 task-stratified paired bootstrap 95% CI。

## 决策树

1. `ACPD-v2-Decay` 提升 20K：先测试 decay-to-zero，不做大规模权重扫描。
2. Decay 无提升而 `ACPD-v2-TaskAdapt` smoke 稳定：正式验证 TaskAdapt。
3. 两者均无提升：进行 confidence-weighted contribution 单卡短跑；仍无提升则将结论
   收敛为 ACPD 主要提供 early-learning curriculum，而非最终上限提升。
4. ACL-only BS64 35K 和 Decay 结果出来前，不再提交新的 4 卡大规模 sweep。

## 资源边界

前两阶段每个候选最多 1 张 GPU、数小时到半天；400 回合筛选不是最终统计证据。只有
阶段 3 的 2,000 回合 paired evaluation 才能支持论文成功率结论。新实验必须先提交
独立 protocol commit，再提交训练任务；不移动或复制数据集、teacher 或 checkpoint。
