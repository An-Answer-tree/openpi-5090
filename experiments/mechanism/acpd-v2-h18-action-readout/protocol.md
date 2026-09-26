# H18：Contribution 动作读出诊断

预注册日期：2026-09-27。性质：探索性接口筛选，不是任务成功率实验。

## 问题与做法

H9 后期能预测 teacher contribution，但尚未证明能提高成功率上限。本实验检查：
在不改变已学 contribution 的情况下，小型动作修正网络能否利用该表征。

起点固定为 H9-Fixed BS64 30K：
`/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/batch_scaling_30k/pi05_libero_backview_acpd_v2_layer10/pi05_libero_backview_acpd_v2_lora_fsdp4_bs64_30k/29999/params`。

冻结全部 H9 参数，包括 LoRA、layer-10 predictor、原 gate、action projection。
原始注入保持开启。每个 action token 的修正输出加到原 velocity 的真实7维动作上。
只训练三组独立小网络；它们共用同一批样本、noise、time和冻结主干前向。

| 组别 | 修正网络输入 | 作用 |
|---|---|---|
| hidden-only | 最终注入前 hidden；两个额外输入块置零 | 检查新增动作网络本身的收益 |
| hidden-layer10 | 最终 hidden；layer-10 hidden 复制到两个输入块 | 控制跨层表征连接的收益 |
| hidden-contribution | 最终 hidden；预测 agentview、wrist contribution | 检查贡献表征的读出收益 |

三个输入块分别做参数无关的零均值、单位方差归一化，再拼接。
网络为 `3×1024 → 128 → SiLU → 7`，共享初始随机权重，各自独立优化；输出层
全零初始化，训练前严格等于冻结 H9。三组名义参数量相同，但 hidden-only 的零输入
列没有有效梯度，不能声称有效容量完全匹配。layer10 对照用于补足这一限制。

当前 predictor 是 layer-10 hidden 的线性函数，预测贡献没有新增观测信息。
该比较只能检验表征参数化/动作接口，不能证明贡献包含 layer10 之外的新信息。
不加载 teacher，不复制 checkpoint 或数据集，不保存冻结主干副本。

## 固定预算与指标

| 项目 | 固定设置 |
|---|---|
| 数据 | fixed_dataset，student 仅 backview；原归一化统计不变 |
| 资源 | 单张5090，8CPU，24G主存，正常 batch 队列，时限12小时 |
| 更新 | 500 optimizer steps，physical BS8，累积1，seed42 |
| 优化器 | Adam，固定 LR=1e-3；仅小网络，不沿用 LoRA 的2.5e-5 |
| 目标 | 真实7维动作 flow velocity MSE；没有新的 contribution/ACL loss |
| 噪声 | 与H9相同：time~Beta(1.5,1)×0.999+0.001，noise~N(0,1) |
| 划分 | 按episode固定90%/10%，seed42；训练和诊断episode不交叉 |
| 诊断 | 128个BS8 batch，固定noise/time；只在500步末读取，不挑选最佳step |
| 主指标 | held-out episode均值的7维flow MSE；同时报告frame均值和冻结H9 |
| 区间 | episode-cluster bootstrap 2,000次；贡献组减对照组的配对95%区间 |
| 输出 | 三组小网络参数、配置/episode列表、训练记录、逐样本误差和JSON汇总 |

episode只对本次小网络训练留出；冻结H9此前已在这些episode训练过，不是未见数据
泛化测试。诊断使用专家轨迹和单次noisy-action前向，不代替闭环仿真。
只对修正网络参数求梯度，特征显式stop-gradient，正式主干不更新。

## 预先固定的后续判据

贡献组相对两个对照的episode平均MSE均至少下降1%，且两个配对区间上界均小于0，
才进入仿真验证阶段。该门槛只是筛选规则，不是论文显著性结论，也不保证任务成功率。
不达标则不扩展成长训练：若三组均改善而贡献组不胜，支持一般动作接口而非贡献特异
收益；若layer10相当或更好，优先考虑直接跨层读出；若三组均无收益，此接口未获支持。

TDCA保留。TDCA从base训练至5K，开放原预测支路的任务梯度；H18从H9 30K冻结
诊断后期读出接口。两者起点、训练范围不同，不能直接比较数值或视为彼此替代。
