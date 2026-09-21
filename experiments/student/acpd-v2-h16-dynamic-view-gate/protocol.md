# H16：Token-conditioned dynamic view gate

## 问题

H9 分别预测 agentview 与 wrist contribution，却先将两者相加，再使用一个全局标量
注入所有样本和 action token。该结构不能根据遮挡和动作阶段选择更可靠的特权视角。
H16 检验：让每个 action token 独立控制两个视角，能否提高 5K 任务成功率。

## 唯一方法变化

保持 H9 的 layer-10 exact contribution、两个 contribution predictor、ACL、最终 hidden
注入和所有训练超参数不变。将全局注入系数扩展为：

`gate[b, h, v] = tanh(global_gate + Linear(stop_gradient(hidden[b, h])))`

其中 `v` 为 agentview 或 wrist。线性层权重和 bias 均初始化为 0，因此训练第 0 步两个
视角的 gate 都等于 H9 的全局 gate，初始行为完全一致。Predicted contribution 与 gate
输入对 flow 路径 stop-gradient；flow loss 只直接训练全局 gate 和新增的 token-view gate，
不改变 H9 的 predictor 梯度边界。

## 小实验

- 数据与 teacher：与 H9-scale-b 相同。
- Student：backview pi0.5 LoRA，layer 10。
- Loss 权重：flow `1.0`、contribution `0.2`、ACL `0.5`。
- 训练：4×RTX 5090，physical global BS64，无梯度累积，seed 42，cosine LR。
- 步数：5K；只在通过后续训到 30K。
- 验证：四个 LIBERO suites，每套 500 episodes，与 H9 使用相同协议。

提交正式训练前，使用 debug01 单卡 BS1、1 step smoke 验证图结构、梯度和显存。

## 判据

主要对照为 H9-scale-b 5K pooled `23.15%`。若 H16 高至少 `1.5` 个百分点且成对
bootstrap 95% CI 下界大于 0，则支持动态视角选择，并续训同一配置到 30K。否则停止
该分支，不用训练 loss 代替任务成功率结论。

同时报告 agentview/wrist gate 均值、两者绝对差和 gate gradient norm，用于确认新增
机制实际参与训练；这些健康指标不替代主要判据。
