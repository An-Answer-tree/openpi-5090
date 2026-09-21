# H9 30K 推理注入诊断

## 问题

H9 30K 相对匹配 SFT 的优势可能缩小。该实验判断原因是推理时 residual 注入，
还是训练期 contribution loss 与 ACL 对共享 LoRA 的影响。

## 严格对照

复用 H9-scale-b step 29,999 checkpoint，不进行训练。验证时仅将
`exact_contribution_injection` 从开启改为关闭；模型权重、backview 相机、policy
输入、seed 7、任务顺序和初始状态全部不变。四套 LIBERO benchmark 各运行
500 episodes，共 2,000 episodes。

## 判据

对开启与关闭注入的逐 episode 结果进行 task-stratified paired bootstrap：

- 关闭注入显著更高：推理注入在 30K 已产生负作用。
- 开启注入显著更高：注入仍有效，若 H9 不优于 SFT，原因在训练期辅助目标。
- 差异不显著：H9 与 SFT 的差异主要归因于训练期辅助目标，而非推理开关。

该诊断不使用训练 loss 代替任务成功率结论。

