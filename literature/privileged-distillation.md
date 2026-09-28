# Privileged information distillation

## 1. Lopez-Paz et al., 2015

**题目**：Unifying distillation and privileged information  
**来源**：<https://arxiv.org/abs/1511.03643>

论文把 privileged information 写成训练样本三元组 `(x, x*, y)`：`x*` 只在
训练时可见，student 在测试时只能使用 `x`。作者指出，一个直接方案是先从
`x` 重建 `x*`，再把预测的 `x*` 拼回 student 输入；但这可能比原任务更难。
论文提出的 generalized distillation 直接把 teacher 的输出作为额外监督，
而不是要求 student 完整重建 teacher 的特权描述。

**对本项目的可检验启发**：ACPD 的 attention contribution 是一种高维内部
特权描述。只证明它能从 backview 预测，不等于把它注入动作 hidden 就能提高
动作任务。应分别检验“预测目标可恢复”和“预测目标能改善任务输出”，不能把
前者当作后者的证据。

## 2. Xiao et al., 2024

**题目**：Provably Efficient Expert Policy Distillation  
**来源**：NeurIPS 2024，论文原文：
<https://proceedings.neurips.cc/paper_files/paper/2024/file/74d188c51d97fcfbc0269f584d6a53b7-Paper-Conference.pdf>

论文区分“特权策略训练后蒸馏”和“特权价值学习”。其命题指出，在部分可观测
环境中，直接模仿拥有完整状态的 expert policy 可能严格次优：不同真实状态
可能对应相同 student 观测，但 expert 动作不同，student 无法逐样本复制该动作。
论文给出 deterministic filter 等条件，在满足可辨识性时才有更强保证。

**对本项目的可检验启发**：teacher 的 agentview+wrist contribution 可能包含
backview 无法辨识的瞬时信息。若 contribution loss 很低但闭环收益不稳定，问题
可能不是 predictor 容量，而是目标中含有 student 观测无法确定的部分。可行的
后续方向是把 teacher 信号投影到“能被 student 观测和真实 action 共同解释”的
部分，再做低成本 5K 训练或冻结动作误差诊断。

## 当前设计含义

上述文献不证明 ACPD-v2 无效，也不提供本项目的成功率结论。它们只改变实验
优先级：先验证同一 H9 checkpoint 的推理注入消融和 ACL-only 对照，再决定是否
继续增加 contribution predictor 的复杂度。若要改目标，优先测试行为相关的
teacher residual / action-direction 信号，而不是继续扩大内部 contribution
重建网络。
