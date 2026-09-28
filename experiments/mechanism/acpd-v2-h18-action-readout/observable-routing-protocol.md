# H18：可观测动作读出路由器

## 目的

验证 H18 的理想路由上限能否由 Student 自己可见的 hidden 和 predicted contribution
特征近似。理想 oracle 使用真实 flow target，只能证明存在上限；本实验训练一个小型
误差预测器，在推理选择动作读出头时不访问真实 target。

## 固定设置

- 输入：H18 验证集保存的 Student 特征；不输入真实 flow target。
- 候选动作输出：冻结 H9、hidden-only、hidden-layer10、hidden-contribution 四路。
- 特征：每个候选输入的 action-horizon 平均值，再按每 32 个 hidden 维度平均，得到
  `3×96=288` 维可见特征。
- 路由器：带偏置的 ridge 线性回归，预测四路 flow MSE；正则系数 `1e-2`。
- 泛化划分：按 episode id 奇偶性做两折交叉评估；每折只用训练 episode 的 flow target
  拟合路由器，测试 episode 只使用特征选择候选头。
- 主要对照：固定 hidden-layer10，使用 H18 的预先确定结果，不用测试集选择对照。
- 统计：按测试 episode 配对 bootstrap 2,000 次。

## 判定

若可观测路由相对固定 hidden-layer10 的 episode mean flow MSE 至少下降 1%，且95%
区间上界小于 0，则进入 5K 小规模闭环验证；否则不提交正式路由训练。

该实验仍是冻结模型的动作误差诊断，不是 benchmark 成功率验证。
