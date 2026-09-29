# H18：动作读出固定融合

## 目的

验证不同动作读出头的误差互补性是否可以通过固定平均融合获得，而不需要预测
每个 episode 的最佳分支。该方案只使用已有 H18 readout 参数，不新增训练网络。

## 设置

- 数据：H18 可见特征提取结果，1,024 个样本、194 个 episode。
- 候选输出：hidden-only、hidden-layer10、hidden-contribution 三个已训练动作修正头。
- 主要方案：`0.5 * hidden-only + 0.5 * hidden-contribution` 的动作输出平均值。
- 对照：固定 hidden-layer10 输出；对照在 H18 中预先确定。
- 次要探索：在 episode 奇偶两折中，从 `alpha ∈ {0, 0.25, 0.5, 0.75, 1}` 选择
  `alpha * hidden-layer10 + (1-alpha) * hidden-contribution`，只在训练 episode 选 alpha，
  在另一半 episode 测试。
- 统计：按 episode 配对 bootstrap 2,000 次。

## 判定

若主要方案相对固定 hidden-layer10 的 flow MSE 至少下降 1%，且95%区间上界小于0，
则进入冻结模型闭环验证；否则不提交正式融合训练。

该分析不产生 benchmark 成功率结论。
