# H18：动作读出条件路由上限

## 目的

判断 hidden-contribution 是否只在部分样本或 episode 上有用。若不同动作读出头
在不同 episode 上互补，则固定使用一个读出头可能掩盖 contribution 的收益，动态路由
值得继续研究；若理想路由也没有明显收益，则不再优先增加路由复杂度。

## 数据与方法

- 复用 H18 已完成的冻结验证结果 `validation_errors.npz`。
- 四列误差分别为冻结 H9、hidden-only、hidden-layer10、hidden-contribution。
- 以 episode 平均 flow MSE 作为主要指标。
- 比较三个固定读出头、逐样本理想选择和逐 episode 理想选择。
- 逐 episode 理想选择使用该 episode 的真实 flow target，仅作为不可部署的上限。
- 对 episode 做 2,000 次 bootstrap，报告理想选择相对最佳固定读出头的差值和95%区间。

## 判定

若 episode-level oracle 相对最佳固定读出头的 MSE 至少下降 1%，且差值区间上界小于 0，
则记录为“存在可观测的条件互补”，后续才考虑使用 student 可见特征学习路由；否则不
增加路由网络。

该分析不产生仿真成功率结论，也不把使用真实 target 的 oracle 当作可部署方法。
