# H18：动作读出 oracle 的 episode 内交叉复核

## 目的

验证 H18 的 episode-level oracle 上限是否能在未参与选择的样本上复现。该实验仍然
使用真实 flow target 选择读出头，只用于判断互补性是否稳定，不作为可部署方法。

## 方法

- 输入：H18 已保存的 1,024 个样本、194 个 episode 的四路 flow MSE。
- 对每个 episode 按样本顺序交替分成两半。
- 用第一半选择四路中 episode 平均误差最低的读出头，在第二半测试；再交换两半。
- 固定 hidden-layer10 作为预先确定的对照。
- 只保留每半至少一个样本的 episode；按 episode 对两个方向取平均。
- 使用 2,000 次 episode bootstrap，报告 holdout oracle 与固定 layer10 的差值。

## 判定

若 holdout oracle 相对固定 layer10 的 MSE 至少下降 1%，且95%区间上界小于0，
则确认状态条件互补值得进入可观测路由或端到端融合；否则将原 oracle 视为主要是
同样本选择上限，不再以它支持动态路由。
