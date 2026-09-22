# H9 早期训练轨迹快速验证

## 问题

ACPD-v2 相对匹配 SFT 的早期优势在 5K 到 20K 之间何时开始缩小？

## 设置

比较 H9 ACPD-v2 与 backview SFT 的 10K、15K checkpoint。两种方法均使用
physical BS64、seed 42 和相同的 30K cosine schedule；H9 checkpoint 来自
job 132398 的 5K 精确续训轨迹。

每个 checkpoint 使用固定 seed 7，每套 LIBERO benchmark 的每个任务运行 10 次，
四套共 400 episodes。ACPD-v2 与 SFT 使用相同初始状态和 episode 顺序。该协议是
探索性筛查，不替代每任务 50 次、共 2,000 episodes 的正式验证。

## 判断

分别计算 10K 和 15K 的 `ACPD-v2 - SFT` pooled success：

- 差值大于 3 个百分点：该点仍有明确正向信号。
- 差值在正负 3 个百分点内：该点接近优势消失区间，补做正式 2,000 episodes。
- 差值小于负 3 个百分点：早期优势已在该点前消失。

训练 loss 只用于健康检查，不作为任务成功率结论。若快速验证差异较小，只将其用于
选择正式验证点，不报告为论文最终数值。
