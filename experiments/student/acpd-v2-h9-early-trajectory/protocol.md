# H9 早期训练轨迹验证

## 问题

ACPD-v2 相对匹配 SFT 的早期优势在 5K 到 20K 之间何时开始缩小？

## 设置

比较 H9 ACPD-v2 与 backview SFT 的 10K、15K checkpoint。两种方法均使用
physical BS64、seed 42 和相同的 30K cosine schedule；H9 checkpoint 来自
job 132398 的 5K 精确续训轨迹。

SFT 10K、15K 已使用固定 seed 7，每套 LIBERO benchmark 的每个任务运行 10 次，
四套共 400 episodes，作为探索性筛查结果。

H9 10K、15K 改为正式验证：固定 seed 7，每套 benchmark 的每个任务运行 50 次，
每个 checkpoint 四套共 2,000 episodes。四套分别报告成功率，并计算 pooled success。
H9 结果可与使用相同正式协议的 H9 5K、25K、30K 比较。由于 SFT 10K、15K 仍为
400 episodes，不将跨方法差值作为正式论文结论。

## 判断

H9 10K、15K 首先用于定位自身训练轨迹。与探索性 SFT 结果比较时仍沿用以下筛查规则：

- 差值大于 3 个百分点：该点仍有明确正向信号。
- 差值在正负 3 个百分点内：该点接近优势消失区间，补做正式 2,000 episodes。
- 差值小于负 3 个百分点：早期优势已在该点前消失。

训练 loss 只用于健康检查，不作为任务成功率结论。只有两种方法都使用 2,000 episodes
时，才报告正式的跨方法差值和配对置信区间。
