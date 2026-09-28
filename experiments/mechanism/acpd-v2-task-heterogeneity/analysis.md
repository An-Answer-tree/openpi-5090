# ACPD-v2 逐任务差异：探索性结果

使用匹配的 H9-Fixed 与 backview SFT 20K、25K、30K 全量验证日志。
每套10任务、每任务50个相同初始状态，逐任务差值为 `H9成功率-SFT成功率`。
读取前逐项核对任务和episode键一致；未重新训练或仿真。

| 套件 | 20K 正/负/零任务数 | 25K 正/负/零 | 30K 正/负/零 | 三点均正/均负 | 20K与30K任务差值Spearman |
|---|---:|---:|---:|---:|---:|
| Spatial | 6/3/1 | 5/5/0 | 3/7/0 | 1/1 | -0.881 |
| Object | 8/1/1 | 7/0/3 | 5/4/1 | 2/0 | +0.480 |
| Goal | 5/5/0 | 4/5/1 | 3/7/0 | 2/4 | +0.526 |
| LIBERO-10 | 6/3/1 | 6/4/0 | 4/5/1 | 2/2 | -0.198 |

| 任务例子 | H9−SFT 20K | 25K | 30K |
|---|---:|---:|---:|
| Object：把 salad dressing 放入篮子 | +26点 | +12点 | +24点 |
| Goal：把 cream cheese 放入碗中 | -26点 | -4点 | -28点 |
| Spatial：从盘子与小烤碗之间拿起黑碗 | +18点 | +10点 | -30点 |

20K 的 Object 套件差值为 `+9.60` 点，不由单个任务独占：8/10任务为正。
但到30K仅5/10任务为正。Spatial 的逐任务差值次序明显改变；
例如同一任务由20K `+18` 点变为30K `-30` 点。因此不能依据20K的任务优劣
制定跨训练阶段固定不变的模型路由。Goal 中有4个任务在三点都低于SFT。

这些是同一训练seed、同一40个任务和同一批初始状态的探索性观察，
不是独立复现；单任务仅50回合，未做40个任务的多重比较校正。
该分析支持继续检验任务条件下的方法差异，不证明差异由 contribution 注入导致。
推理期注入因果检验见 `../acpd-v2-h9-inference-injection-10k/`。

原始日志：

| Step | SFT | H9-Fixed |
|---:|---|---|
| 20K | `/opt/liutong/openpi-5090-evals/sft-backview-bs64-20k/20000/logs/` | `/opt/liutong/openpi-5090-evals/acpd-v2-training-trajectory/final-hidden/19999/logs/` |
| 25K | `/opt/liutong/openpi-5090-evals/sft-backview-bs64-25k/25000/logs/` | `/opt/liutong/openpi-5090-evals/acpd-v2-training-trajectory/final-hidden/24999/logs/` |
| 30K | `/opt/liutong/openpi-5090-evals/sft-backview-bs64-30k/29999/logs/` | `/opt/liutong/openpi-5090-evals/acpd-v2-training-trajectory/final-hidden/29999/logs/` |
