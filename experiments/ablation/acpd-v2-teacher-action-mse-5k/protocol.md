# ACPD-v2 Teacher-Action MSE：直接动作蒸馏

预注册日期：2026-09-29。性质：低预算训练消融；目标是检验 teacher 的动作输出
是否比仅使用 ACL 更适合作为 student 的动作学习信号。

## 动机

H9 训练日志显示 teacher 的 7 维动作 task loss 在所有记录点都低于 student，
`teacher_better_ratio` 为 `99.58%--100%`。当前 H9 只使用 teacher/student 的
中心化方向相似度（ACL），没有直接最小化动作值差异。本实验增加一个可选的 teacher
action MSE，使 student 直接靠近同一 noisy action 和 flow time 下的 teacher 输出。

## 单变量设置

| 项目 | H9 对照 | 本实验 |
|---|---:|---:|
| student | backview | backview |
| teacher | agentview+wrist | agentview+wrist |
| flow loss | 1.0 | 1.0 |
| contribution loss | 0.2 | 0.2 |
| ACL | 0.5 | 0.5 |
| teacher-action MSE | 0 | 0.1 |
| 训练预算 | 5K steps | 5K steps |
| 资源 | 4×5090，FSDP，global BS64，累积1 | 完全相同 |
| seed、LR、数据 | H9 完全相同 | H9 完全相同 |

新增项只比较有效 7 维动作，teacher 输出 stop-gradient；Libero 的其余输出维度不
参与该项。teacher 前向已经由 ACPD 训练使用，不新增 teacher 计算路径。

## 判定

使用与 H9 5K 相同的四套 2,000 回合配对验证。若 pooled 成功率相对 H9 提高至少
1.5 个百分点且配对区间下界高于 0，继续验证 30K；否则不扩展训练。训练 loss 和
teacher/student MSE 只作机制指标，不替代仿真成功率。

## 风险与解释边界

teacher 动作本身使用 privileged images，直接拟合可能提高早期动作精度，也可能让
student 过度追随 teacher 的不可见细节。即使 5K 成功率提高，也只能证明动作级蒸馏
在该预算下有用，不能单独证明提升最终上限；30K 仍需独立验证。
