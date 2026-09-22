# H9 20K--30K 训练轨迹验证

## 问题

H9 ACPD-v2 是否在 20K 或 25K 已达到任务成功率峰值，并在 30K 因过拟合或
持续辅助约束而下降？

## 设置

验证 H9 的 step 19,999 和 24,999 checkpoint。两者均使用 layer 10、FSDP4、
physical BS64、seed 42、固定权重 `1.0/0.2/0.5` 和同一 30K cosine schedule。
25K checkpoint 来自原始 H9-scale-b 训练；20K checkpoint 来自相同配置的确定性
恢复训练 job 130704，生成后再启动验证。

同时验证匹配的 backview SFT BS64 step 20,000 checkpoint。SFT 与 H9 20K 使用
相同的 backview 相机、评测 seed 和 episodes，作为判断 ACPD-v2 同训练进度增益的
公平对照。

每个 checkpoint 使用固定 backview 相机、seed 7、每任务 50 次，四套 LIBERO
benchmark 各 500 episodes，共 2,000 episodes。不进行第二 seed 复测。

## 判断

主要比较同一原始训练轨迹的 25K 与 30K。若 25K pooled success 高于 30K，且
逐 task 配对 bootstrap 的 95% CI 下界大于 `0`，则支持 25K 后性能下降。20K
用于补充轨迹；由于它来自确定性恢复训练，解释时单独注明来源。H9 20K 与 SFT
20K 的差值用于判断 ACPD-v2 的早期优势在 20K 是否仍然存在。

训练 loss 只用于健康检查，不作为过拟合结论。
