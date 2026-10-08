# MV-SV-KD：pi0.5 离线固定视角适配

类型：预注册比较实验。目的：为 ACPD 提供多视角教师到单视角学生的第三方方法对照。

## 文献与适配边界

论文：*Visual-Policy Learning through Multi-Camera View to Single-Camera View
Knowledge Distillation for Robot Manipulation Tasks*（2023）。
原文：https://arxiv.org/html/2303.07026v2 ，方法 IV-B，式 (5)、(8)。

| 内容 | 原论文 | 本次适配 |
|---|---|---|
| 策略 | SAC | pi0.5 flow matching |
| 数据 | 教师采集、学生在线采集并聚合 | 既有 fixed_dataset，离线配对样本 |
| 视角 | 多相机教师、单相机学生；相机随机化 | agentview+wrist 教师、固定 backview 学生 |
| 输出蒸馏 | 策略动作 MSE | 相同 noisy action、相同时间的 7D flow velocity MSE |
| 特征蒸馏 | 视觉编码特征的欧氏距离 | 最终动作输出层前的 1024D hidden，归一化 MSE |
| 损失权重 | 不能直接移用到 pi0.5 | GT/输出/特征：1.0/0.5/0.1，适配超参数 |

本实验保留“输出与特征联合蒸馏”的思想，不是原论文完整复现。
特别是最终 action hidden 替代视觉编码特征，是额外的方法适配；论文报告必须注明。
不启用 ACPD contribution、ACL、cross-attention 选择器、gate 或新增动作融合网络。

## 训练协议

| 项目 | 固定设置 |
|---|---|
| Teacher | 冻结全量 SFT pi0.5，agentview+wrist，30K |
| Teacher params | `/opt/liutong/openpi_checkpoints/fixed_dataset/sft/agentview_wrist/29999/params` |
| Student | `pi05_libero_backview_lora`；部署仅 backview |
| 初始化 | `/home/liutong/.cache/openpi-5090/pi05_base/params` |
| 数据集 | `/opt/liutong/libero_multiview_tuned_6view_lerobot` |
| 归一化 | `/home/liutong/Projects/openpi/assets/pi05_libero_agentview_wrist`，`libero_multiview` |
| 资源 | 普通 batch 分区，4×5090，32 CPU，96G RAM，8天冗余；不绑定节点，排除 gpu03 |
| Batch / Seed | physical global BS64，梯度累计1，seed42 |
| LoRA / Horizon | 沿用已有 backview LoRA；action horizon=10，输出32D |
| 优化器 | 沿用 backview LoRA 配置，关闭 EMA |
| LR | warmup1000，peak=2.5e-5，cosine decay30000，末端2.5e-6 |
| 步数 / 保存 | 30,000 optimizer steps；每5K保存，`keep_period=1`全部保留 |
| 保留目录编号 | `4999/9999/14999/19999/24999/29999` |
| 环境 | conda `openpi`，完全离线，不使用代理 |
| Checkpoint 根目录 | `/opt/liutong/openpi_checkpoints/fixed_dataset/baseline/mv_sv_kd/backview_bs64_30k` |
| 日志根目录 | `/opt/liutong/openpi-5090-research/mv-sv-kd-backview-bs64-30k` |

两模型接收同一状态、语言、真实动作、噪声与时间；只改变可见图像。
教师不接受梯度。GT flow loss 保留全部32维，输出蒸馏只使用 LIBERO 真实7维。

`L = L_GT + 0.5 L_velocity + 0.1 L_feature`。

- `L_GT = mean((v_student - (noise - action_GT))^2)`。
- `L_velocity = mean((v_student[..., :7] - stop_gradient(v_teacher[..., :7]))^2)`。
- 对每个动作 token，将 hidden 除以最后一维的 RMS，随后计算逐元素 MSE；
  等价于对单位范数向量的平方欧氏距离取 token 均值。非零向量时为 `2(1-cosine)`。
  教师目标使用 stop_gradient；不拟合特征幅值。

输出、特征权重是固定的适配选择，不宣称为原论文默认或已优化参数。

## 检查与评测

先验证损失数值、教师无梯度、学生梯度、无 ACPD 参数、旧训练分支不变，以及
普通 backview LoRA 推理参数兼容。单卡 debug 使用真实数据与权重完成两步训练，
不保存 smoke checkpoint。正式任务依赖 smoke 成功，避免未验证的代码直接长训。

主比较：30K，与相同视角/BS64/初始化/LR 的 SFT、H9-Fixed、H19 比较。
后续全量评测四个 LIBERO suite，每任务50回合，合计2,000回合，seed7，
replan_steps=5，使用 fixed_dataset 相机位姿。5K--25K保留供后续探索性轨迹评测。
本轮只提交训练；不创建未经用户要求的自动验证链。

记录各项原始与加权 loss、特征 cosine、梯度范数与学习率。
训练 loss 只作运行诊断，尚无成功率结论；最终判断依据正式仿真结果。
