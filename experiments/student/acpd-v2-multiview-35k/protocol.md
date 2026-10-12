# ACPD-v2 多视角 Student 35K 训练协议

## 目的

在 H9-Fixed 的相同蒸馏设置下，分别训练 leftview、rightview 和 topview
student，获得与 backview H9 可直接比较的多视角结果。

## 固定设置

| 项目 | 设置 |
|---|---|
| Student | pi0.5 LoRA，单一弱视角（left/right/top） |
| Teacher | agentview + wrist，固定 `/opt/liutong/openpi_checkpoints/fixed_dataset/sft/agentview_wrist/29999/params` |
| Privileged target | layer 10 exact attention contribution |
| Loss | flow 1.0，contribution 0.2，ACL 0.5 |
| 注入 | 保留 H9 exact contribution fusion 与 residual injection |
| 数据 | `libero_multiview_tuned_6view_lerobot`，fixed_dataset 资产 |
| 训练 | 4×5090 FSDP，physical global BS64，gradient accumulation 1，seed 42 |
| 学习率 | warmup 1K，peak `2.5e-5`，cosine decay 30K 到 `2.5e-6` |
| 总步数 | 35K；5K 间隔保存，`keep_period=1` 保留全部检查点 |

三个实验只改变 Student 视角、对应 config 和输出目录；teacher、随机种子、损失、
优化器、数据集与硬件配置保持一致。`decay-steps=30000` 延续 H9，因此 30K--35K
阶段使用末端学习率平台，不把多视角结果与 H9 的学习率边界混为新的方法消融。

## 预注册比较

每个视角完成 5K--35K 检查点后，使用相同的四套 LIBERO benchmark 和相同 episode
协议验证 30K、35K；5K、10K、15K、20K、25K 检查点保留，是否评测由后续资源安排
决定。主要比较是各视角 ACPD-v2 与相同视角 BS64 SFT baseline，不能使用 backview
baseline 替代对应视角对照。

## 5K 全量验证（2026-10-06）

三个视角的 `4999` checkpoint 已完整保存。本次各评测 Spatial、Object、Goal、
LIBERO-10，每套 10 个任务、每任务 50 回合，共 2,000 回合；seed 7、replan 5、
fixed_dataset 相机位姿与既有验证一致。推理使用对应视角的
`pi05_libero_{view}_acpd_v2_layer10_lora`，保留两路贡献预测和注入。

每个模型提交一个四子任务的单卡数组及一个 `afterok` 汇总任务，每模型最多一套
同时运行，三个模型共用最多三张卡。单卡请求 8 CPU、24G 内存、24 小时；
允许 `batch,debug01` 分区，排除 `gpu03`，由 Slurm
使用未分配显卡。训练任务保持原配置继续运行，不移动或复制 checkpoint。

结果、日志和视频按视角归档到
`/opt/liutong/openpi-5090-evals/acpd-v2-multiview-35k/{view}/4999/`。
5K 结果只描述早期表现；方法收益须与相同视角、相同训练进度的 SFT 比较，
不能用 30K SFT 对照推断早期收益。最终 30K、35K 比较保持不变。

| Student | 验证数组 | 自动汇总 | 推理配置 |
|---|---:|---:|---|
| leftview | 139971 | 139972 | `pi05_libero_leftview_acpd_v2_layer10_lora` |
| rightview | 139973 | 139974 | `pi05_libero_rightview_acpd_v2_layer10_lora` |
| topview | 139975 | 139976 | `pi05_libero_topview_acpd_v2_layer10_lora` |

复用 `run_fixed_backview_benchmark_array.sbatch`，依次传入完整 checkpoint 路径、
对应验证目录、表中推理配置、`{view}_image`、`50`；提交时指定
`--partition=batch,debug01 --array=0-3%1 --time=1-00:00:00`。
汇总复用 `summarize_fixed_backview_benchmark.sbatch`，依赖对应数组全部成功。

## 资源调整（2026-10-08）

用户取消 topview 蒸馏训练 `139445`，优先安排 H19 30K→35K 续训。
leftview/rightview 训练 `139443/139444` 继续运行，已有结果与 checkpoint 不删除。
为避免与 H19 竞争资源，按用户要求取消尚未启动的三项 SFT 5K 验证
`140296/140297/140298` 及汇总 `140299/140300/140301`；SFT 5K 尚无结果。
已完成的 ACPD-v2 三视角 5K 验证不受影响。

## 通过条件

训练必须完成首个检查点且无 NaN/Inf；最终结论只使用闭环成功率。若任务失败，记录
工程状态，不把训练 loss 当作方法结论。

## 35K 全量验证（2026-10-12）

leftview/rightview 的30K与35K checkpoint已完整保存，训练日志确认35K最终
写入完成。按用户更正，本次提交两视角的35K（目录`34999`）验证。
此前提交的leftview 30K数组`147056`在启动前取消，rightview 30K尚未提交。

| 项目 | 设置 |
|---|---|
| 模型 | ACPD-v2-Fixed，layer 10，保留两路贡献预测与注入 |
| Checkpoint | 原训练目录`multiview_35k/{view}/pi05_libero_{view}_acpd_v2_layer10/pi05_libero_{view}_acpd_v2_layer10_lora_fsdp4_bs64_35k/34999` |
| 推理配置与视角 | `pi05_libero_{view}_acpd_v2_layer10_lora`、`{view}_image` |
| 验证协议 | fixed_dataset相机位姿；四套各500回合，共2000回合；seed 7，replan 5，图像224，等待10步；评测协议与既有SFT一致，但当前相同视角SFT只有30K结果，不能称为同训练进度对照 |
| GPU任务 | 每模型单卡数组`0-3`，不限制并发；每项8 CPU、24G、24小时，batch分区、排除gpu03 |
| 汇总 | 每模型一个1 CPU、2G、15分钟任务，依赖四个子任务全部成功 |
| 环境 | `openpi`策略服务、`openpi-libero`仿真；`CONDA_ENVS_PATH=/home/liutong/miniconda3/envs`；联网关闭、代理清除 |
| 输出 | `/opt/liutong/openpi-5090-evals/acpd-v2-multiview-35k/{view}/34999/`；套件日志在`logs/`，视频在各套`videos/`，调度日志在`slurm-log/`，最终汇总为`summary.txt` |

复用已有验证数组和汇总脚本，不复制或移动checkpoint。任务号提交后记录于
`research-state.yaml`与结果台账；验证完成前，两视角35K均为“尚无结论”。

| Student | 35K验证数组 | 自动汇总 | 2026-10-12 17:00 CST状态 |
|---|---:|---:|---|
| leftview | `147057_[0-3]` | 147058 | 四套验证完成，summary.txt已生成 |
| rightview | `147059_[0-3]` | 147060 | 四套验证完成，summary.txt已生成 |

2026-10-12 12:59 CST原地将两个待运行数组及汇总任务绑定到`res_gpu04`
（gpu04，有效期10月12日12:48至10月16日12:48）。未重提任务，任务号、
评测配置、并发设置、输出路径和依赖关系不变；等待预约节点的已有任务释放资源。

## Topview 15K→35K续训（2026-10-12）

按用户要求接续已取消的topview训练。原目录已保留完整的4999/9999/14999；
本次恢复最新14999的模型、优化器与全局步数，使用同seed、BS64、累计1，
从batch15000接续数据顺序。训练目标仍为35K，teacher、loss和学习率沿用原配置。
30K以后LR保持2.5e-6，不重启warmup。本次仅改变保存频率，尚无35K任务结论。

| 项目 | 设置 |
|---|---|
| 资源 | batch，4×5090 FSDP，32 CPU、96G、8天；排除gpu03，正常排队，不限定节点 |
| 保存 | 30K及之前每5K；30K之后每1K；keep_period=1全部保留 |
| 预期新增目录 | 19999/24999/29999/30999/31999/32999/33999/34999 |
| 数据接续 | --resume-data-loader；恢复step15000，累计1，对应batch15000 |
| Checkpoint目录 | `/opt/liutong/openpi_checkpoints/fixed_dataset/distillation/acpd_v2/multiview_35k/topview/pi05_libero_topview_acpd_v2_layer10/pi05_libero_topview_acpd_v2_layer10_lora_fsdp4_bs64_35k` |
| 脚本 | `scripts/train_slurm/pi05_libero_topview_acpd_v2_layer10_lora_fsdp4_bs64_35k.sbatch` |

续训协议先提交到git，再提交Slurm任务。沿用原checkpoint目录，不复制或移动模型。

2026-10-12 17:27:38 CST提交任务`147339`，协议commit `fa4f232`先于提交。
保存边界、恢复状态与数据接续8项测试通过；实际sbatch命令CLI解析、bash语法、
Slurm test-only和已提交脚本核对通过。17:28任务为Priority排队，未分配GPU；
实际恢复step15000和batch15000须待启动日志确认。
