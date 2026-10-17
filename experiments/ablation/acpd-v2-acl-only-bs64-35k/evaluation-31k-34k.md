# ACL-only backview BS64 31K–34K全量验证

## 目的与固定协议

补齐ACL-only在30K与35K之间的成功率轨迹，供同进度SFT/H9比较。
本次为探索性checkpoint比较；不预设最优步数，不将训练loss作为成功率证据。
模型来自[30K→34K补训](trajectory-31k-34k-protocol.md)，不是新方法；
原25K/30K/35K验证结果保持不变。

| 项目 | 固定设置 |
|---|---|
| Checkpoint | 30999/31999/32999/33999，分别含31K/32K/33K/34K优化步 |
| 推理配置 | `pi05_libero_backview_acpd_v2_layer10_loss_only_lora`；关闭注入，与ACL-only训练一致；推理不加载teacher |
| 相机 | `backview_image`；fixed_dataset tuned相机位姿，沿用已有验证器 |
| 回合 | Spatial/Object/Goal/LIBERO-10各10任务×50回合；每模型2000回合，共8000回合 |
| 推理 | seed7、224×224、replan5、等待10步；保存视频 |
| GPU任务 | 每模型数组0-3，不设并发上限；每子任务1GPU/8CPU/24G/8小时；batch自动选节点，排除gpu03 |
| 汇总 | 每模型1CPU/2G/15分钟，afterok依赖整个验证数组，无GPU |
| 启动条件 | 31K–33K已完整保存，直接提交；34K afterok依赖训练149325，启动时检查checkpoint完成标记 |
| 环境 | conda openpi推理、openpi-libero仿真；Hugging Face离线，清除代理 |

复用已有`examples/libero/eval_slurm/run_fixed_backview_benchmark_array.sbatch`，
按顺序传入`CHECKPOINT_DIR EVAL_DIR POLICY_CONFIG backview_image 50`。
汇总复用`summarize_fixed_backview_benchmark.sbatch`，传入同一个`EVAL_DIR`。
数组0/1/2/3依次对应Spatial/Object/Goal/LIBERO-10。
Slurm输出覆盖为各模型目录的`slurm-log/%x_%A_%a.out`（数组）和`%x_%j.out`（汇总）。
不指定节点或预约，避免绑定即将到期的预约；Slurm仍可能自动使用磁性预约。

## 路径

```text
checkpoint:
/opt/liutong/openpi_checkpoints/fixed_dataset/ablation/acpd_v2_acl_only_bs64_trajectory_31k_34k/pi05_libero_backview_acl_only_bs64_35k/pi05_libero_backview_acl_only_lora_fsdp4_bs64_resume_34k/{30999,31999,32999,33999}

evaluation:
/opt/liutong/openpi-5090-evals/acpd-v2-acl-only-bs64-trajectory-31k-34k/{step}/

已有参考:
/opt/liutong/openpi-5090-evals/acpd-v2-acl-only-bs64-35k/{24999,29999,34999}/summary.txt
```

每模型`logs/`保存四套原始日志，`{suite}/videos/`保存视频，`slurm-log/`保存
调度日志，`summary.txt`保存最终汇总。不覆盖旧结果，不移动、复制或删除checkpoint。
不调整已有训练、验证、优先级或预约。

## 检查与判定

提交前核对已产出checkpoint的完成标记、params/train_state元数据和norm stats，
确认推理配置关闭注入，运行bash语法与Slurm test-only检查。不另申请GPU smoke任务。
四套各完成500回合且原始计数与汇总一致后，才报告最终pooled成功率。
记录每套成功/总数；不合并不同checkpoint回合。
最高观测点属于探索性选择；未计算配对区间时不作显著性判断。
协议先commit，实际任务号与执行状态随后记录；目前尚无新增成功率结论。

## 实际提交

协议commit：`0e307c1`。2026-10-16 10:17 CST提交，10:18核对：

| Checkpoint | 四套单卡数组 | 自动汇总 | 状态 |
|---|---|---|---|
| 31K / 30999 | 149556_[0-3] | 149557 | 等待Priority；checkpoint完整 |
| 32K / 31999 | 149558_[0-3] | 149559 | 等待Priority；checkpoint完整 |
| 33K / 32999 | 149560_[0-3] | 149561 | 等待Priority；checkpoint完整 |
| 34K / 33999 | 149562_[0-3] | 149563 | afterok等待149325；checkpoint尚未产出 |

四个汇总分别afterok依赖对应整个验证数组。已核对每项1GPU/8CPU/24G/8小时、
未绑定节点、排除gpu03、没有数组并发上限，四组日志路径与模型一一对应。
31K–33K完成标记和三个JSON元数据均有效；CPU导入确认推理模型layer10、
action_horizon10、action_dim32、injection=False、feature_fusion=False。
bash语法、git diff与五项Slurm test-only通过。复用现有验证流程，未另提交GPU测试。

此时训练149325仍在gpu04运行，进度约33.9K；最新完整loss记录为33800，数值有限。
验证尚未获得GPU，不能视为已经正常开始仿真，尚无新增成功率结果。
没有修改其他活动任务或操作checkpoint文件。

## 已完成结果

2026-10-17 11:09 CST核对原始四套日志与汇总一致，每项2000回合。

| Checkpoint | Spatial | Object | Goal | LIBERO-10 | Pooled |
|---|---|---|---|---|---|
| 31K / 30999 | 345/500=69.00% | 364/500=72.80% | 328/500=65.60% | 158/500=31.60% | 1195/2000=59.75% |
| 32K / 31999 | 342/500=68.40% | 374/500=74.80% | 317/500=63.40% | 142/500=28.40% | 1175/2000=58.75% |

汇总路径为上述evaluation根目录下`30999/summary.txt`和`31999/summary.txt`，
分别于10月17日03:59、07:48生成。未计算配对区间，不作显著性判断。
该补训与原35K来自同一30K状态的不同续训轨迹，不宣称逐位复现原35K。

## 尚未完整的验证

| Checkpoint | 已完成套件 | 剩余状态 |
|---|---|---|
| 33K / 32999 | Spatial323/500、Object372/500、Goal319/500 | LIBERO-10尚未完成；汇总149561依赖未满足 |
| 34K / 33999 | Spatial340/500、Object356/500、Goal346/500 | 149562_3在gpu04验证LIBERO-10；汇总149563等待依赖 |

两者尚无完整pooled结论，不将部分回合与完整模型比较。

## 33K补跑协议

按用户要求，仅补跑33K的LIBERO-10（数组索引3），保持原checkpoint、配置、
seed7及500回合；复用其余三套已完成结果，不重复验证。单卡8CPU/24G/8小时，
batch自动选节点、排除gpu03，可自动使用磁性预约。新提交的数组脚本改用
20000–29999端口，避开系统临时连接端口范围；原失败日志另名保留。
汇总149561改为依赖该单卡任务成功，再读取同目录四套结果。其他活动任务不变。
