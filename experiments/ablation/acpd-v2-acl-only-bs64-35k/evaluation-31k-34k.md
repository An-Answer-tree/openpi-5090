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
