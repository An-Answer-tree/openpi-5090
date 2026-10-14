# 2026-10-14 缓存清理与验证修复

本文件记录工程处理，不作为方法实验结论。模型、相机、seed7及每套500回合协议不变。

## 缓存清理

| 项目 | 实际检查与处理 |
|---|---|
| 来源 | Hugging Face datasets将fixed_dataset本地Parquet转换为Arrow；337819样本、2000个源文件 |
| 生成时间 | 2026-10-13 23:38至2026-10-14 01:50 |
| 已删除 | 下方明确路径内253个Arrow文件及dataset_info.json；约122 GiB；无子目录，所有文件硬链接数均为1 |
| 活动任务核对 | debug01、gpu02、gpu04的用户进程没有映射此路径；142275实际映射`/opt/liutong/huggingface/datasets/.../*.arrow` |
| 空间变化 | `/home`文件系统使用率90%→88%；可用空间755217989632→886259580928字节；共享盘数值受其他用户活动影响 |
| 后续默认位置 | conda `openpi`、`openpi-libero`环境设置HF_HOME=`/opt/liutong/huggingface`、HF_DATASETS_CACHE=`/opt/liutong/huggingface/datasets`；只影响之后激活的环境 |
| 恢复方式 | 缓存删除不可直接撤销，可由原始Parquet重新生成；没有移动、复制或删除数据集、checkpoint |

已删除缓存路径：
`/home/liutong/.cache/huggingface/datasets/parquet/default-b858ebb0a6d98a77/0.0.0/9c460aabd2aa27d1496e5e38d2060760561f0ac2cd6a110134eefa5b3f153b8d`。

只检查上述缓存目录的直接文件及明确配置路径，没有递归扫描数据、checkpoint、日志或视频目录。

## 启动失败与修复

| 项目 | 事实 |
|---|---|
| 失败项 | 148526_1：31K Object；148528_1：32K Object；148530_0：33K Spatial |
| 异常 | torch.distributed导入时调用tempfile.TemporaryDirectory；TMPDIR父目录不存在，报FileNotFoundError |
| 原临时路径 | `/dev/shm/openpi-slurm-tmp/{job_id}/{suite}`；gpu04上整个`openpi-slurm-tmp`已不存在，运行中的其他任务也使用这个旧前缀 |
| 原因边界 | 确认是临时目录消失；没有OOM或磁盘满的错误证据。何进程删除目录尚未确认，不归因于缓存占用或系统清理机制 |
| 修改 | `run_fixed_backview_suite.bash`用mktemp在该suite的`/opt`评测目录创建独立TMPDIR；EXIT先停止policy server，再删除自身临时目录，不清理共享父目录 |
| 检查 | bash语法与git diff检查通过；CPU模拟完整shell流程，PyTorch同一导入链成功，成功与模拟启动失败两种退出均清理自身临时目录 |
| 不变 | 模型、checkpoint位置、离线设置、相机、回合、随机种子不变；142275及两个运行中的LIBERO-10验证不停止、不修改 |
| 重跑范围 | 只补三项失败suite；原小日志保留后再重跑；汇总依赖改为所需的成功补跑和仍在运行的原suite |

失败证据位于评测根目录
`/opt/liutong/openpi-5090-evals/acpd-v2-multiview-35k/topview/`：
`30999/logs/object.log`、`31999/logs/object.log`、`32999/logs/spatial.log`。

## 重跑任务

补跑任务号及实际启动检查将在提交后补充。已有完整suite日志不重跑。
