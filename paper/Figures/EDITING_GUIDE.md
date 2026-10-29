# ACPD 论文绘图源文件与后续编辑说明

本包保存截至 2026-10-29 的绘图源码、可编辑 PPT、数据、图片和导出文件。
解压后保留目录结构。下面的命令均从压缩包根目录运行。

## 文件与编辑方式

| 内容 | 主要文件 | 如何调整 |
|---|---|---|
| Figure 1：视角敏感性 | `paper/Figures/gen_fig_viewpoint_performance.py`、同目录 JSON | Python 调整布局、字体、颜色；JSON 保存成功次数和画布尺寸 |
| Figure 1：同步四视角高清素材 | `paper/Figures/Figure1_views/` | 四张 1536×1536 PNG，可直接替换或重新排版；manifest 保存来源 |
| Figure 3：同步多视角过程图 | `paper/Figures/Figure3_synchronized_multiview.pdf` | 保存当前工作区 PDF；另附六视角 × 八帧的原始图片，可重新排版 |
| Figure 4：ACPD 执行切片 | `paper/Figures/Figure4_ACPD_rollouts.pptx` | 五页可编辑 PPT；图片、文字、标签都是独立对象 |
| Figure 4：排版源码 | `paper/Figures/ACPD_rollouts/generate_rollouts.js` | 调整版式后重新生成 PPT；字体、颜色、间距等集中在脚本开头 |
| Figure 4：数据与素材 | `paper/Figures/ACPD_rollouts/` | 16 个视角—套件单元、96 张切片；manifest 保存帧号、任务和评测来源 |
| 相机位置示意图 | `paper/Figures/Camera_layout_material*.png` | PNG 素材及生成提示词；不是分层矢量图或可编辑 PPT |
| 相机几何依据 | `paper/Figures/Camera_layout_geometry.json`、`gen_camera_layout_guide.py` | 保存实际相机坐标、方向和几何参考图 |

当前保留的可编辑 PPT 是 Figure 4。Figure 1 可通过源码重绘；
Figure 3 未找到原始可编辑排版文件，本包保留现有 PDF 和过程图素材。
相机示意图 v2 是按实际几何参考生成的近似插画，精确相机方位以 JSON 为准。

## 最快的修改方式：直接编辑 PPT

用 PowerPoint 或 LibreOffice Impress 打开：

`paper/Figures/Figure4_ACPD_rollouts.pptx`

- 第 1 页：四视角 × 四套件的完整论文图。
- 第 2–5 页：Back、Left、Right、Top 各自的放大版本。
- 使用 Liberation Serif 字体；换电脑后安装同名字体可减少排版偏移。
- 可单独移动、缩放图片，修改任务名称、标签、颜色和间距。
- 手动编辑完成后，直接从 PPT 导出 PDF；运行生成脚本会覆盖手动修改。
- 已附单页 PDF、PNG 及五页完整 PDF，方便直接使用和核对。

## 通过源码重新生成 Figure 1

建议使用独立的 Python 3.11 环境：

```bash
python -m pip install matplotlib==3.10.3 numpy==1.26.4 pillow
python paper/Figures/gen_fig_viewpoint_performance.py
```

成功率数据在 `Figure1_viewpoint_performance.json` 中，保存的是成功次数，
脚本据此计算百分比。`width_cm` 和 `height_cm` 控制画布尺寸；
坐标轴位置、字体和颜色在 Python 脚本中调整。
重新排版只使用包内 JSON 和四张图片，不需要连接原服务器或加载模型。

## 通过源码重新生成 Figure 4

安装 Node.js 后运行：

```bash
cd paper/Figures/ACPD_rollouts
npm ci
node generate_rollouts.js ../Figure4_ACPD_rollouts.pptx
```

`package-lock.json` 固定 JavaScript 依赖版本，`helpers/` 已随包保留。
`npm ci` 首次需要联网下载依赖；无需原服务器的临时工作目录。

- 修改 `generate_rollouts.js`：画布比例、分区、图片大小、字体和间距。
- 修改 `display_titles.json`：主图中的简短任务名称。
- 修改 `manifest.json` 中各记录的 `frames`：图片相对路径和帧号。
- 原始 96 张 PNG 已保存，重排版不需要重新采样视频或运行评测。

自动导出 PDF 和 PNG 需要 LibreOffice、Poppler 和 Python 的 `pypdf`：

```bash
python -m pip install pypdf
python export_pdf.py ../Figure4_ACPD_rollouts.pptx --output-dir ..
```

`soffice` 和 `pdftoppm` 需能在终端直接运行。也可以在 PowerPoint 中直接导出 PDF。

## 以后想重新选帧

`materials/evaluation_videos/` 包含当前 16 个单元所选 episode 的原始评测视频。
`index.json` 将每个本地视频与原始来源、视角、套件和 SHA-256 对应起来。
这些视频可用于同一 episode 重新选时刻；不包含其他评测 episode。
原视频是 224×224，现有 PNG 保留原始解码像素。

当前排版每条 episode 使用六帧，从视频首帧到末帧均匀取样。
更换帧后，更新 `ACPD_rollouts/manifest.json` 的 `frames`，再运行 PPT 生成脚本。
`export_frames.py` 和 `sample_episodes.py` 保留的是原服务器取样流程，
包含原始数据索引和绝对路径；在其他电脑上重取帧时，需要改用本包的视频路径。
评测结果标签来自原始评测器，不应根据某张图片的姿态重新修改。

`materials/synchronized_episode/` 另存 Spatial 第一个任务、`demo_0` 的
六视角 × 八帧素材，以及时间点和来源记录。它们是示范数据，
与 Figure 4 的闭环评测视频用途不同。

`export_fig1_multiview.py` 和 `gen_camera_layout_guide.py` 记录原始场景恢复过程；
若要重新渲染，需要原 HDF5、LIBERO/robosuite 环境和仿真资源。
现有高清图、几何 JSON 和示意图已包含在本包中，重新排版无需这些外部数据。

## 备份与完整性

- `MANIFEST.json` 记录打包时间、Git 版本和每个文件的 SHA-256。
- 压缩包旁的 `.zip.sha256` 可用于校验整个压缩包。
- 包内未包含训练数据集、checkpoint、Node/Python 环境或大型缓存。
- 重新打包可在仓库根目录运行 `python paper/package_figure_sources.py`。

本次已校验 ZIP 完整性及全部文件摘要，并在独立解压目录中成功重绘 Figure 1、
重新生成 Figure 4 的五页 PPT。原 PPT 的五页结构、96 张切片和 16 段视频均已核对。
源码试运行使用本机现有 Python 和 Node 依赖；新电脑仍需按上述步骤安装环境。
