# ACPD 四视角 × 四套件执行切片

主图为四分区布局：左上 Back、右上 Left、左下 Right、右下 Top。
每个分区依次展示 Spatial、Object、Goal、Long，每个单元一条独立随机抽取的
评测 episode，包含从首帧到末帧的六张原始画面，共 16 个单元、96 张图片。

PPT 共五页：第一页为完整论文图，后四页为四个视角各自的放大版本。
所有图片、文字和标签均为独立的 PowerPoint 对象，可单独调整。
主图采用紧凑排版，共用进度标签，减少重复文字和图间留白。

## 取样与来源

- 仅使用主实验最终归档模型对应的真实闭环评测视频。
- 固定随机种子 `20261029`；每个视角和套件内，从 10 个任务 × 50 个评测回合中
  均匀随机抽取一条。没有按成功、失败或画面效果筛选，也没有重抽。
- 六帧取自 `round(linspace(0, N - 1, 6))`，包括记录的首帧和末帧。
- Start、20%、40%、60%、80%、End 表示各自视频的相对进度，不表示跨回合的相同时刻。
- 图中的结果标签与 16 段原始评测日志逐一核对一致。本次抽样为 12 段成功、4 段失败。
- 原始视频为 `224 × 224`。PNG 保留解码后的原始像素，不裁剪、不增强、不生成图像细节。
  PDF 的文字为矢量；提高导出分辨率不会增加原始视频的细节。
- 评测器在执行每个动作前保存画面。因此 End 是最后一个记录帧；结果标签取自评测器，
  不以图片中某一瞬间的姿态重新判断成功。

`selection.json` 保存随机选择；`manifest.json` 保存完整指令、任务和回合编号、
视频来源、SHA-256、评测日志、模型归档、帧编号和原始分辨率。
源模型对应关系来自仓库的 `experiments/curated-checkpoints.yaml`。
图内短任务名称见 `display_titles.json`；完整指令也保存在 PPT 备注中。

## 文件

- `../Figure4_ACPD_rollouts.pptx`：可编辑 PPT，含完整主图及四张放大页。
- `../Figure4_ACPD_rollouts.pdf`：单页论文主图，宽 17 cm。
- `../Figure4_ACPD_rollouts.png`：主图预览。
- `../ACPD_rollouts_all.pdf`：五页完整导出。
- `frames/`：96 张原始视频切片。
- `generate_rollouts.js`：PptxGenJS 排版源码。
- `helpers/`：随 slides skill 提供的排版辅助代码，保留 Apache-2.0 许可证。

## 重新生成

在本目录运行：

```bash
npm ci
node generate_rollouts.js ../Figure4_ACPD_rollouts.pptx
python export_pdf.py ../Figure4_ACPD_rollouts.pptx --output-dir ..
```

PDF 导出需要 `pypdf`、LibreOffice 的 `soffice` 与 Poppler 的 `pdftoppm`。
当前文件已完成导出，可直接使用，不需要先运行源码。

如需重取原始帧，保留现有 `selection.json` 并运行 `export_frames.py`；需要
OpenCV、NumPy、PyYAML，以及原始评测视频和日志的访问权限。
`sample_episodes.py` 记录随机取样逻辑，需要 LIBERO 环境；现有选择已保存，
重新排版无需重新抽样。

## 验证

已核对 16 个视角—套件组合、96 张切片、首末帧覆盖和递增帧序，
以及视频与原始日志的成功/失败一致性。
排版源码包含重叠和越界检查；交付前逐页渲染审阅并检查内容溢出。
没有重新训练或运行新评测。主图已接入论文 Overall Performance 小节，
正文引用与图注仅说明图中内容；最终图号由 LaTeX 自动生成。
