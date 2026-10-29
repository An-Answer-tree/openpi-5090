# Figure 2：ACPD 主方法图重设计

新增版本对应仓库根目录 `ACPD_method_overview.pptx` 的第三页。
第一、二页保留原样；第三页根据当前论文方法重新设计，
不再使用旧版可训练 Cross-Attention Cue Selector。

## 文件

- `../Figure2_ACPD_method_overview.pptx`：独立单页可编辑版本。
- `../Figure2_ACPD_method_overview.pdf`：宽 17 cm 的矢量 PDF，供论文插图使用。
- `../Figure2_ACPD_method_overview.png`：新版预览。
- `generate_method_overview.js`：PptxGenJS 源码；模块、文字、箭头及特征示意均可编辑。
- `append_slide.py`：将独立单页附加到原 PPT，直接保留既有页面及媒体的原始 XML。
- `assets/`：同步的 Agent、Wrist、Back 相机图片和来源记录。
- `helpers/`：字体测量、图片放置和布局检查所需的本地辅助代码，附原始许可证。

## 图中内容

上方为冻结的多视角教师，下方为可训练的单视角学生。
教师在动作专家第 10 层，使用原有注意力及对应视角的 value 提取 cue；
学生从同层动作状态通过 LN 和仿射预测器预测两路 cue。
PCD 对齐每个视角的教师目标和学生预测。

两路预测相加，经过 stop-gradient 和可学习标量 `tanh(gamma)`，
再加入学生最终动作表征。原动作输出层生成 flow velocity。
ACL 比较教师和学生预测速度的 Pearson correlation，
总目标另保留 ground-truth flow supervision。
部署时移除教师，学生中的预测器和 gate 均保留。

注意力小矩阵仅为结构示意，不是实验测得的注意力热图。
相机图片来自同一 Spatial episode 的同一状态；图片细节见 `assets/provenance.json`。

## 重新编辑或生成

直接在原 PPT 第三页修改即可。用源码生成时：

```bash
cd paper/Figures/ACPD_method_overview
npm ci
node generate_method_overview.js ../Figure2_ACPD_method_overview.pptx
```

依赖版本见 `package-lock.json`。字体使用 Liberation Sans 和 Liberation Serif。
独立单页与原 PPT 均采用相同的 16:9 画布。
可以将重新生成的单页复制到原 PPT 并替换第三页。

首次附加到**原始两页版本**时，可运行：

```bash
python append_slide.py original_two_slides.pptx ../Figure2_ACPD_method_overview.pptx updated_three_slides.pptx
```

此工具只执行附加，不执行页面替换。已有第三页时请不要直接重复附加。
源码重新生成不会保留在 PowerPoint 中手动进行的调整，修改前请另存版本。

PDF 可在 PowerPoint 或 LibreOffice 中导出。
本次不修改论文中的主图占位框，待图稿审阅后再接入正文。

## 交付核验

最终三页 PPT 已完整渲染。前两页的 XML、媒体和渲染像素均与原文件一致。
新增页含 166 个原生形状或文本对象和 3 张真实相机图片，
通过布局重叠检查及 slides skill 的画布越界测试。
PDF 字体全部嵌入；具体记录见 `validation.json`。
