# Figure 2：ACPD 主方法图重设计

新增版本对应仓库根目录 `ACPD_method_overview.pptx` 的第三页。
第一、二页保留原样；第三页根据当前论文方法重新设计，
不再使用旧版可训练 Cross-Attention Cue Selector。
当前采用紧凑布局：删除顶部标题和共享输入横条、底部训练与部署说明，
教师和学生位于同一连续框体内，PCD 和 ACL 嵌入右侧支路，不再占用整行间隔。

## 文件

- `../Figure2_ACPD_method_overview.pptx`：宽幅紧凑单页可编辑版本。
- `../Figure2_ACPD_method_overview.pdf`：宽 17 cm、高约 5.87 cm 的矢量 PDF。
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

依赖版本见 `package-lock.json`。所有文字和数学符号统一使用 Arial，
字号较紧凑初稿整体增加 2 pt。输入图像放大约 28–31%，模块框同步放大，
边框和连接线适当加粗，减少模块间留白。
文字框禁用自动换行，并给 Layer 10 等标签预留宽度。
独立单页采用 `16.6667 × 5.75` 英寸的宽幅画布，方便直接作为论文主图导出。
原三页 PPT 保持 16:9 画布以保留前两页，新版图居中放在第三页。

首次附加到**原始两页版本**时，可运行：

```bash
node generate_method_overview.js --full-canvas full_canvas.pptx
python append_slide.py original_two_slides.pptx full_canvas.pptx updated_three_slides.pptx
```

此工具只执行附加，不执行页面替换。已有第三页时请不要直接重复附加。
源码重新生成不会保留在 PowerPoint 中手动进行的调整，修改前请另存版本。

PDF 可在 PowerPoint 或 LibreOffice 中导出。
本次不修改论文中的主图占位框，待图稿审阅后再接入正文。

## 交付核验

最终三页 PPT 已完整渲染。前两页的 XML、媒体和渲染像素均与原文件一致。
新增页的模块、箭头和文字均为原生对象，包含 3 张真实相机图片，
通过布局重叠检查及 slides skill 的画布越界测试。
PDF 字体全部嵌入；具体记录见 `validation.json`。
