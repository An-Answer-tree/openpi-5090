# Figure 1：四视角同步高清图

- 数据：LIBERO-Spatial，第一个任务、第一个 episode（`demo_0`）。
- 任务：将盘子与小碗之间的黑碗放到盘子上。
- 时刻：从 0 编号的第 43 步，即第 44 帧；episode 共 98 帧（索引 0–97）。
- 文件：`backview.png`、`leftview.png`、`rightview.png`、`topview.png`。
- 每张图片为 1536×1536 PNG，来自记录的 MuJoCo XML、状态和原始相机参数重新渲染。
- 渲染时未推进物理状态，没有裁剪、放大原始低分辨率图像或修改场景内容。
- `manifest.json` 记录源文件、状态与图片摘要，以及 256×256 重渲染图与原始帧的核对误差。
- 图片用于展示同一场景的视角差异；Figure 1 的成功率来自四套任务的独立评测汇总，并非该 episode 的指标。

重新导出见上一级 `export_fig1_multiview.py`；重新绘制整图运行
`python paper/Figures/gen_fig_viewpoint_performance.py`。
