# P001：精简introduction和related work操作

## 定义

| 项目 | 内容 |
|---|---|
| 用户指定名称 | 精简introduction和related work操作 |
| 建立日期 | 2026-10-28 |
| 状态 | 两轮精简已应用；仅建立检查点，尚未回退 |
| 目标文件 | `paper/ELSP_Latex_Template.tex` |
| 精简前版本 | `7ef9284b390c4c7085133c01000cfd17e9e8bd6c` |
| 第一轮精简 | `6771a1ee82c6303db57f3acc180104b6b482d8a0` |
| 第二轮精简／当前版本 | `2c62cd758d43c3be8852649a3146d49115150c00` |
| 回退目标 | 撤销上述两轮精简，恢复精简前被删减或缩写的文字，同时保留其他内容和后续独立修改 |

## 精确范围

从 `\section{Introduction}` 开始，到 `\section{Action-Conditioned Privileged Distillation}` 之前结束，即 Introduction 和完整 Related Work。后面的 Method 及其他章节不在回退范围内。

保留精简前已经采纳的修改，包括动作与 action chunk 术语、特权蒸馏定义、Acar 方法对比、视角差异说明及 motivation 图的第二页放置安排。Abstract、导言区、图框与图位置设置、Method、实验、表格、参考文献文件均不属于本次精简。

## 文件

| 文件 | 用途 |
|---|---|
| [before.tex](before.tex) | 精简前两个章节的逐字原文 |
| [after.tex](after.tex) | 完成两轮精简后两个章节的逐字原文 |
| [undo.patch](undo.patch) | 由精简后到精简前的撤销补丁，方向已经是回退方向 |

精简内容：删去重复背景和重复方法总结，压缩视角限制、相关工作和方法动机表述。Introduction 约 687 → 574 词；Related Work 约 421 → 302 词。词数仅作长度参考，恢复以原文快照和补丁为准。

已核对精简前、第一轮后和第二轮后的全文引用：27 处引用命令的内容及顺序完全一致，没有删除、更换或重排引用。

## 回退步骤

以下命令仅供用户将来要求回退时执行；本次创建检查点不执行实际回退。

1. 读取当前正文和本记录，保留当前正文副本用于前后校验。
2. 在仓库根目录检查补丁能否直接应用：

   ```bash
   git apply --check paper/checkpoints/P001_intro_related_work_condensation/undo.patch
   ```

3. 若检查通过，再应用补丁：

   ```bash
   git apply paper/checkpoints/P001_intro_related_work_condensation/undo.patch
   ```

4. 若检查失败，按两个章节分别比较 `before.tex`、`after.tex` 与当前正文，逐处撤销精简；保留后续独立修改。不得强行覆盖整章、整篇文件或直接 `git revert` 原提交。
5. 检查 diff，确认只撤销所记录的精简，并确认当前正文在上述章节范围之外逐字节不变。若尚无后续章节内修改，恢复后的两个章节应与 `before.tex` 完全一致。
6. 记录回退日期及提交版本，提交并推送本次回退涉及的文件。

后续编辑若改动了同一句，须合并文字才能兼顾两次操作；完整快照用于确定原始内容，不应直接覆盖后续修改。

## 完整性校验

- 已在临时副本应用撤销补丁，确认两个章节精确恢复到精简前原文，并保留范围外模拟的后续修改；实际正文未执行回退。
- 修改前后快照均与对应 Git 版本逐字一致。快照和补丁保留历史文本中已有的尾随空格，以支持精确恢复。
- 建立检查点时的正文 SHA-256：`dd684b1940c594c5e01efa6501c89d1366fb0fab4e0a563ad2440f91d70150ce`
- `before.tex` SHA-256：`2cb746e614ec96d9e57d9cbb4f90e018e017bf417dbfe20e732a6ebc23223cc4`
- `after.tex` SHA-256：`3e25577d2d4508fcdfe9195c7f5bf9053ba320bc46486ac8ed708b3f4d98f92b`

## 回退历史

尚未回退。
