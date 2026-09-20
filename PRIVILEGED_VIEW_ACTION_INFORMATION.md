# 特权视角动作信息

## 输入表示

教师模型接收两张同步图像、任务指令和带噪动作序列：

- `agentview`：场景相机图像；
- `wrist`：腕部相机图像；
- 任务指令：描述当前操作目标的文本；
- 带噪动作序列：真实动作与随机噪声按当前流时间混合得到的训练输入。

这些输入在进入 Transformer 前被转换为 token。这里的 token 是一个向量，不只表示
文本词元：图像被切分并编码为一组图像 token，文本被编码为一组语言 token，动作序列中
的每个位置被编码为一个动作 token。

每个动作 token 都有一个内部向量。Transformer 逐层更新该向量，模型最终用它预测动作。
下文将其称为动作隐藏向量。

对动作 token \(i\) 而言，**可读取的输入 token**是注意力掩码允许它参与计算的 token。
在当前教师模型中，这些 token 包括：

- agentview 中未被填充的图像 token；
- wrist 中未被填充的图像 token；
- 任务指令中未被填充的语言 token；
- 动作序列中的动作 token。

“可读取”只表示注意力掩码没有屏蔽该 token。填充位置、缺失图像和被掩码的位置不参与
注意力计算。

## 计算方式

注意力由 \(H\) 个并行分支计算，每个分支称为一个注意力头。在教师模型第 \(\ell\) 层，
注意力头 \(h\) 将动作 token \(i\) 转换为查询向量
\(\mathbf{q}_{\ell,h,i}\)，并将输入 token \(j\) 转换为键向量
\(\mathbf{k}_{\ell,h,j}\) 和值向量 \(\mathbf{v}_{\ell,h,j}\)。动作 token 对输入
token 的注意力权重为

$$
p_{\ell,h,i,j}
=
\frac{
\exp\!\left(
\mathbf{q}_{\ell,h,i}^{\mathsf{T}}
\mathbf{k}_{\ell,h,j}/\sqrt{d_h}
\right)
}{
\sum_{r\in\mathcal{S}_i}
\exp\!\left(
\mathbf{q}_{\ell,h,i}^{\mathsf{T}}
\mathbf{k}_{\ell,h,r}/\sqrt{d_h}
\right)
}.
$$

\(\mathcal{S}_i\) 是注意力掩码允许动作 token \(i\) 读取的全部输入 token。分母同时
包含图像、语言和动作 token，因此每个视角的权重是在完整输入上计算的。

设 \(\mathcal{T}_v\) 是视角 \(v\) 的图像 token 集合。该视角写入动作 token \(i\)
的向量为

$$
\mathbf{m}_{\ell,i}^{(v)}
=
\mathbf{g}_{\ell,i}
\odot
\mathbf{W}_{\ell}^{O}
\operatorname{Concat}_{h=1}^{H}
\left(
\sum_{j\in\mathcal{T}_v}
p_{\ell,h,i,j}\mathbf{v}_{\ell,h,j}
\right).
$$

其中：

- \(p_{\ell,h,i,j}\) 决定动作 token 从图像 token \(j\) 读取多少信息；
- \(\mathbf{v}_{\ell,h,j}\) 是被读取的图像内容；
- \(\mathbf{W}_{\ell}^{O}\) 将所有注意力头的结果映射回动作隐藏空间；
- \(\mathbf{g}_{\ell,i}\) 控制注意力输出以多大幅度加到原动作隐藏向量上；
- \(\mathbf{m}_{\ell,i}^{(v)}\) 与动作 token 的隐藏向量维度相同。

当前方法分别计算 agentview 和 wrist 对应的向量：

$$
\mathbf{m}_{\ell,i}^{(\mathrm{agent})},
\qquad
\mathbf{m}_{\ell,i}^{(\mathrm{wrist})}.
$$

这两个向量称为**特权视角动作信息**。它们是教师注意力输出中由两个特权视角产生的部分。

## 与动作的关系

注意力权重由动作 token 的查询向量 \(\mathbf{q}_{\ell,h,i}\) 决定。不同动作位置具有
不同的查询向量，因此可以从同一张图像读取不同内容。带噪动作和流时间也参与动作 token
的计算，所以特权视角动作信息随动作位置、带噪动作、流时间和输入场景变化。

特权视角动作信息位于动作隐藏空间，不是固定的图像表示。

## 与完整注意力输出的关系

将动作 token 可以读取的输入按来源划分为互不重叠的集合，例如 agentview、wrist、语言
和动作序列。由于值向量求和、输出投影和残差门对各来源都是线性运算，各来源向量之和等于
该动作 token 的完整注意力残差：

$$
\Delta\mathbf{h}_{\ell,i}^{(\mathrm{attn})}
=
\sum_{s\in\mathcal{P}_i}
\mathbf{m}_{\ell,i}^{(s)}.
$$

\(\mathcal{P}_i\) 表示输入来源的完整划分。agentview 和 wrist 只对应其中两个来源；
语言和动作 token 产生的部分不属于特权视角动作信息。

## 学生模型如何使用

学生模型只接收一个弱视角。训练时，预测器根据学生第 \(\ell\) 层的动作隐藏向量
\(\mathbf{h}_{\ell,i}^{(S)}\)，分别预测教师的两路特权视角动作信息：

$$
\left[
\widehat{\mathbf{m}}_{\ell,i}^{(\mathrm{agent})},
\widehat{\mathbf{m}}_{\ell,i}^{(\mathrm{wrist})}
\right]
=
f_{\theta}\!\left(\mathbf{h}_{\ell,i}^{(S)}\right).
$$

预测向量用于蒸馏损失。启用注入时，两路预测向量相加后，通过可学习门加入学生的动作隐藏
向量：

$$
\widetilde{\mathbf{h}}_{i}^{(S)}
=
\mathbf{h}_{i}^{(S)}
+
\tanh(\alpha)
\operatorname{sg}\!\left(
\widehat{\mathbf{m}}_{\ell,i}^{(\mathrm{agent})}
+
\widehat{\mathbf{m}}_{\ell,i}^{(\mathrm{wrist})}
\right).
$$

\(\alpha\) 是可学习门，\(\operatorname{sg}\) 表示注入路径不向预测器反向传播梯度。
部署时移除教师，学生只使用弱视角。

## 含义边界

| 名称 | 数学对象 |
|---|---|
| 图像特征 | 视觉编码器产生的图像 token |
| 注意力权重 | 标量 \(p_{\ell,h,i,j}\) |
| 特权视角动作信息 | 指定视角经过注意力加权、输出投影和残差门后得到的动作空间向量 |
| 因果效应 | 对输入进行干预后测得的输出变化 |

特权视角动作信息是教师单层注意力输出的来源分量。该定义不表示图像区域的重要性，也不表示
特权视角对最终动作的因果效应。
