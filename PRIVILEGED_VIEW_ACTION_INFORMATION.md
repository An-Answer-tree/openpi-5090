# 特权视角动作信息

## 定义

特权视角动作信息不是普通的视觉特征，也不是表示图像重要性的标量。它表示：

> 在教师模型生成某个动作 token 时，一个特权视角通过真实注意力计算写入动作表征的定向更新。

本文将这一计算对象称为**视角分解的动作注意力残差**
（view-decomposed action-attention residual）。它位于教师模型的动作表征空间，描述
视觉信息如何被教师模型用于当前动作预测。

## 数学形式

考虑教师模型第 \(\ell\) 层的动作 token \(i\)。对注意力头 \(h\)，动作 token
产生查询向量 \(\mathbf{q}_{\ell,h,i}\)，所有可见 token 产生键和值
\(\mathbf{k}_{\ell,h,j}\) 和 \(\mathbf{v}_{\ell,h,j}\)。教师原始注意力权重为

$$
p_{\ell,h,i,j}
=
\frac{
\exp\!\left(\mathbf{q}_{\ell,h,i}^{\mathsf{T}}
\mathbf{k}_{\ell,h,j}/\sqrt{d_h}\right)
}{
\sum_{r\in\mathcal{S}}
\exp\!\left(\mathbf{q}_{\ell,h,i}^{\mathsf{T}}
\mathbf{k}_{\ell,h,r}/\sqrt{d_h}\right)
},
$$

其中，\(\mathcal{S}\) 包含该动作 token 可以读取的全部有效图像、语言和动作 token。
因此，视角信息始终在教师模型的完整上下文中计算，而不是在单个视角内部重新归一化。

设 \(\mathcal{T}_v\) 是视角 \(v\) 的图像 token 集合。该视角传递给动作 token \(i\)
的多头注意力消息为

$$
\mathbf{a}_{\ell,i}^{(v)}
=
\operatorname{Concat}_{h=1}^{H}
\left(
\sum_{j\in\mathcal{T}_v}
p_{\ell,h,i,j}\,\mathbf{v}_{\ell,h,j}
\right).
$$

经过教师模型真实的注意力输出投影 \(\mathbf{W}_{\ell}^{O}\) 和残差门
\(\mathbf{g}_{\ell,i}\) 后，得到该视角的动作信息：

$$
\mathbf{m}_{\ell,i}^{(v)}
=
\mathbf{g}_{\ell,i}
\odot
\mathbf{W}_{\ell}^{O}
\mathbf{a}_{\ell,i}^{(v)}
$$

其中，\(\mathbf{m}_{\ell,i}^{(v)}\) 与动作 token 的隐藏状态维度相同。对于
agentview 和 wrist 两个特权视角，教师分别产生

$$
\mathbf{m}_{\ell,i}^{(\mathrm{agent})}
,\qquad
\mathbf{m}_{\ell,i}^{(\mathrm{wrist})}.
$$

二者的和表示两个特权视角共同写入当前动作 token 的信息：

$$
\mathbf{m}_{\ell,i}^{(\mathrm{priv})}
=
\mathbf{m}_{\ell,i}^{(\mathrm{agent})}
+
\mathbf{m}_{\ell,i}^{(\mathrm{wrist})}.
$$

## 为什么它与动作相关

注意力权重由动作 token 的查询向量决定：

$$
p_{\ell,h,i,j}
\propto
\exp\!\left(
\mathbf{q}_{\ell,h,i}^{\mathsf{T}}\mathbf{k}_{\ell,h,j}
/\sqrt{d_h}
\right).
$$

不同动作 token 具有不同的查询向量，因此会从相同图像中读取不同的信息。例如，接近物体、
调整抓取姿态和闭合夹爪所需的视觉证据并不相同。该向量还随噪声动作、流时间和上下文变化，
所以它不是固定的图像表示，而是**由当前动作计算条件化的特权视角信息**。

## 与常见视觉量的区别

| 对象 | 表示内容 | 与特权视角动作信息的区别 |
|---|---|---|
| 视觉编码器特征 | 图像内容的通用表示 | 尚未说明这些内容如何参与动作计算 |
| 注意力权重 | 动作 token 对各输入 token 的读取比例 | 只有权重，没有包含被读取的 value 内容和输出投影 |
| 显著性或重要性分数 | 某区域可能有多重要 | 通常是标量，不能作为动作隐藏状态的更新量 |
| 因果贡献 | 删除或干预某信息造成的输出变化 | 当前量是注意力输出的精确分解，不等同于因果效应 |
| 特权视角动作信息 | 特权视角写入动作残差流的向量更新 | 同时包含 action query、attention、value、输出投影和残差门 |

## 可加分解性质

由于注意力的 value 聚合、输出投影和残差门对各来源的消息保持线性，若所有可见 token
被划分为互不重叠的来源集合 \(\mathcal{V}\)，则完整的动作注意力更新可以写成

$$
\Delta\mathbf{h}_{\ell,i}^{(\mathrm{attn})}
=
\sum_{v\in\mathcal{V}}
\mathbf{m}_{\ell,i}^{(v)}.
$$

因此，\(\mathbf{m}_{\ell,i}^{(v)}\) 不是人为定义的辅助特征，而是教师模型原始注意力更新
按照 token 来源得到的精确分量。agentview 和 wrist 分量只覆盖两个特权视觉来源；语言、
动作和其他 token 的分量仍属于完整注意力更新的其余部分。

## 在学生模型中的作用

训练时，学生只能观察弱视角。学生根据自己的动作隐藏状态预测教师的两路特权视角动作信息：

$$
\widehat{\mathbf{m}}_{\ell,i}^{(\mathrm{agent})},
\widehat{\mathbf{m}}_{\ell,i}^{(\mathrm{wrist})}
=
f_{\theta}\!\left(\mathbf{h}_{\ell,i}^{(S)}\right).
$$

该目标要求学生从弱视角和动作上下文中恢复教师在训练时从特权视角获得的动作相关信息。
预测结果可以作为辅助监督，也可以通过可学习门控补充学生的动作表征：

$$
\widetilde{\mathbf{h}}_{i}^{(S)}
=
\mathbf{h}_{i}^{(S)}
+
\tanh(\alpha)
\sum_{v\in\{\mathrm{agent},\mathrm{wrist}\}}
\operatorname{sg}\!\left(
\widehat{\mathbf{m}}_{\ell,i}^{(v)}
\right),
$$

其中，\(\alpha\) 是可学习门，\(\operatorname{sg}\) 表示停止梯度。部署时不再需要教师或
特权视角；学生只使用弱视角生成动作。

## 准确的表述边界

该向量可以表述为：

> 教师模型中由动作 token 查询、从指定特权视角读取并写入动作残差流的信息。

不应将其直接表述为：

- 图像中全部与任务有关的信息；
- 特权视角对最终动作的因果效应；
- 视觉区域的重要性或显著性；
- 教师模型完整的动作知识。

它精确描述的是**教师单层注意力计算中的来源特定动作更新**。
