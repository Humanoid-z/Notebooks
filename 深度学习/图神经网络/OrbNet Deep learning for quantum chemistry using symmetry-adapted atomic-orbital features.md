# 摘要

我们介绍了一种机器学习方法，其中薛定谔方程的能量解是使用对称自适应原子轨道特征和图神经网络结构预测的。在利用从半经验电子结构计算中获得的低成本特征预测密度泛函理论结果时，OrbNet在学习效率和可转移性方面优于现有方法。用于药物类分子数据集的应用，包括QM7b-T、QM9、GDB-13-T、DrugBank和Folmsb的构象基准数据。OrbNet在密度泛函理论的化学精度范围内预测能量，其计算成本降低了1000倍或更多。

# 1 引言

密度泛函理论能够给出势能面的精确解，但成本较高，与力场和半经验量子力学理论相比，DFT的适用性仅限于相对较小的分子或适度的构象取样。

在量子化学的背景下，许多应用都集中在使用原子或几何特定的特征表示和基于核的或神经网络的机器学习架构。最近的研究集中在分子的抽象表征上，例如从低成本的电子结构计算中获得的量子力学性质以及新型的基于图的神经网络的应用提高迁移学习和学习效率的技术。

在这方面，我们提出了一种新的方法(OrbNet)，基于分子的对称性适应原子轨道(SAAOs)特征和使用深度学习量子力学性质的图神经网络方法。我们展示了新方法在分子性质预测方面的性能，包括在一系列有机和类药物分子数据集中分子的总和相对构象能。该方法能够以完全量子力学精度预测分子势能面，同时能够大幅降低计算成本;此外，该方法在训练效率和跨不同分子体系的可转移精度方面优于现有方法。

# 2 方法

这项工作的目标是机器学习从输入特征值{f}到回归标签(量子机械能)的迁移映射，
$$
E≈E^{ML}[\{f\}]
$$
OrbNet的关键要素包括在SAAO基础上对特征的高效评估，利用具有边缘和节点属性和消息传递层(MPLs)的图神经网络架构，以及确保结果能量的广博性的预测阶段。我们在本节中总结了这些要素，并讨论了OrbNet和其他ML方法之间的关系。尽管本文提出了将特征从半经验质量特征映射到DFT质量标签的结果，但相对于用于特征的平均场方法[即也允许Hartree-Fock (HF)和DFT]和用于生成标签的理论水平(即也允许耦合聚类和其他相关波函数方法参考数据)，该方法是通用的。

## A.SAAO特征

设${φ^A_{n,l,m}}$为具有原子指数A和标准主量子数和角动量量子数n,l,m的原子轨道基函数集合。设C为通过HF理论、DFT或半经验方法等平均场电子结构计算得到的对应分子轨道系数矩阵。在AO基上的分子系统的单电子密度矩阵为
$$
P_{\mu v}=2 \sum_{i \in \mathrm{occ}} C_{\mu i} C_{v i}
$$
(对于闭壳系统)。我们通过对角化与指标a、n、l相关的对角密度矩阵块，构造了一个旋转不变的对称原子轨道基{ˆφA n,l,m}
$$
\mathbf{P}_{n l}^{A} \mathbf{Y}_{n l}^{A}=\mathbf{Y}_{n l}^{A} \operatorname{diag}\left(\lambda_{n l m}^{A}\right),
$$
其中$[\mathbf{P}_{n l}^{A}]_{mm'}=\mathbf{P}_{n lm,nlm'}^{A}$。对于s轨道(l = 0)，这个对称过程显然是微不足道的，可以跳过。通过构建，saao是局域的，并且与分子的几何扰动相一致，与最小化局域目标函数(如Pipek-Mezey和Boys)得到的局域分子轨道(LMOs)相比，SAAO是通过一系列非常小的对角化得到的，不需要迭代过程。SAAO特征向量$Y^A_{nl}$被聚合形成一个块对角变换矩阵Y，它指定从AO到SAAO的完整变换
$$
\left|\hat{\phi}_{p}\right\rangle=\sum_{\mu} Y_{\mu p}\left|\phi_{\mu}\right\rangle
$$
其中μ和p分别代表AOs和SAAOs。

我们利用在SAAO基中计算量子化学算符得到的由张量组成的ML特征{f}。此后，所有的量子力学矩阵都将被假定为SAAO基，包括密度矩阵P和重叠矩阵S。根据我们之前的工作，特征包括在SAAO基上的福克(F)，库仑(J)和交换(K)算子的期望值。在这项工作中，我们另外包括SAAO密度矩阵P，轨道质心距离矩阵D，核心哈密顿矩阵H，和重叠矩阵S;其他量子力学矩阵元素也可以用于特征化。

## B.近似库仑和交换SAAO特征

当采用半经验量子化学理论时，由于需要计算四指数电子排斥积分，SAAO特征生成的计算瓶颈为J项和K项。我们通过引入广义的Mataga-Nishimoto-Ohno-Klopman公式(如sTDA-xTB方法)来解决这个问题
$$
(p q \mid r s)^{\mathrm{MNOK}}=\sum_{A} \sum_{B} Q_{p q}^{A} Q_{r s}^{B} \gamma_{A B} .​
$$
A，B为原子索引，p q r s是SAAO索引
$$
\gamma_{A B}^{\{\mathrm{J}, \mathrm{K}\}}=\left(\frac{1}{R_{A B}^{y_{\{, \mathrm{K}\}}}+\eta^{-y_{\{, \mathrm{K}\}}}}\right)^{1 / y_{\{, \mathrm{K}\}}}
$$
其中$R_{AB}$为原子A和B之间的距离，η为原子A和B的平均化学硬度，$y_{\{J,K\}}$为表示阻尼相互作用核衰减行为的经验参数，$γ^{\{J,K\}}_{AB}$。在这项工作中，我们使用的$y_J = 4$和$y_K = 10$类似于sTDA-RSH方法。过渡密度$Q^A_{pq}$是通过Löwdin种群分析计算出来的，

## C.OrbNet

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/OrbNet_workflow..png" alt="image-20220805153635103" style="zoom:50%;" />

OrbNet将分子系统编码为图结构数据，并利用图神经网络(GNN)机器学习体系结构。GNN表示数据为一个属性图$G(V, E, X, X^e)$，节点V，边E，节点属性$X: V→R^{n×d}$，边属性$X^e: E→R^{m×e}$，其中n = |V|， m = |E|， d和e分别为每个节点和每个边的属性个数。

具体而言，OrbNet对分子系统采用图表示，节点属性对应对角SAAO特征$Xu = [F_{uu}, J_{uu}, K_{uu}, P_{uu}, H_{uu}]$，边属性对应非对角SAAO特征$X^e_{uv}=[F_{uv}, J_{uv}, K_{uv},D_{uv}, P_{uv},S_{uv}, H_{uv}]$。通过引入包含边的边属性截止值，将在无限远处分离的非相互作用的分子系统编码为断开图，从而满足大小一致性。

通过径向基函数在图表示中引入非线性输入特征变换，提高了模型的容量
$$
\begin{array}{l}
\mathbf{h}_{u}^{\mathrm{RBF}}=\left[\phi_{1}^{\mathrm{h}}\left(\tilde{X}_{u}\right), \phi_{2}^{\mathrm{h}}\left(\tilde{X}_{u}\right), \ldots, \phi_{n_{\mathrm{r}}}^{\mathrm{h}}\left(\tilde{X}_{u}\right)\right], \\
\mathbf{e}_{u v}^{\mathrm{RBF}}=\left[\phi_{1}^{\mathrm{e}}\left(\tilde{X}_{u v}^{\mathrm{e}}\right), \phi_{2}^{\mathrm{e}}\left(\tilde{X}_{u v}^{\mathrm{e}}\right), \ldots, \phi_{m_{\mathrm{r}}}^{\mathrm{e}}\left(\tilde{X}_{u v}^{\mathrm{e}}\right)\right],
\end{array}
$$
其中$\tilde{X}$和$\tilde{X}^e$为$n\times d$和$m\times e$的预归一化属性矩阵。采用正弦基函数$φ^h_n(r) = sin(πnr)$进行节点嵌入。受最近一项基于原子的GNN研究引入的嵌入方法的启发，我们采用了0阶球面贝塞尔函数进行边嵌入
$$
\phi_{m}^{\mathrm{e}}(r)=j_{0}^{m}\left(r / c_{\mathbf{X}}\right) \cdot I_{\mathbf{X}}(r)=\sqrt{\frac{2}{c_{\mathbf{X}}}} \frac{\sin \left(\pi m r / c_{\mathbf{X}}\right)}{r / c_{\mathbf{X}}} \cdot I_{\mathbf{X}}(r)
$$
式中$c_X (X∈\{F, J, K, D, P, S, H\})$为$\tilde{X}^e_{uv}$操作符特定的上截止值。为了保证当节点进入截止点时特征变化平稳，我们进一步引入软化器$I_X(r)$，
$$
I_{\mathbf{X}}(r)=\left\{\begin{array}{ll}
\exp \left(-\frac{c_{\mathbf{X}}^{2}}{\left(|r|-c_{\mathbf{X}}\right)^{2}}+1\right), & \text { if } 0 \leq|r|<c_{\mathbf{X}} \\
0, & \text { if }|r| \geq c_{\mathbf{X}}
\end{array}\right.
$$
注意$φ^e_m(r)$在边界处衰减为零以保证尺寸一致性，软化剂在边界处是无限阶可微的，这消除了分子几何扰动可能产生的表示噪声。为了确保在添加任意数量的零边特征时输出在机器精度上是恒定的，这对于提取分析梯度和训练势能面至关重要，我们还引入了一种与消息传递机制相结合的“辅助边缘”方案
$$
\mathbf{e}_{u v}^{\mathrm{aux}}=\mathbf{W}^{\mathrm{aux}} \cdot \mathbf{e}_{u v}^{\mathrm{RBF}},
$$
其中$\mathbf{W}^{\mathrm{aux}}$为可训练参数矩阵。利用神经网络模块对径向基函数嵌入进行变换，得到0阶节点和边缘属性，
$$
\mathbf{h}_{u}^{0}=\operatorname{Enc}_{\mathrm{h}}\left(\mathbf{e}_{u v}^{\mathrm{RBF}}\right), \mathbf{e}_{u v}^{0}=\operatorname{Enc}_{\mathrm{e}}\left(\mathbf{h}_{u}^{\mathrm{RBF}}\right)
$$
其中$Enc_h$和$Enc_e$为残差块，包含三层dense层。与基于原子的消息传递神经网络相比，这种额外的嵌入转换捕获物理操作符之间的交互。

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/OrbNet_MPL_update.png" alt="image-20220805153520534" style="zoom:50%;" />

节点和边属性通过transformer-motivated消息传递机制进行更新。对于给定的消息传递层(MPL)$l+1$，将每条边携带的信息编码为消息函数$m^l_{uv}$和相关的注意权$w^l_{uv}$，通过图卷积运算积累为节点特征。整个消息传递机制如下
$$
\mathbf{h}_{u}^{l+1}=\mathbf{h}_{u}^{l}+\sigma\left(\mathbf{W}_{\mathrm{h}}^{l} \cdot\left[\bigoplus_{i}\left(\sum_{v \in N(u)} w_{u v}^{l, i} \cdot \mathbf{m}_{u v}^{l}\right)\right]+\mathbf{b}_{\mathrm{h}}^{l}\right)
$$
其中$m^l_{uv}$为在每个节点计算的消息函数
$$
\mathbf{m}_{u v}^{l}=\sigma\left(\mathbf{W}_{\mathrm{m}}^{l} \cdot\left[\mathbf{h}_{u}^{l} \odot \mathbf{h}_{v}^{l} \odot \mathbf{e}_{u v}^{l}\right]+\mathbf{b}_{\mathrm{m}}^{l}\right)
$$
卷积核权重$w^{l,i}_{uv}$作为(多头)注意得分来表征轨道对的相对重要性
$$
w_{u v}^{l, i}=\sigma_{\mathrm{a}}\left(\sum\left[\left(\mathbf{W}_{\mathrm{a}}^{l, i} \cdot \mathbf{h}_{u}^{l}\right) \odot\left(\mathbf{W}_{\mathrm{a}}^{l, i} \cdot \mathbf{h}_{v}^{l}\right) \odot \mathbf{e}_{u v}^{l} \odot \mathbf{e}_{u v}^{\mathrm{aux}}\right] / n_{\mathrm{e}}\right)
$$
求和作用于被加向量的元素。其中，指数i表示注意力头，$n_e$为隐边特征$e^l_{uv}$的维数，⊕表示向量拼接运算，⊙表示Hadamard乘积，⋅表示矩阵向量乘积。

边特征更新形式如下：
$$
\mathbf{e}_{u v}^{l+1}=\sigma\left(\mathbf{W}_{\mathrm{e}}^{l} \cdot \mathbf{m}_{u v}^{l}+\mathbf{b}_{\mathrm{e}}^{l}\right)
$$
$\mathbf{W}_{\mathrm{m}}^{l}, \mathbf{W}_{\mathrm{h}}^{l}, \mathbf{W}_{\mathrm{e}}^{l}, \mathbf{b}_{\mathrm{m}}^{l}, \mathbf{b}_{\mathrm{h}}^{l}, \mathbf{b}_{\mathrm{e}}^{l}$和$a^l$为MLP的可训练参数矩阵。$\mathbf{W}^{l,i}_a$为MLP和注意力头的可训练参数矩阵，$\sigma(\cdot)$为带归一化层的激活函数，$\sigma_a(\cdot)$为生成注意力得分的激活函数。

OrbNet的解码阶段旨在确保能量预测的规模-广泛性。该机制输出嵌入层(l = 0)和所有MPLs (l = 1,2,…,l)的节点解析能量贡献，以预测和所有节点及MPLs相关的能量成分。最后的能量预测$E^{ML}$是通过对l求和得到的

为每个节点u，然后对节点(即轨道)进行one-body求和
$$
E^{\mathrm{ML}}=\sum_{u \in \mathbf{V}} \varepsilon_{u}=\sum_{u \in \mathbf{V}} \sum_{l=0}^{L} \operatorname{Dec}^{l}\left(\mathbf{h}_{u}^{l}\right),
$$
其中解码网络$Dec^l$是多层感知机

# Dataset

**GDB-13-T**：对于来自GDB-13数据集的1000个分子，每个分子有6种构象，有13个重原子(C,O,N,S,Cl)。GDB-13：SMILES字符串 小分子数据库 近10亿结构

**DrugBank-T**：DrugBank数据集的168个分子，每个有6种构型，有14到30个重原子(C,O,N,S,Cl)

**Hutchison构象异构体数据集**：622个分子，每个有多达10种构象，9到50个重原子(C、O、N、F、P、S、Cl、Br和I)