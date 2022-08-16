# 摘要

SchNetPack是一个预测分子和材料的势能面及其它链子化学性质的深度神经网络工具包。它能够提供原子神经网络的基本构建模块，管理它们的训练，并提供对公共基准数据集的轻松访问。目前SchNetPack包括(加权的)原子中心对称函数和深度张量神经网络SchNet的实现，以及现成的脚本。基于pytorch框架构建。

# 1 引言

量子化学机器学习模型的一个常见子类是原子神经网络。这些模型有各种各样的体系结构，可以大致分为两类：（1）基于描述符的模型，它采用原子系统的预定义表示作为输入；（2）端到端架构，它直接从原子类型和位置学习表示。

SchNetPack包含SchNet的实现(一种端到端连续卷积架构)，Behler-Parrinello网络(基于原子中心对称函数(ACSF))以及它的一个扩展，使用了加权原子中心对称函数(wACSF)。

# 2 模型

SchNetPack的模型有2个主要部件：表示模块和预测模块。前者接收原子系统的构型作为输入并生成描述每个原子在其化学环境中的特征向量，后者使用基于原子的表示预测原子系统的属性。基于描述符的结构与端到端结构的唯一区别是表示模块是固定的还是从数据中学习的。

## 2.1表示层

一个包含n个原子的原子系统可以描述成它的原子序数向量$\mathrm{Z}=(Z_1,...,Z_n)$和位置矩阵$R=(\mathrm{r_1},...,\mathrm{r_n})$。原子相互距离为$r_{ij}=\parallel r_i-r_j\parallel $

### 2.1.1(w)ACSF

Behler−Parrinello网络已经证明对小分子、金属、分子团簇、大块材料、表面、水和固液界面这样的系统非常有用。由于这些庞大的应用数量，Behler - Parrinello网络已经成为原子系统中非常成功的神经网络体系结构。

对于这些网络，所谓的原子中心对称函数(ACSFs)形成了原子系统的表示。与SchNet的方法不同，ACSF必须在训练前决定。因此在可用训练数据不足以让端到端范式学习到合适的表示的情况下，使用对称函数是有利的。另一方面，使用严格的人工特征可能降低模型的泛化性。

ACSF通过径向分布函数和角分布函数的组合来描述中心原子周围的局部化学环境。

#### 2.1.1.1径向对称函数

径向ACSF描述符形式如下：
$$
G_{i, \alpha}^{\mathrm{rad}}=\sum_{j \neq i}^{N} g\left(Z_{j}\right) e^{-\gamma_{\alpha}\left(r_{i j}-\mu_{\alpha}\right)^{2}} f\left(r_{i j}\right)
$$
其中i为中心原子，j是所有邻近的原子的总和 。$\gamma_\alpha$和$\mu_\alpha$是调整高斯函数宽和中心的参数。通常，一组$n_{rad}$个径向对称函数具有不同的参数组合$\alpha \in\{1,...,n_{rad}\}$。在SchNetPack中，合适的$\gamma_\alpha$和$\mu_\alpha$是通过0和空间截断系数$r_c$之间的等距网格自动确定，采用参考文献24中的经验参数化策略。

截断函数$f$确保只有接近中心原子i的原子能加入求和，其形式如下：
$$
f\left(r_{i j}\right)=\left\{\begin{array}{ll}
\frac{1}{2}\left(\cos \left(\frac{\pi r_{i j}}{r_{c}}\right)+1\right) & \text { if } r_{i j} \leq r_{c} \\
0 & \text { else }
\end{array}\right.
$$
为方便起见，下面使用符号$f_{ij}=f(r_{i j})$。最后$g(Z_j)$是一个元素相关的权重函数。ACSF中形式如下：
$$
g\left(Z_{j}\right)=\delta_{Z_{j}, Z_{a}}=\left\{\begin{array}{ll}
1 & \text { if } Z_{j}=Z_{a} \\
0 & \text { else }
\end{array}\right.
$$
因此，径向ACSF总是定义在中心原子和属于特定化学元素的相邻原子之间。

#### 2.1.1.2角对称函数

原子间的角度信息由$n_a$角对称函数编码
$$
\begin{aligned}
G_{i, \alpha}^{\text {ang }}=& 2^{1-\zeta_{\alpha}} \sum_{j \neq i, k>j}^{N} g\left(Z_{j}, Z_{k}\right)\left(1+\lambda \theta_{i j k}\right)^{\zeta_{\alpha}} \\
& \times \exp \left[-\gamma_{\alpha}\left(r_{i j}^{2}+r_{i k}^{2}+r_{j k}^{2}\right)\right]_{i j} f_{i k} f_{j k}
\end{aligned}
$$
其中$\theta_{i j k}$是原子i,j,k的夹角，参数$\lambda$取值$\lambda=±1$，在0和π之间移动角项的最大值。变量$\zeta_{\alpha}$为控制这个最大值附近宽度的超参数。$\gamma_\alpha$再次控制高斯函数的宽度。同径向ACSF一样，一组$n_{rad}$个具有不同的参数组合的角函数$\alpha \in\{1,...,n_{rad}\}$被选来描述局部环境。对于角ACSF，权重函数$g(Z_k,z_j)$形式如下：
$$
g\left(Z_{k}, Z_{j}\right)=\frac{1}{2}\left(\delta_{Z_{j} Z_{a}} \delta_{Z_{k} Z_{b}}+\delta_{Z_{j} Z_{b}} \delta_{Z_{k} Z_{a}}\right)
$$
上式计算了属于特定元素对(如O-H或O-O)的相邻原子j和k的贡献

由于g的选择，ACSF总是为元素对(径向)或三元组(角度)定义，对于这些组合必须提供至少一个参数化函数$G_{i,\alpha}$。因此，ACSF的数量与不同化学物质的数量成二次增长。对于包含超过4个元素的系统如(QM9)，这可能导致ACSF的数量过于庞大。

最近，人们提出了替代的权重函数来规避上述问题。在所谓的加权ACSF(wACSFs)中，径向加权函数设为$g(Z_j)=Z_j$，角函数设为$g(Z_k,Z_j)=Z_kZ_j$。通过这种简单的重新参数化，所需对称函数的数量与系统中存在的实际元素数量无关，从而得到更紧凑的描述符。SchNetPack使用wACSFs作为Behler−Parrinello势的标准描述符。

不管权重g的选择是什么，径向和角对称函数都被串联起来作为最后一步，以形成原子系统的表示
$$
X_{i}=\left(G_{i, 1}^{\mathrm{rad}}, \ldots G_{i, n_{\mathrm{rad}}}^{\mathrm{rad}},\left.G_{i, 1}^{\mathrm{ang}}\right|_{\lambda=\pm 1}, \ldots,,\left.G_{i, n_{\mathrm{ang}}}^{\mathrm{ang}}\right|_{\lambda=\pm 1}\right)
$$
$X_i$能够作为预测模块的输入

### 2.1.2 SchNet

SchNet是基于连续过滤器卷积的端到端深度神经网络架构。它遵循深度张量神经网络框架，例如在通过一系列交互块引入系统构型之前，先从刻画原子类型的embedding向量开始构建原子表示。

深度学习中的卷积层通常作用于离散信号，如图像。连续滤波器卷积是对非网格对齐的输入信号的一种推广，例如任意位置的原子。与(w)ACSF网络基于严格的手工特征相反，SchNet将原子系统的表示适应于训练数据。SchNet是一个多层神经网络，由一个嵌入层和几个交互块组成

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/schnetpack.png" alt="image-20220803142640695" style="zoom: 67%;" />

#### 2.1.2.1 原子嵌入层

使用一个嵌入层，每个原子类型$Z_i$由特征向量$x_i^0\in \mathbb{R}^F$表示，集中于一个矩阵$X^0=(x_1^0,...,x_n^0)$。特征维度为$F$。随机初始化嵌入层，并在训练过程中进行调整。在SchNet的所有其他层中，原子都是类似地描述，$l$层的特征用$X^l=(x_1^l,...,x_n^l)$表示，$x_i^l\in \mathbb{R}^F$。

#### 2.1.2.2 交叉层

这个构建块使用特征$x^l$和位置矩阵$R$计算交互。为了吸收相邻原子的影响，采用连续滤波器卷积，其定义如下:
$$
\mathbf{x}_{i}^{l+1}=\left(X^{l} * W^{l}\right) \equiv \sum_{j \in \operatorname{nbh}(i)} \mathbf{x}_{j}^{l} \odot W^{l}\left(r_{i j}\right)
$$
使用$\odot$符号指代元素乘，$\operatorname{nbh}(i)$为原子i的邻居。特别地，对于大型系统，建议引入径向截止范围。对于我们的实验，使用5$\mathring{A}$作为距离截止。

这里过滤器不是标准卷积层中的参数张量，而是一个过滤器生成神经网络$W^l:\mathbb{R}\rightarrow\mathbb{R}^F$，它将原子距离映射到过滤器值。过滤器生成器在径向基函数网格上展开原子位置，这些基函数与(w)ACSF的径向对称函数密切相关。

若干atom-wise层，即全连接层，定义为
$$
\mathbf{x}_{i}^{l+1}=W^{l} \mathbf{x}_{i}^{l}+\mathbf{b}^{l}
$$
分别应用于每个原子i，并重新组合每个原子表示中的特征。注意权重$W_l$和偏差$b_l$与i无关，所有原子特征$x_i^l$共享相同参数。因此，原子层参数的数量与原子的数量n无关。

综上所述，SchNet通过首先使用嵌入层获得特征$X^0$来获得原子系统的潜在表示。然后用L个交互块对这些特征进行处理，得到潜在表示$X^L$，可以传递给预测块。

## 2.2预测模块

对于具有n个原子的原子系统，SchNet和(w)ACSF都提供表示$X_i,i\in\{1,...,n\}$，这些表示被一个预测块处理，以获得想要的原子系统的属性。根据感兴趣的属性，预测块有多种选择。通常，预测块由几个具有非线性的atom-wise层组成，这些层降低了特征维数，然后是跨原子的属性依赖聚合。

最常见的选择是Atomwise预测块，它将期望的分子特性P表示为原子贡献的总和:
$$
P=\sum_{i=1}^{n} p\left(\mathbf{x}_{i}\right)
$$
虽然这是一个适合大量性质的模型，如能量，密集性质，这些性质不随原子系统的原子数量n增长，而是表示为平均贡献。

Atomwise预测块适用于许多属性；然而特定于属性的预测块可以将先验知识合并到模型中。偶极矩预测块将偶极矩(μ)表示为
$$
\mu=\sum_{i=1}^{n} q\left(\mathbf{x}_{i}\right)\left(\mathbf{r}_{i}-\mathbf{r}_{0}\right)
$$
其中$q:\mathbb{R}^F\rightarrow\mathbb{R}$可以理解为潜在的原子电荷，$r_0$表示系统的质心。

ElementalAtomwise预测块和Atomwise预测块不同在于：不会对所有原子特征$X_i$使用相同的网络，而是对不同的化学元素使用独立的网络。这对于(w)ACSF的表示特别有用。类似地，ElementalDipoleMoment是为偶极矩定义的。