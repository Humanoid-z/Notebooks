# 基于黎曼流形的测地线卷积神经网络Geodesic Convolutional Neural Networks on Riemannian Manifolds

# 摘要

特征描述符在几何分析和处理应用中发挥着至关重要的作用，包括形状对应、检索和分割。在本文中，我们引入了测地线卷积神经网络(GCNN)，它是卷积网络(CNN)范式在非欧几里得流形上的推广。我们的构造是基于极坐标的局部测地线系统来提取“patch”，然后通过一连串卷积和线性及非线性算子。卷积核系数和线性权重均为优化变量，学习以最小化特定任务的成本函数。我们使用GCNN学习不变的形状特征，允许在形状描述、检索和对应等问题中实现最先进的性能。

# 1 引言

特征描述符是形状分析中普遍使用的工具。广义地说，局部特征描述符为形状上的每个点分配一个多维向量，表示该点周围形状的局部结构。全局描述符描述整个形状。局部特征描述符用于更高级的任务，如建立形状之间的对应关系、形状检索或分割。全局描述符通常是通过聚合局部描述符产生的，例如使用bag-of-features范式。描述符构造在很大程度上依赖于应用场景，人们通常试图使描述符具有鉴别性(捕捉对特定应用重要的结构，例如区分两类形状、健壮性(对某些类型的变换或噪声不变性)、紧凑性(低维)和计算效率。

**贡献：**提出 GCNN，基于类似于图像块的局部测地坐标系将CNN范式扩展到非欧氏流形。与以往在非欧cnn上的工作相比，我们的模型是可泛化的(即可以在一组形状上训练，然后应用到另一组形状上)、局部的，并允许捕获各向异性结构。证明了HKS， WKS，最优谱描述子和内部形状上下文可以作为GCNN的特定构型;因此，我们的方法是对以前流行的描述符的推广。实验结果表明，该模型能够在包括形状描述符构造、形状检索和形状对应等一系列问题上取得良好的性能。

# 2 背景

我们将3D形状建模为一个连通的光滑紧凑的二维流形(表面)X，可能带有边界$\partial X$。在每一点x附近，流形与一个二维欧氏空间同胚，称为切平面，由$T_xX$指代。黎曼度量是在在切空间上平滑地依赖于x的内积$\langle  \cdot,\cdot \rangle_{T_xX}:T_xX\times T_xX\rightarrow \mathbb{R}$

**拉普拉斯-贝尔特拉米算子(LBO)**是一个正半定算子$$

**离散化**：表面X采样了N个点$x_1,...,x_N$。在这些点上构造一个三角形网格(V, E, F)，其顶点$V=\{1,...,N\}$，其中每个内边$ij\in E$正好被两个三角形面$ikj$和$jhi$共享，边界边只属于一个三角面。直接连接到i的顶点集$\{j\in V:ij\in E\}$被称作i的1环。一个实值函数$f:X\rightarrow \mathbb{R}$在网格的顶点上采样，可以用一个N维向量$f=(f(x_1),...,f(x_N))^T$表示。LBO的离散版本为一个$N\times N$矩阵$L=A^{-1}W$

# 4 流形上的卷积神经网络

## 4.1 测地线卷积

我们引入了非欧几里得域上的卷积概念，它遵循“与模板相关”的思想，通过在点x构造一个测地线极坐标的局部系统，如图1所示，来提取流形上的补丁。径向坐标构造为p-level集$\{x':d_X(x,x')=\rho\}$的测地线(最短路径)距离函数$\rho\in[0,\rho_0]$;称$\rho_0$为测地线盘的半径。根据经验，选择一个足够小的$\rho_0≈形状的测地线直径的1\%$产生有效的拓扑盘。角坐标以一组以$\theta$方向从x出发的测地线$\Gamma (x,\theta)$表示。这样的射线垂直于测地线距离水平集。注意，角的坐标原点的选择是任意的。对于边界点，过程是非常相似的，唯一的区别是，我们不是映射到圆盘，而是映射到半圆盘。

让$\Omega(x):B_{\rho_0}\rightarrow[0,\rho_0]\times[0,2\pi)$表示从流形到x周围局部测地线极坐标$(\rho,\theta)$的双目标映射，并让$(D(x)f)(\rho,\theta)=(f\circ \Omega^{-1}(x))(\rho,\theta)$作为在局部坐标中插入$f$的patch算子。我们可以把$D(x)f$看作流形上的一个“patch”，用它来定义我们所说的测地线卷积(GC)，
$$
(f \star a)(x)=\sum_{\theta, r} a(\theta+\Delta \theta, r)(D(x) f)(r, \theta)
$$
其中$a(\theta,r)$是应用于patch的filter。由于角度坐标任意，过滤器可以旋转任意角度$\Delta \theta$。

**Patch算子**：
$$
\begin{aligned}
(D(x) f)(\rho, \theta) &=\int_{X} v_{\rho, \theta}\left(x, x^{\prime}\right) f\left(x^{\prime}\right) d x^{\prime} \\
v_{\rho, \theta}\left(x, x^{\prime}\right) &=\frac{v_{\rho}\left(x, x^{\prime}\right) v_{\theta}\left(x, x^{\prime}\right)}{\int_{X} v_{\rho}\left(x, x^{\prime}\right) v_{\theta}\left(x, x^{\prime}\right) d x^{\prime}}
\end{aligned}
$$
径向插值权重是一个到x的测地线距离的高斯函数$v_{\rho}\left(x, x^{\prime}\right)$，正比于$e^{-(d_X(x,x')-\rho)^2/\sigma^2_\rho}$，以$\rho$为中心。角权值为一个高斯函数$v_\theta(x,x')$正比于$e^{-d_X^2(\Gamma (x,\theta),x')/\sigma^2_\rho}$。point-to-set距离$d_X(\Gamma (x,\theta),x')=\min_{x''\in \Gamma (x,\theta)}d_X(x'',x')$，即$x'$到射线$\Gamma (x,\theta)$最短测地距离。

**离散Patch算子**：在三角形网格上，测地线极坐标的离散局部系统具有$N_θ$角和$N_ρ$径向箱(radial bins)。从顶点i开始，首先用$N_θ$射线将i的1环划分到等角的容器中，将第一条射线与其中一条边对齐。接下来使用一种展开步骤将射线传播到相邻的三角形，产生形成角容器的多条线。径向容器作为使用快速行进(fast marching)计算的测地线距离函数的水平集。

我们将离散Patch算子表示为一个$N_\theta N\rho N\times N$矩阵，应用于定义在网格顶点上的函数，并在每个顶点产生patch。这个矩阵是非常稀疏的，因为函数在几个邻近顶点的值只对每个局部geodesic polar bin有贡献。

## 4.2测地线卷积神经网络

测地卷积(Geodesic convolution, GC)层取代了经典欧氏cnn中的卷积层。由于角坐标模糊，我们计算所有$N_\theta$旋转的滤波器的测地线卷积结果，
$$
f_{\Delta \theta, q}^{\text {out }}(x)=\sum_{p=1}^{P}\left(f_{p} \star a_{\Delta \theta, q p}\right)(x), \quad q=1, \ldots, Q
$$
其中$a_{\Delta \theta, q p}(\theta,r)=a_{qp}(\theta+\Delta\theta,r)$为第q个filter库的第p个filter旋转$\Delta\theta=0,\frac{2 \pi}{N_{\theta}}, \ldots, \frac{2 \pi\left(N_{\theta}-1\right)}{N_{\theta}}$的系数