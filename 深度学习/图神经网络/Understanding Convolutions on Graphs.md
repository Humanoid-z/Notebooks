https://distill.pub/2021/understanding-gnns/

https://zhuanlan.zhihu.com/p/107162772

# 图数据的相关问题

- 节点分类
- 图分类
- 节点聚类
- 边预测
- 影响力最大化：识别受影响的节点

解决以上这些问题的一种常见提前准备是节点表示学习：学习节点的embedding

通常GNN使用迭代过程计算节点的embedding，$h_v^{(k)}$表示节点v第k代的embedding。每次迭代都可以被认为是标准神经网络中的“层”。

# 图数据上的卷积操作

类似于2D卷积的3*3卷积，每做一次卷积只对目标节点的邻接点及其自身进行加权和计算。

PyG的代码实现：

定义$\mathcal{N}(i)$为节点i的邻接点的集合，对单一节点的公式如下：

$\mathbf{x}^{\prime}_i = \mathbf{\Theta}^{\top} \sum_{j \in \mathcal{N}(v) \cup \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j \hat{d}_i}} \mathbf{x}_j$

其中：$e_{j,i}$为边权，$\frac{e_{j,i}}{\sqrt{\hat{d}_j \hat{d}_i}}$为归一化， $\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}$

矩阵形式为：

$\mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta}$

其中：$\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}$即邻接矩阵$\mathbf{A}$+单位阵$\mathbf{I}$。$\mathbf{\hat{D}}：\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}$即对角的节点度数矩阵，$\mathbf{\Theta}$为可学习参数

