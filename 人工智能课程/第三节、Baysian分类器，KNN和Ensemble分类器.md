# 贝叶斯分类器

- 分类问题的概率框架



- 条件概率：<img src="D:\TyporaData\第三节、Baysian分类器，KNN和Ensemble分类器\image-20200809142527541.png" alt="image-20200809142527541" style="zoom:50%;" />

- 贝叶斯定理：<img src="D:\TyporaData\第三节、Baysian分类器，KNN和Ensemble分类器\image-20200809142619504.png" alt="image-20200809142619504" style="zoom:50%;" />

- 将每个属性和类标签看作随机变量
- 给定一个具有属性(A1, A2，…，An)的记录
  - 目标是预测类C
  - 具体地说，我们想找到最大后验概率的C值(C| A1, A2，…，An)

方法:

利用贝叶斯定理计算所有C值的后验概率P(C | A1, A2，…，An)

<img src="D:\TyporaData\第三节、Baysian分类器，KNN和Ensemble分类器\image-20200809142951175.png" alt="image-20200809142951175" style="zoom: 67%;" />

- 找到使P(C | A1, A2，…，An)最大化的C值=找到试P(A1, A2, …, An|C) P(C)最大化的C的值


P(A1, A2, …, An |Cj) = P(A1| Cj) P(A2| A1,Cj)… P(An|P(A1, A2, …, An-1,Cj)

假设在给定类C时，属性之间是独立的:

P(A1, A2, …, An |Cj) = P(A1| Cj) P(A2| Cj)… P(An| Cj)

如果P(X|Cj)P(Cj)最大从而使P(Cj|X)最大，则新点X被分类为C~j~，P(X|Cj)计算如下：<img src="D:\TyporaData\第三节、Baysian分类器，KNN和Ensemble分类器\image-20200815181403655.png" alt="image-20200815181403655" style="zoom:67%;" />

## 如何从数据中估计概率?

对于连续属性:

- 将范围离散到箱子中
  - 每个容器有一个序号属性
  - 违反独立假设
- 双向拆分:(A < v)或(A > v)
  - 只选择两个分割中的一个作为新属性

- 概率密度估计:
  - 假设属性服从正态分布
  - 使用数据估计分布参数(例如，平均值和标准差)
  - 一旦概率分布已知，可以用它来估计条件概率P(Ai|c)

对于(Ai,cj)对，高斯分布：<img src="D:\TyporaData\第三节、Baysian分类器，KNN和Ensemble分类器\image-20200815183856428.png" alt="image-20200815183856428" style="zoom:50%;" />

### 朴素贝叶斯分类器的例子

提供一个测试记录：X=(Refund=No,Married,Income=120k)

<img src="D:\TyporaData\第三节、Baysian分类器，KNN和Ensemble分类器\image-20200815184154792.png" alt="image-20200815184154792" style="zoom:50%;" />

P(X|Class=No) = P(Refund=No|Class=No)× P(Married| Class=No)× P(Income=120K| Class=No)
	= 4/7 × 4/7 × 0.0072 = 0.0024
P(X|Class=Yes) = P(Refund=No| Class=Yes)× P(Married| Class=Yes)× P(Income=120K| Class=Yes)
	= 1 × 0 × 1.2 × 10-9 = 0

显然P(X|No)P(No) > P(X|Yes)P(Yes)
则P(No|X) > P(Yes|X)
=> Class = No

- 概率估计<img src="D:\TyporaData\第三节、Baysian分类器，KNN和Ensemble分类器\image-20200815184510240.png" alt="image-20200815184510240" style="zoom: 45%;" />

c:类数 p:先验概率 m:参数

### 朴素贝叶斯分类器的优缺点

优点

1. 对孤立噪声点具有鲁棒性
2. 通过在概率估计计算期间忽略实例来处理丢失的值
3. 对不相关属性健壮

缺点

1. 对于某些属性，独立性假设可能不成立
2. 使用其他技术，如贝叶斯信念网络(BBN)

# 基于实例的Classifers:KNN

- 存储训练记录(不需要训练明确的模型)
- 直接使用训练记录来预测未见案例的类标签

**Rote-learner**

- 记忆整个训练数据，只有当记录的属性与其中一个训练例子完全匹配时，才进行分类

**k-Nearest Neighbors (k-NN)**

- 使用k个“最近”点(最近的邻居)来执行分类

## Nearest-Neighbor Classifiers

<img src="D:\TyporaData\第三节、Baysian分类器，KNN和Ensemble分类器\image-20200815195005976.png" alt="image-20200815195005976" style="zoom:33%;" align="left" />

需要三件事

- 存储的训练记录
- 距离度量来计算记录之间的距离
- k的值，要检索的最近邻居的数量

分类未知的记录:

- 计算到其他训练记录的距离
- 识别k个最近邻居
- 使用最近邻居的类标签来确定未知记录的类标签(如多数票)

最近邻的定义：记录x的最近邻是距离x k个最小的数据点

**选择k的值:**

- 如果k太小，对噪声点敏感
- 如果k太大，邻域可能包括来自其他类的点

**计算两点p与q之间的距离:**

Euclidean distance:<img src="D:\TyporaData\第三节、Baysian分类器，KNN和Ensemble分类器\image-20200815235711632.png" alt="image-20200815235711632" style="zoom:33%;" />

Majority voting:<img src="D:\TyporaData\第三节、Baysian分类器，KNN和Ensemble分类器\image-20200816000059293.png" alt="image-20200816000059293" style="zoom:33%;" />

**从最近邻列表中确定类**

- 取k个最近邻中类标签的多数票
- 根据距离来权衡投票

权重因子w = 1/d^2^

<img src="D:\TyporaData\第三节、Baysian分类器，KNN和Ensemble分类器\image-20200815235926748.png" alt="image-20200815235926748" style="zoom: 50%;" align="left"/>

**扩展问题**

属性可能必须进行缩放，以防止距离度量被某个属性所控制

例如：

一个人的身高可能从1.5米到1.8米不等

一个人的体重从90磅到300磅不等

个人收入从1万美元到100万美元不等

将给定属性的整个值集映射到一组新替换值集的函数，这样每个旧值都可以用其中一个新值标识

- 简单的数学函数:x^k^, log(x), e^x^, |x|, 1/x, sin x
- 归一化(或标准)

### Normalization

#### **Min-max normalization:**

[minA, maxA] →[new_minA, new_maxA]

<img src="D:\TyporaData\第三节、Baysian分类器，KNN和Ensemble分类器\image-20200816001600392.png" alt="image-20200816001600392" style="zoom: 50%;" align="left"/>

例:

收入范围[$12,000，$98,000]归一化到[0.0,1.0]。然后73000美元映射到

<img src="D:\TyporaData\第三节、Baysian分类器，KNN和Ensemble分类器\image-20200816001710244.png" alt="image-20200816001710244" style="zoom:50%;"  align="left"/>

#### **Z-score normalization**

(μA: mean, σA: standard deviation):
$$
v'=\frac{v-μ_A}{σ_A}
$$
例如:考虑一个值v=73,000，μ~a~ = 54,000，而σ~a~ = 16,000。然后$v'=\frac{73,000-54,000}{16,000}=1.225$

#### Normalization by Decimal Scaling

$$
v'=\frac{v}{10^j}
$$

其中，j是最小的整数，使Max(|v'|) < 1

### 优缺点

优点：

- 容易实现
- 增量添加的训练数据琐碎

缺点：

- knn分类器是lazylearner，不显式地建立模型。当对一个测试/未知记录进行分类时，这可能比渴望学习的人(比如决策树)花费更多。
- 与决策树试图找到适合整个输入空间的全局模型不同，最近邻分类器基于局部信息进行预测，而局部信息更容易受到噪声的影响。

# 集成分类器（Ensemble Classifiers）

- 根据训练数据构造一组分类器
- 通过聚合多个分类器做出的预测来预测以前未看到的记录的类标签

假设:

单个分类器(选民)可能很糟糕(愚蠢)，但总体(选民)通常可以正确地分类(决定)。

## 集成方法的例子

- Bagging
- Boosting

### Bagging

放回抽样

在每个引导样本上构建分类器

每个样本有(1 -1 /N)^N^被选中的概率

对于较大的N，这个值趋向于0。63

**优点**

- 减少差异，提高稳定性(对噪声的容忍)
- 可以并行

**缺点**

- 降低了稳定分类器的准确性，因为样本大小减少了36%!

### Boosting

- 一个迭代的过程，以适应改变训练数据的分布，更多地关注以前错误分类的记录
  - 最初，给所有N条记录分配相同的权值
  - 与装袋不同，抽样权重可能会在推进回合结束时发生变化

- 被错误分类的记录将增加其权重
- 正确分类的记录将减少其权重

### Example: AdaBoost

<img src="D:\TyporaData\第三节、Baysian分类器，KNN和Ensemble分类器\image-20200816134600751.png" alt="image-20200816134600751" style="zoom:67%;" />

基本分类器:C1, C2，…，CT

分类器Ci错误率:

<img src="D:\TyporaData\第三节、Baysian分类器，KNN和Ensemble分类器\image-20200816134040650.png" alt="image-20200816134040650" style="zoom:50%;" />

分类器Ci的重要性:

<img src="D:\TyporaData\第三节、Baysian分类器，KNN和Ensemble分类器\image-20200816134102537.png" alt="image-20200816134102537" style="zoom:50%;" />

权重更新:

<img src="D:\TyporaData\第三节、Baysian分类器，KNN和Ensemble分类器\image-20200816134134186.png" alt="image-20200816134134186" style="zoom:50%;" />

Z~j~是归一化因子

如果任何中间轮产生的错误率超过50%，则权值恢复到1/n，并重复采样过程

最终分类器:

<img src="D:\TyporaData\第三节、Baysian分类器，KNN和Ensemble分类器\image-20200816134316417.png" alt="image-20200816134316417" style="zoom:50%;" />

**说明AdaBoost**

![image-20200816140125031](D:\TyporaData\第三节、Baysian分类器，KNN和Ensemble分类器\image-20200816140125031.png)

P(A=1|+)=0.5

P(B=1|+)=0.5

P(C=1|+)=1

P(A=1|-)=1/3

P(B=1|-)=1/3

P(C=1|-)=1/3

P(+)=0.4

P(-)=0.6



X=(A=1,B=1,C=1)

P(X|Class=+)=0.5*0.5\*1=0.25

P(X|Class=-)=1/3*1/3\*1/3=1/27

P(X|Class=+)P(+)=0.1

P(X|Class=-)P(-)≈0.022

∵P(X|Class=+)P(+)>P(X|Class=-)P(-)

Class=+

<img src="D:\TyporaData\第三节、Baysian分类器，KNN和Ensemble分类器\image-20200816151014835.png" alt="image-20200816151014835" style="zoom: 80%;" />

a:-

b+