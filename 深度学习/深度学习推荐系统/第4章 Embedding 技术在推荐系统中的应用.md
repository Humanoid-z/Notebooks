# 1 什么是Embedding

Embedding 就是用一个低维稠密的向量“表示”一个对象。“表示”意味着Embedding向量能够表达相应对象的某些特征，同时向量之间的距离反映了对象之间的相似性。

## 1.1 Embedding 技术对于深度学习推荐系统的重要性

1. 推荐场景中大量使用one-hot 编码对类别、id 型特征进行编码，导致样本特征向量极度稀疏，而深度学习的结构特点使其不利于稀疏特征向量的处理，因此几乎所有深度学习推荐模型都会由Embedding 层负责将高维稀疏特征向量转换成稠密低维特征向量。因此， 掌握各类Embedding 技术是构建深度学习推荐模型的基础性操作。
2. Embedding 本身就是极其重要的特征向量。相比MF 等传统方法产生的特征向量， Embedding 的表达能力更强，特别是Graph Embedding 技术被提出后，Embedding 几乎可以引人任何信息进行编码，使其本身就包含大量有价值的信息。在此基础上， Embedding 向量往往会与其他推荐系统特征连接后一同输入后续深度学习网络进行训练。
3. Embedding 对物品、用户相似度的计算是常用的推荐系统召回层技术。在局部敏感哈希（ Locality-Sensitive Hashing ）等快速最近邻搜索技术应用于推荐系统后， Embedding 更适用于对海量备选物品进行快速“初筛”，过滤出几百到几千量级的物品交由深度学习网络进行“精排” 。

# 2 Word2vec——经典的Embedding 方法

## 2.1什么是Word2vec

 Word2vec 是一个生成对“词”的向量表达的模型。

为了训练Word2vec 模型， 需要准备由一组句子组成的语料库。假设其中一个长度为T的句子为$w_1,w_1,...,w_T$，假定每个词都跟其相邻的词的关系最密切，即每个词都是由相邻的词决定的（图中CBOW 模型的主要原理），或者每个词都决定了相邻的词（图中Skip-gram 模型的主要原理）。如图所示， CBOW模型的输入是$\omega_t$周边的词，预测的输出是$\omega_t$，而Skip-gram 则相反。经验上讲，Skip-gram 的效果较好。本节以Skip-gram 为框架讲解Word2vec 模型的细节。

![image-20220705154136343](https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/CBOW_Skip-gram.png)

## 2.2 Word2vec 模型的训练过程

为了基于语料库生成模型的训练样本，选取一个长度为2c+1（目标词前后各
选c 个词）的滑动窗口， 从语料库中抽取一个句子，将滑动窗口由左至右滑动，
每移动一次，窗口中的词组就形成了一个训练样本。

有了训练样本，就可以着手定义优化目标了。既然每个词$w_t$都决定了相邻
词$w_{t+j}$勺， 基于极大似然估计的方法， 希望所有样本的条件概率$p(w_{t+j}|w_t)$之积最大，这里使用对数概率。因此， Word2vec 的目标函数如下所示。

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/Word2vec_optim.png" alt="image-20220705162150784" style="zoom:50%;" />

接下来的核心问题是如何定义$p(w_{t+j}|w_t)$ ，作为一个多分类问题， 最直接的方法是使用softmax 函数。Word2vec 的“愿景”是希望用一个向量$v_w$表示词w,用词之间的内积距离$v_i^Tv_j$表示语义的接近程度，条件概率$p(w_{t+j}|w_t)$的定义如下所示，其中$w_O$代表$w_{t+j}$勺，被称为输出词；$w_I$代表$w_t$，被称为输入词。

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/Word2vec_softmax.png" alt="image-20220706155225105" style="zoom:50%;" />