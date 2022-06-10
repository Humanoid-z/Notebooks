[TOC]

本章仅为理解attention及transformer，有所取舍

# 一、循环神经网络

## 1.1 序列模型

序列sequence指语音数据、文本数据、视频数据等一系列具有连续关系的数据。序列问题有很多不同类型，输入输出是不定长的

## 1.2 符号约定

- 输入数据：$x^{<1>}$,$x^{<2>}$,$x^{<3>}$,...,$x^{<t>}$      $t$索引序列中的位置
- 输出数据：$y^{<1>}$,$y^{<2>}$,$y^{<3>}$,...,$y^{<t>}$        $t$索引序列中的位置
- $T_{x}$：输入序列的长度
- $T_{y}$：输出序列的长度
- 第$i$个训练样本中第$t$个元素:$x^{\left(i \right) <t>}$
- 第$i$个训练样本中第$t$个输出:$y^{\left(i \right) <t>}$
- 第$i$个训练样本的输入序列长度：$T_{x}^{(i)}$
- 第$i$个训练样本的输出序列长度：$T_{y}^{(i)}$

对于文本序列任务，需要建立词典，根据词典将对应单词表示成**one-hot**向量，$x^{<t>}$指代句子里任意词对应的**one-hot**向量。对于词典外的单词使用\<UNK>标记

## 1.3 循环神经网络模型

对于序列问题如何建立神经网络模型学习$X$到$Y$的映射，一种方法是使用标准神经网络：

对于人名识别问题，假如有9个输入单词，把9个**one-hot**向量输入到一个标准神经网络中，经过一些隐藏层，最终输出9个值为0或1的项

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/aa.png" style="zoom:67%;" />

这个方法有两个缺陷：

1. 难以处理输入数据长度$T_{x}$变化的情况
2. 此种神经网络结构不共享从文本的不同位置上学到的特征。

此外对于文字识别问题庞大的输入层（每个单词都是词库大小维度的**one-hot**向量），权重矩阵会拥有巨量的参数

而循环神经网络没有上述的两个问题。当$T_{x}$=$T_{y}$的情况下，循环神经网络的处理方式为：一个时间步t只处理一个词$x^{<t>}$，当在时间步t+1输入$x^{<t+1>}$时，还输入来自上个时间步t的激活值$a^{<t>}$，根据当前和过去的信息输出${\hat{y}}^{<t+1 >}$。在零时刻需要构造一个激活值$a^{<0>}$，通常为零向量。

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/rnn1.png" style="zoom:50%;" />

循环神经网络从左向右扫描数据，同时每个时间步的参数也是共享的

用$W_{\text{ax}}$来表示从$x^{<t>}$到隐藏层的一系列参数，每个时间步使用的都是相同的参数$W_{\text{ax}}$。而激活值也就是水平联系是由参数$W_{aa}$决定的，每一个时间步都使用相同的参数$W_{aa}$，同样的输出结果由$W_{\text{ya}}$决定。

前向传播公式如下：

$a^{< t >} = g_{1}(W_{aa}a^{< t - 1 >} + W_{ax}x^{< t >} + b_{a})$或简化成$a^{<t>} =g(W_{a}\left\lbrack a^{< t-1 >},x^{<t>} \right\rbrack +b_{a})$

$\hat y^{< t >} = g_{2}(W_{{ya}}a^{< t >} + b_{y})$

**RNN**前向传播示意图：

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/rnn_forward.png" alt="nn-" style="zoom: 80%;" />

循环神经网络的激活函数通常是**tanh**

此种循环神经网络的一个缺点是它只使用了每个输入之前的信息来做出预测，后节的双向循环神经网络（**BRNN**）能够解决这个问题

## 1.4 循环神经网络的反向传播

损失函数：$L^{<t>}( \hat y^{<t>},y^{<t>}) = - y^{<t>}\log\hat  y^{<t>}-( 1- y^{<t>})log(1-\hat y^{<t>})$

整个序列的损失函数：$L(\hat y,y) = \ \sum_{t = 1}^{T_{x}}{L^{< t >}(\hat  y^{< t >},y^{< t >})}$

**RNN**反向传播示意图：

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/rnn_backward.png" alt="nn_cell_backpro" style="zoom:80%;" />

反向传播需要按从后往前的时刻计算

## 1.5 不同类型的循环神经网络

1. $T_{x}>1,T_{y}=1$的多对一网络

   <img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/many2one_rnn.png" alt="image-20220323220706289" style="zoom:67%;" />

2. $T_{x}=1,T_{y}>1$的一对多网络

   <img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/one2many_rnn.png" alt="image-20220323220758319" style="zoom:67%;" />

3. $T_{x}≠T_{y}$的多对多网络（编码器-解码器）

   <img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/many2many_rnn.png" alt="image-20220323220830626" style="zoom:67%;" />

## 1.6语言模型和序列生成

语言模型就是根据某几个单词预测下一个单词，选择概率高的为最终结果。

为了训练RNN模型，需要包括很大语料库的训练集。

将每个单词都转成one-hot向量，包括结尾标记和标点符号、未见单词，作为输入。

在训练时，将$x^{<t>}$设为$y^{<t-1>}$，第一个时间步的输入是零向量，$a^{<0>}$按照惯例也设为0向量，通过softmax计算$\hat y^{<1>}$，以后每一步的输入为上一步对应的标签$y^{<t-1>}$，输出下一个单词的概率。对所有输出交叉熵求和，再反向传播。将输出相乘得到整个句子的概率

## 1.7 新序列采样

在训练一个序列模型之后，为了了解到这个模型学到了什么，一种非正式的方法就是进行一次新序列采样。

一个序列模型模拟了任意特定单词序列的概率，所要做的就是对这些概率分布进行采样来生成一个新的单词序列。

首先对模型生成的第一个词进行采样，输入$x^{<1>} =0$，$a^{<0>} =0$，如果输出是经过**softmax**层后得到的概率，则根据这个**softmax**的分布进行随机采样,得到$\hat y^{<1>}$。然后继续下一个时间步,把刚刚采样得到的$\hat y^{<1>}$作为$x^{<2>}$，得到结果$\hat y^{<2>}$，然后再次用这个采样函数来对$\hat y^{<2>}$进行采样。可以一直进行采样直到得到**EOS**标识或者达到所设定的时间步

## 1.8 循环神经网络的梯度消失

RNN一个最大的缺陷就是**梯度消失与梯度爆炸问题，** 由于这一缺陷，使得RNN在长文本中难以训练， 这才诞生了LSTM及各种变体。

参考[RNN梯度消失和爆炸的原因](https://zhuanlan.zhihu.com/p/28687529)

下图给出RNN结构：

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/rnn_1.jpeg" alt="img" style="zoom:67%;" />

假设处理$T_{x}=3$的时间序列，神经元没有激活函数，则RNN最简单的前向传播过程如下：

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/forward_formula.png" alt="image-20220324144538370" style="zoom:50%;" />

假设在t=3时刻，损失函数为$L_3$，进行反向传播时：

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/backward_t3.png" alt="image-20220324145415244" style="zoom: 50%;" />

可以看出对$W_o$求偏导并没有长期依赖，但是对于$W_x,W_s$会随着时间序列的拉长而产生梯度消失和梯度爆炸问题。

可以得出任意时刻对$W_x,W_s$求偏导的公式：

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/20220324150233.png" alt="image-20220324150233794" style="zoom:50%;" />

导致梯度消失和爆炸的就在于$\prod_{j=k+1}^{t}\frac{\delta S_j }{\delta S_{j-1}} $,而加上激活函数后的S的表达式为：$S_j=tanh(W_xX_j+W_SS_{j-1}+b_1)$，则有：$\prod_{j=k+1}^{t}\frac{\delta S_j }{\delta S_{j-1}}= \prod_{j=k+1}^{t}tanh’*W_s$

由于$tanh’$总是小于1,如果$W_s$也是一个大于0小于1的值， 那么随着t的增大， 越早时间步偏导项的值越来越趋近于0，导致了梯度消失问题，从最近的几个时间中传递而来的梯度占了绝大部分。如果$W_s$很大，也会产生了梯度爆炸。

loss对时间步j的梯度值反映了时间步j对最终输出$y^{<t>}$的影响程度。j对最终输出$y^{<t>}$的影响程度越大，loss对时间步j的梯度值也就越大。如果loss对时间步j的梯度值趋于0，说明j对最终输出$y^{<t>}$没影响，就是所谓长期遗忘问题。

**RNN中的梯度消失不是指损失对参数的总梯度消失了，而是RNN中对较远时间步的梯度消失了**。GRU(门控循环单元网络)和LSTM(长短期记忆网络)能有效解决梯度消失的问题

针对梯度爆炸问题，可以采用梯度削减：

设置一个clip_gradient作为梯度阈值，求出各个梯度，求出这些梯度的L2范数，如果L2范数大于设置好的clip_gradient，则求clip_gradient除以L2范数，然后把除好的结果乘上原来的梯度完成更新。当梯度很大的时候，作为分母的结果就会很小，那么乘上原来的梯度，整个值就会变小，从而可以有效地控制梯度的范围。

## 1.9 门控循环单元(Gated Recurrent Unit（**GRU**）)

GRU改变了RNN的隐藏层，使其可以更好地捕捉深层连接，并改善了梯度消失问题。

GRU的输入输出结构与普通的RNN是一样的。结合$x^t$和$a^{t-1}$,输出$\hat y^t$和传递给下一个节点$a^t$

GRU的关键思想之一在于设有两个"门"：$\Gamma_{r}$和$\Gamma_{u}$。$r$代表重置门；$u$代表更新门，都是一个0到1之间的值。

$\Gamma_{r}= \sigma(W_{r}\left\lbrack c^{<t-1>},x^{<t>} \right\rbrack + b_{r})$

$\Gamma_{u}=\sigma(W_{u}\left\lbrack c^{<t-1>},x^{<t>} \right\rbrack +b_{u})$

**GRU**有新的变量称为$c$，即记忆细胞。在时间$t$处，有记忆细胞$c^{<t>}$，**GRU**输出的激活值$a^{<t>}$=$c^{<t>}$。在每个时间步，计算一个更新值${\tilde{c}}^{}$：

${\tilde{c}}^{<t>} =tanh(W_{c}\left\lbrack \Gamma_{r}*c^{<t-1>},x^{<t>} \right\rbrack +b_{c})$

$\Gamma_{r}$用于控制$c^{t-1}$对当前词$x^{t}$的影响，如果$c^{t-1}$对$x^{t}$不重要，即从当前词$x^{t}$开始表述了新的意思，与上文无关，则$\Gamma_{r}$可以置零，使$c^{t-1}$对$x^{t}$不产生影响。$\Gamma_{u}$用于决定是否忽略当前词$x^{t}$，$\Gamma_{u}$判断当前词对整体意思的表达是否重要。

$c^{<t>} = \Gamma_{u}*{\tilde{c}}^{<t>} +\left( 1- \Gamma_{u} \right)*c^{<t-1>}$

计算更新值${\tilde{c}}^{t}$后,$\Gamma_{u}$决定是否要真的更新它。如$\Gamma_{u} =1$则$c^{<t>} = {\tilde{c}}^{<t>}$；如$\Gamma_{u}= 0$，则$c^{<t>} =c^{<t-1>}$，这种情况使得梯度有效地进行反向传播。

一张简化的电路图可作补充理解

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/GRU.png" alt="img" style="zoom:67%;" />



## 1.10 长短期记忆（**LSTM**（long short term memory）unit）

**LSTM**即长短时记忆网络，能够在序列中学习非常深的连接，甚至比**GRU**更加有效。原始 RNN 的隐藏层只有一个状态，即a，它对于短期的输入非常敏感。**LSTM**增加一个状态c来保存长期的状态。

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/lstm.png" alt="ST" style="zoom:67%;" />

候选值${\tilde{c}}^{<t>}$：${\tilde{c}}^{<t>} = tanh(W_{c}\left\lbrack a^{<t-1>},x^{<t>} \right\rbrack +b_{c}$

在**LSTM**中不再有$a^{<t>} = c^{<t>}$的情况,也不再使用$\Gamma_{r}$,此外，LSTM有3个“门”

更新门$\Gamma_{u}$：$\Gamma_{u}= \sigma(W_{u}\left\lbrack a^{<t-1>},x^{<t>} \right\rbrack +b_{u})$

遗忘门$\Gamma_{f}$：$\Gamma_{f} =\sigma(W_{f}\left\lbrack a^{<t-1>},x^{<t>} \right\rbrack +b_{f})$

输出门$\Gamma_{o} $：$\Gamma_{o} =\sigma(W_{o}\left\lbrack a^{<t-1>},x^{<t>} \right\rbrack +>b_{o})$

更新值$c^{<t>}$：$c^{<t>} =\Gamma_{u}*{\tilde{c}}^{<t>} + \Gamma_{f}*c^{<t-1>}$

输出值$a^{<t>}$：$a^{<t>} = \Gamma_{o}*c^{<t>}$

**LSTM**使用$\Gamma_{f}$将细胞状态中的信息选择性的遗忘；$\Gamma_{u}$将新的信息选择性的记录到细胞状态中 ；$\Gamma_{o} $选择性地把细胞状态的信息保存到隐层

**相较于LSTM,GRU的优势**：GRU的参数量少，减少过拟合的风险。LSTM的参数量是Navie RNN的4倍，参数量过多就会存在过拟合的风险，GRU只使用两个门控开关，达到了和LSTM接近的结果。其参数量是Navie RNN的三倍

[LSTM,GRU为什么可以缓解梯度消失问题?](https://zhuanlan.zhihu.com/p/149819433)

RNN的梯度消失或爆炸问题关键在于$\frac{\delta S_j }{\delta S_{j-1}} $要么始终大于1，要么始终在[0,1]范围内。但在LSTM中，有：

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/20220325153635.png" alt="image-20220325153635649" style="zoom:67%;" />

求导得到：<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/lstm_f2.png" alt="image-20220325153815414" style="zoom:67%;" />

一方面有$\Gamma_{f}$在导数里，另外剩下的几项是相加关系，时间步t扩展到无限时，数学上不能保证递归梯度收敛于0或无穷大。

## 1.11 双向循环神经网络

双向**RNN**模型可以在序列的某点处不仅可以获取之前的信息，还可以获取未来的信息。

在该模型中，先计算前向的${\overrightarrow{a}}^{<1>}$，接着是${\overrightarrow{a}}^{<2>}，{\overrightarrow{a}}^{<3>}，...，{\overrightarrow{a}}^{<t>}$,反向序列从计算${\overleftarrow{a}}^{<t>}$开始，反向进行计算直到${\overleftarrow{a}}^{<1>}$，把所有激活值都计算完就可以计算预测结果$\hat y^{<t>}$：

$\hat y^{<t>} =g(W_{g}\left\lbrack {\overrightarrow{a}}^{< t >},{\overleftarrow{a}}^{<t>} \right\rbrack +b_{y})$

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/brnn.png" style="zoom: 67%;" />

## 1.12 深层循环神经网络

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/deepRnn.png" style="zoom:67%;" />

$a^{\lbrack l\rbrack <t>}$表示第t个时间步第l层的激活值

由于时间的维度，RNN的深度不会特别深，不像卷积神经网络一样有大量的隐含层。

# 二、自然语言处理与词嵌入

## 2.1词嵌入(**word embedding**)

之前的章节中一直用**one-hot**向量来表示词，这种方法的一大缺点是每个词之间的关系或者说词本身的特征无法从中表现，使得算法对相关词的泛化能力不强。

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/onehot.png" alt="image-20220327203730485" style="zoom:50%;" />

而词嵌入可以让算法自动理解词与词之间的联系，比如男人对女人，国王对王后。

词嵌入的目标是特征化地表示每个词，假设一个词有300个不同的特征，如**Size**（**尺寸大小**），**Cost**（**花费多少**），这个东西是不是**alive**（**活的**），是不是一个**Action**（**动作**），或者是不是**Noun**（**名词**）或者是不是**Verb**（**动词**）等等，能够用300维的向量来表示，从而能够让模型发现某些词是否在特征空间更加靠近，或者有着对应或相反的关系，让模型更容易进行特征抽取。此外还能提高模型的泛化能力，可以理解为模型从机械式地记忆词句组合到根据词义进行搭配，有效提高了在小训练样本情况下的能力。

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/wordembedding.png" alt="image-20220327204500148" style="zoom:50%;" />

余弦相似度是一种常用的比较两个词的两个嵌入向量之间相似度的函数,给定两个向量$u$和$v$，余弦相似度定义如下： 
${CosineSimilarity(u, v)} = \frac {u . v} {||u||_2 ||v||_2} = cos(\theta) \tag{1}$

其中 $u.v$ 是两个向量的点积（或内积），$||u||_2$是向量$u$的范数（或长度），并且 $\theta$ 是向量$u$和$v$之间的角度。

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/CosineSimilarity.png" alt="osine_si" style="zoom: 33%;" />

## 2.2嵌入矩阵（Embedding Matrix）

学习词嵌入实际上是学习一个用来表示整个词汇表的嵌入矩阵。假设词汇表含有10,000个单词，设特征维度数为3000。所要做的就是学习一个嵌入矩阵$E$，它将是一个300×10,000的矩阵，**某个词的嵌入向量就是嵌入矩阵的它所对应的那一列**。通常为了表示方便，也相当于$E×O_{w}$(E为嵌入矩阵，$O_{w}$为单词w对应的**one-hot**向量)，但由于要进行很多乘法运算，在实现中会单独查找矩阵$E$的某列。

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/embeddingMatrix.png" style="zoom:67%;" />

## 2.3学习词嵌入（Learning Word Embeddings）

本节介绍一些学习词嵌入的算法

1. 建立神经网络，输入待预测单词(目标词)的前**N**个单词，预测序列中的下一个单词

   <img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/learningWordEmbedding1.png" style="zoom:50%;" />

   随机生成一个参数矩阵$E$，要预测单词的前**N**(超参数)个单词的onehot向量通过与$E$相乘再送入神经网络通过**softmax**分类器预测单词，$E$会在反向传播中得到学习。

2. 把目标词左右各N个词作为上下文，预测中间的词

3. 把目标词邻近的单个词作为上下文（**Skip-Gram**模型）

   在**Skip-Gram**模型中，需要抽取上下文和目标词配对，来构造一个监督学习问题。随机选一个词作为上下文词，然后在一定词距内随机选另一个词，作为目标词。嵌入矩阵$E$乘以向量$O_{c}$(上下文的**one-hot**向量)得到上下文的嵌入向量$e_{c}$，送入**softmax**单元预测不同目标词的概率：

   $Softmax:p\left( t \middle| c \right) = \frac{e^{\theta_{t}^{T}e_{c}}}{\sum_{j = 1}^{10,000}e^{\theta_{j}^{T}e_{c}}}$

   $\theta_{t}$是一个与输出$t$有关的**softmax**单元参数

4. **CBOW**连续词袋模型（**Continuous Bag-Of-Words Model**）

   获得中间词两边的的上下文，然后用周围的词去预测中间的词

   **CBOW**是从原始语句推测目标字词；而**Skip-Gram**正好相反，是从目标字词推测出原始语句。**CBOW**对小型数据库比较合适，而**Skip-Gram**在大型语料中表现更好。

# 三、序列模型和注意力机制

## 3.1 seq2seq模型

在机器翻译任务中，输入序列和输出序列往往是不定长的，seq2seq模型能够有效应对此类问题。seq2seq是一个 Encoder-Decoder 结构的神经网络，它的输入是一个序列(Sequence)，输出也是一个序列(Sequence)，因此而得名“Seq2Seq”。在 Encoder 中，将可变长度的序列转变为固定长度的向量表达，Decoder 将这个固定长度的向量转换为可变长度的目标的信号序列。

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/seq2seq.webp" alt="img" style="zoom: 33%;" />

Encoder是一个**RNN**的结构，只是没有输出， **RNN**的单元可以是**GRU** 也可以是**LSTM**。Decoder网络类似于第一章的语言模型，不同在于语言模型总是以零向量开始，而**encoder**网络会计算出固定长度的向量来表示输入的句子，Decoder以这个向量开始，也可以称作条件语言模型，相比语言模型会输出任意句子的概率，这个模型会基于输入序列的条件给出输出$P(y|x)$。模型的任务就是找到合适的$y$值使得条件概率最大化。

由于RNN是一个时序的过程，每个时间步计算所有词在该位置的概率，一种想法是使用贪心搜索（每次找最大概率的词作为输出，同时作为下一个时间步输入），然而这个方法并不能找到全局最大的$P(y|x)$，例如：第一个时间步，单词A的概率$P(A|x)$=0.2,单词B的概率$P(B|x)$=0.1；第二个时间步在选择A的条件下单词C的概率最高，$P(C|x,A)$=0.1,在选择B的条件下单词D的概率最高,$P(D|x,B)$=0.8。显然$P(B,D|x)>P(A,C|x)$，然而贪心搜索只保留了前一个单词A，不会找到B,D这种选择。

虽然贪心搜索精度较低，但是对句子组合进行穷举又太过耗时，因此需要有一个有较高精度而复杂度不会过大算法，即束搜索（Beam Search）

## 3.3 束搜索（Beam Search）

束搜索是一种在贪心搜索和穷举法之间进行折中的算法。束搜索算法有一个参数**B**，即集束宽。

束搜索在一个时间步找前**B**个概率最大的单词，将其保存作为候选。第二步该算法会针对每个第一个单词考虑第二个单词是什么，如果词表大小为N，则会得到$B*N$个结果，束搜索考虑的是单词组合作为一个整体的概率，根据条件概率公式即$P(A,B|x)=P(A|x)*P(B|x,A)$，再次保留前**B**个概率最大的组合，然后进行第三步，以此类推。

束搜索的目的是最大化概率$P(y^{< 1 >}\ldots y^{< T_{y}>}|X)$，可以表示为：

$P(y^{<1>}|X) P(y^{< 2 >}|X,y^{< 1 >}) P(y^{< 3 >}|X,y^{< 1 >},y^{< 2>})…P(y^{< T_{y} >}|X,y^{<1 >},y^{<2 >}\ldots y^{< T_{y} - 1 >})$

由于这些概率值都是小于1的，通常远小于1，相乘会得到很小的数字，造成数值下溢（**numerical underflow**），导致电脑的浮点表示不能精确地储存，因此在实践中不会最大化概率的乘积，而是最大化概率的对数和。

此外对于目标函数$P(y^{< 1 >}\ldots y^{< T_{y}>}|X)$还可以进行改进，因为参照这个目标函数，翻译出一个长句的概率会很低，因为很多小于1的数字进行累乘会得到一个更小的概率值，这个目标函数可能不自然地倾向于简短的翻译结果。为此，结合求概率的对数和的改进，除以翻译结果的单词数量$T_{y}$进行归一化，取每个单词的概率对数值的平均了，这样很明显地减少了对输出长的结果的惩罚。在实践中通常在$T_{y}$上加上指数$a$，$a$可以等于0.7。如果$a$等于1，就相当于完全用长度来归一化，如果$a$等于0，就相当于完全没有归一化

束搜索的误差分析：如果束搜索算法输出了糟糕的翻译$\hat y$，同时有着人工翻译$y^*$，可以使用模型计算$P(y^*|x)$和$P(\hat y|x)$，如果$P(\hat y|x)<P(y^*|x)$则应该是束搜索出错，如果$P(\hat y|x)>P(y^*|x)$则应该是RNN模型出错。

## 3.4 Bleu 得分

定义序列中连续的n个词称作n元词组,$P_n$为n元词组这一项的**BLEU**得分($P$代表精确度)

**BLEU**得分被定义为，$exp (\frac{1}{N}\sum\limits_{n=1}^{N}{{{P}_{n}}})$,此外还会使用**BP** 惩罚因子（**the BP penalty**）来调整该项。

$P_n$的计算：对candidate中选出的每个n元词组w，设其在reference中出现次数的最大值为**Count_clip**，w在candidate中w出现次数为$n_w$，$P_n=\frac{\sum min(Count\_clip,n_w)}{\sum n_w}$

## 3.5 注意力模型Attention Model

注意力模型或者说注意力这种思想已经成为深度学习中最重要的思想之一。虽然这个模型源于机器翻译，但它也推广到了其他应用领域。

在机器翻译任务中，seq2seq模型的编码器是将不定长的输入序列转换为定长的语义向量c，在压缩过程中损失了部分信息，对应长句子的记忆比较困难。**注意力模型让一个神经网络一次更关注部分的输入句子。当它在生成句子的时候，更像人类翻译**。

Attention Model使用注意力权重来确保对哪些特征花多少注意力。由于计算输入序列特征和计算输出序列是不同步的，所以记$a^{<t'>}$为时间步$t'$上的特征向量，记$y^{<t>}$为时间步$t$上的输出。使用$a^{<t,t'>}$表示$y^{<t>}$应该花在$a^{<t'>}$上的注意力的数量。

$a^{<t,t'>}=\frac{exp(e^{<t,t'>})}{\sum_{t'= 1}^{T_x}exp(e^{<t,t'>})}$

$a^{<t,t'>}$其实是注意力得分$e^{<t,t'>}$经过**softmax**的结果，确保$a^{<t,t'>}$加起来等于1

t时刻的上下文$C^{<t>}=\sum_{t'= 1}^{T_x}a^{<t,t'>}a^{t'}$，RNN根据$C^{<t>}$和$y^{<t-1>}$输出$y^{<t>}$

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/attention.png" alt="image-20220329135951920" style="zoom: 33%;" />

而注意力得分$e^{<t,t'>}$由一个小的神经网络计算得出，输入分别是$s^{<t-1>}$(计算输出序列时神经网络在上个时间步的隐藏状态)和$a^{<t'>}$(t'时间步的输入特征)。以机器翻译为例神经网络会挖掘原文的单词$x^{t'}$和译文前一个单词$y^t$之间的关系，按照这个关系来分配$x^{t'}$特征在计算$y^{<t>}$时的权重。

下图为法文翻英文的注意力权重可视化

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/visualize_attention.png" alt="image-20220329141203522" style="zoom: 67%;" />