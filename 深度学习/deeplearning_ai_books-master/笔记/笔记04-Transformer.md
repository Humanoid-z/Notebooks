[TOC]

# 一、Attention

## 1.1心理学中的注意力提示

心理学家认为生物会基于**非自主性提示和自主性提示** 有选择地引导注意力的焦点。

非自主性提示是基于环境中物体的突出性和易见性。如下图中由于突出性的非自主性提示（红杯子），注意力不自主地指向了咖啡杯

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/20220329143149.svg"  style="zoom: 50%;" />

人希望读书时，依赖于任务的意志提示，注意力被自主引导到书上

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/20220329143259.svg" style="zoom:50%;" />



## 1.2注意力机制

卷积、全连接、池化层都只考虑非自主性提示，如max pooling抽取一定范围内最大的数据，注意力机制则考虑自主性提示。

- 自主性提示称为查询(query)

- 每个输入是一个值(value)和非自主性提示(key)的对，key和value可以相同也可以不相同。
- 给定任何查询，通过注意力池化层（attention pooling） 来有偏向性地选择某些输入。

下图展示了注意力机制通过注意力汇聚将查询和键结合在一起，实现对值的选择倾向

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/attention_pooling.svg" style="zoom:70%;" />



## 1.3注意力评分函数

下图说明了如何将注意力汇聚的输出计算成为值的加权和，其中a表示注意力评分函数。 由于注意力权重是概率分布， 因此加权和其本质上是加权平均值。

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/attention2.svg" alt="../_images/attention-output.svg" style="zoom: 67%;" />

注意力分数是query和key的相似度，注意力权重是分数的softmax结果。

用数学语言描述，假设有一个查询$\mathbf{q} \in \mathbb{R}^q$,和$m$个“键－值”对$(\mathbf{k}_1, \mathbf{v}_1), \ldots, (\mathbf{k}_m, \mathbf{v}_m)$，其中$\mathbf{v}_i \in \mathbb{R}^v$。注意力汇聚函数$f$就被表示成值的加权和：

$f(\mathbf{q}, (\mathbf{k}_1, \mathbf{v}_1), \ldots, (\mathbf{k}_m, \mathbf{v}_m)) = \sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i \in \mathbb{R}^v,$

其中查询$q$和键$k_i$的注意力权重（标量） 是通过注意力评分函数a 将两个向量映射成标量， 再经过softmax运算得到的：

$\alpha(\mathbf{q}, \mathbf{k}_i) = \mathrm{softmax}(a(\mathbf{q}, \mathbf{k}_i)) = \frac{\exp(a(\mathbf{q}, \mathbf{k}_i))}{\sum_{j=1}^m \exp(a(\mathbf{q}, \mathbf{k}_j))} \in \mathbb{R}$

选择不同的注意力评分函数$a$会导致不同的注意力汇聚操作。以下介绍两个流行的评分函数。

### 1.3.1加性注意力

当查询和键是**不同长度**的矢量时， 可以使用加性注意力作为评分函数。 给定查询$\mathbf{q} \in \mathbb{R}^q$和 键$\mathbf{k} \in \mathbb{R}^k$， 加性注意力（additive attention）的评分函数为

$a(\mathbf q, \mathbf k) = \mathbf w_v^\top \text{tanh}(\mathbf W_q\mathbf q + \mathbf W_k \mathbf k)$

其中可学习的参数是$\mathbf W_q\in\mathbb R^{h\times q}、\mathbf W_k\in\mathbb R^{h\times k}、\mathbf w_v\in\mathbb R^{h}$，相当于将查询和键连结起来后输入到一个多层感知机（MLP）中， 感知机包含一个隐藏层，其隐藏单元数是一个超参数h。

### 1.3.2缩放点积注意力

使用点积可以得到计算效率更高的评分函数， 但是点积操作要求查询和键具有相同的长度d。 假设查询和键的所有元素都是独立的随机变量， 并且都满足零均值和单位方差， 那么两个向量的点积的均值为0，方差为$d$。当 d 比较大时，方差会更大，softmax让大的数据更大，小的更小，从而让softmax 的预测值进入梯度比较小的区域。为防止softmax函数的梯度消失，确保无论向量长度如何，点积的方差在不考虑向量长度的情况下仍然是1， 将点积除以$\sqrt{d}$， 则缩放点积注意力（scaled dot-product attention）评分函数为：

$a(\mathbf q, \mathbf k) = \mathbf{q}^\top \mathbf{k}  /\sqrt{d}.$

从小批量的角度，基于n个查询和m个键－值对计算注意力，其中查询和键的长度为$d$，值的长度为$v$。 查询$\mathbf Q\in\mathbb R^{n\times d}$、 键$\mathbf K\in\mathbb R^{m\times d}$和值$\mathbf V\in\mathbb R^{m\times v}$的缩放点积注意力是：

$\mathrm{softmax}\left(\frac{\mathbf Q \mathbf K^\top }{\sqrt{d}}\right) \mathbf V \in \mathbb{R}^{n\times v}.$

### 1.3.3掩蔽softmax操作

在某些情况下，并非所有的值都应该被纳入到注意力汇聚中。 例如，为了高效处理小批量数据集， 某些文本序列被填充了没有意义的特殊词元使其具有相同的长度。 为了仅将有意义的词元作为值来获取注意力汇聚， 可以指定一个有效序列长度（即词元的个数）， 以便在计算softmax时过滤掉超出指定范围的位置。

## 1.4使用注意力机制的seq2seq

循环神经网络编码器将长度可变的序列转换为固定形状的上下文变量， 然后循环神经网络解码器根据生成的词元和上下文变量 按词元生成输出序列词元。 然而，即使并非所有输入词元都对解码某个词元都有用， 在每个解码步骤中仍使用编码相同的上下文变量。为了针对性地使用输入词元的特征，在seq2seq模型中加入注意力机制。

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/attention_seq2seq.svg" alt="../_images/seq2seq-attention-details.svg" style="zoom:67%;" />

在seq2seq模型中的注意力机制：

- 编码器对每个词的输出作为key和value（它们是相同的）
- 解码器的上一个时间步的输出是query
- 注意力的输出和下个词嵌入合并进入RNN

## 1.5自注意力(self-attention)与位置编码(positional encoding)

在NLP任务的背景下，将词元序列输入注意力池化中，**同一组词元同时充当查询、键和值**，这种结构称为自注意力（self-attention），下面给出其定义：

给定一个由词元组成的输入序列$\mathbf{x}_1, \ldots, \mathbf{x}_n$，其中任意$1 \leq i \leq n$。 该序列的自注意力输出为一个长度相同的序列$\mathbf{y}_1, \ldots, \mathbf{y}_n$，其中：

$\mathbf{y}_i = f(\mathbf{x}_i, (\mathbf{x}_1, \mathbf{x}_1), \ldots, (\mathbf{x}_n, \mathbf{x}_n)) \in \mathbb{R}^d$

### 1.5.1自注意力与CNN，RNN的对比

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/cnn_rnn_self-attention.svg" alt="../_images/cnn-rnn-self-attention.svg" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/compare_cnn_rnn_selfAtt.png" alt="image-20220331142321964" style="zoom:50%;" />

自注意力能够完全并行

### 1.5.2位置编码(positional encoding)

与CNN/RNN不同，自注意力没有记录位置信息。 为了使用序列的顺序信息，通过在输入表示中添加位置编码（positional encoding）来注入绝对的或相对的位置信息。 

假设输入表示$\mathbf{X} \in \mathbb{R}^{n \times d}$包含一个序列中n个词元的d维嵌入表示。 位置编码使用相同形状的位置嵌入矩阵$\mathbf{P} \in \mathbb{R}^{n \times d}$输出$\mathbf{X} + \mathbf{P}$， 下面给出一种相对位置编码方案，矩阵第i行、第2j列和2j+1列上的元素为：

$\begin{split}\begin{aligned} p_{i, 2j} &= \sin\left(\frac{i}{10000^{2j/d}}\right),\\p_{i, 2j+1} &= \cos\left(\frac{i}{10000^{2j/d}}\right).\end{aligned}\end{split}$

在位置嵌入矩阵P中， 行代表词元在序列中的位置，列代表每个字符embedding后的不同维度。下图的4条曲线分别对应第6~9列(不同列周期不同，相同列的不同行在同一个曲线上)。

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/positional_encoding.svg" alt="../_images/output_self-attention-and-positional-encoding_d76d5a_40_0.svg" style="zoom:67%;" />

从函数定义中可以得出，频率沿向量维度减小，在波长上形成了一个几何级数。可以认为每个时间步(词元)的d维嵌入表示都有着如下的位置编码

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/positional_encoding.png" alt="image-20220331150720557" style="zoom:50%;" />

(周期的范围是从2$\pi$到20000$\pi$，理论上对于序列长度在5000$\pi$≈15000个词以内的序列，只用看最高位偶数列(sin函数单调递增)就能知道该词元在序列中的绝对位置)

随着行的变化，不同列有着由快到慢的变化率，下图为最大长度为 50 的句子的 128 维位置编码。每一行代表嵌入向量

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/tri_encoding.png" alt="最大长度为 50 的句子的 128 维位置编码。每一行代表嵌入向量" style="zoom: 50%;" />

对比下图计算机使用的二进制编码，可以发现较高比特位的交替频率低于较低比特位， 与上图所示相似，只是位置编码通过使用三角函数在embeddiong维度上降低频率。 由于输出是浮点数，因此此类连续表示比二进制表示法更节省空间。

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/binary_encoding.png" alt="image-20220331151558189" style="zoom:50%;" />

此类编码的好处之一是无论词元的嵌入表示有多长(d维)都可以进行表示，方便位置嵌入矩阵$\mathbf{P}$直接与输入表示$\mathbf{X}$直接相加

除了捕获绝对位置信息之外，上述的位置编码还允许模型学习得到输入序列中相对位置信息。 这是因为对于任何确定的位置偏移δ，位置i+δ处的位置编码可以通过线性投影位置i处的位置编码来表示：

令$\omega_j = 1/10000^{2j/d}$， 对于任何确定的位置偏移δ，任何一对$(p_{i, 2j}, p_{i, 2j+1})$都可以线性投影到$(p_{i+\delta, 2j}, p_{i+\delta, 2j+1})$：

$\begin{split}\begin{aligned}
&\begin{bmatrix} \cos(\delta \omega_j) & \sin(\delta \omega_j) \\  -\sin(\delta \omega_j) & \cos(\delta \omega_j) \\ \end{bmatrix}
\begin{bmatrix} p_{i, 2j} \\  p_{i, 2j+1} \\ \end{bmatrix}\\
=&\begin{bmatrix} \cos(\delta \omega_j) \sin(i \omega_j) + \sin(\delta \omega_j) \cos(i \omega_j) \\  -\sin(\delta \omega_j) \sin(i \omega_j) + \cos(\delta \omega_j) \cos(i \omega_j) \\ \end{bmatrix}\\
=&\begin{bmatrix} \sin\left((i+\delta) \omega_j\right) \\  \cos\left((i+\delta) \omega_j\right) \\ \end{bmatrix}\\
=&
\begin{bmatrix} p_{i+\delta, 2j} \\  p_{i+\delta, 2j+1} \\ \end{bmatrix},
\end{aligned}\end{split}$

$2\times 2$投影矩阵不依赖于任何位置的索引$i$

[Transformer 架构：位置编码](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/#what-is-positional-encoding-and-why-do-we-need-it-in-the-first-place)

## 1.6 多头注意力multihead attention

在实践中，当给定相同的查询、键和值的集合时， 我们希望模型可以基于相同的注意力机制学习到不同的行为， 然后将不同的行为作为知识组合起来， 捕获序列内各种范围的依赖关系 （例如，短距离依赖和长距离依赖关系）。 

为此，多头注意力使用$h$个独立的注意力池化，通过合并各个头的输出，并且通过另一个可以学习的线性投影进行变换，得到最终输出。(类似于CNN使用多个卷积核试图提取图片的不同特征)

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/multi-head-attention.svg" alt="../_images/multi-head-attention.svg" style="zoom: 67%;" />

多头注意力模型的数学形式如下：

给定查询$\mathbf{q} \in \mathbb{R}^{d_q}$、 键$\mathbf{k} \in \mathbb{R}^{d_k}$和值$\mathbf{v} \in \mathbb{R}^{d_v}$， 可学习的参数包括 $\mathbf W_i^{(q)}\in\mathbb R^{p_q\times d_q}$、 $\mathbf W_i^{(k)}\in\mathbb R^{p_k\times d_k}$和 $\mathbf W_i^{(v)}\in\mathbb R^{p_v\times d_v}$,每个注意力头$i = 1, \ldots, h$计算方法为：

$\mathbf{h}_i = f(\mathbf W_i^{(q)}\mathbf q, \mathbf W_i^{(k)}\mathbf k,\mathbf W_i^{(v)}\mathbf v) \in \mathbb R^{p_v},$

多头注意力的输出需要经过另一个线性转换， 它对应着h个头连结后的结果，因此其可学习参数是$\mathbf W_o\in\mathbb R^{p_o\times h p_v}$：

$\begin{split}\mathbf W_o \begin{bmatrix}\mathbf h_1\\\vdots\\\mathbf h_h\end{bmatrix} \in \mathbb{R}^{p_o}.\end{split}$

# 二、Transformer

transformer模型完全基于注意力机制，没有任何卷积层或循环神经网络层。尽管transformer最初是应用于在文本数据上的序列到序列学习，但现在已经推广到各种现代的深度学习中，例如语言、视觉、语音和强化学习领域。

Transformer作为编码器－解码器架构的一个实例，其整体架构图如下：

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/Transformer_en.svg" alt="../_images/transformer.svg" style="zoom:67%;" />

从宏观角度来看，transformer的编码器是由多个相同的层叠加而成的，每个层都有两个子层（sublayer）。第一个子层是多头自注意力（multi-head self-attention）汇聚；第二个子层是基于位置的前馈网络（positionwise feed-forward network）。具体来说，在计算编码器的自注意力时，查询、键和值都来自前一个编码器层的输出。每个子层都采用了残差连接（residual connection）。在transformer中，对于序列中任何位置的任何输入$\mathbf{x}\in \mathbb{R}^d$，都要求满足$\mathrm{sublayer}(\mathbf{x}) \in \mathbb{R}^d$，以便残差连接满足$\mathbf{x} + \mathrm{sublayer}(\mathbf{x}) \in \mathbb{R}^d$。在残差连接的加法计算之后，紧接着应用层规范化（layer normalization）。因此，输入序列对应的每个位置，transformer编码器都将输出一个d维表示向量。

Transformer解码器也是由多个相同的层叠加而成的，并且层中使用了残差连接和层规范化。除了编码器中描述的两个子层之外，解码器还在这两个子层之间插入了第三个子层，称为编码器－解码器注意力（encoder-decoder attention）层。在编码器－解码器注意力中，查询来自前一个解码器层的输出，而键和值来自整个编码器的输出。在解码器自注意力中，查询、键和值都来自上一个解码器层的输出。但是，解码器中的每个位置只能考虑该位置之前的所有位置。这种*掩蔽*（masked）注意力保留了自回归（auto-regressive）属性，确保预测仅依赖于已生成的输出词元。

## 2.1 带掩码的多头注意力masked multi-head attention

[Transformer的mask](https://zhuanlan.zhihu.com/p/368592551)

Transformer的Decoder支持部分并行化训练，基于以下两个关键点：

1. **teacher force**
2. **masked self attention**

对于teacher force，它是指在每一轮预测时，不使用上一轮预测的输出，而强制使用正确的单词。原本的处理思路如下：

- 假设目标语句为：“I love China”
- 第一轮：给解码器模块输入“<start>” 和 编码器的输出结果，解码器输出“I”
- 第二轮：给解码器模块输入“<start> I” 和 编码器的输出结果，解码器输出“Iove”
- 第三轮：给解码器模块输入“<start> I love” 和 编码器的输出结果，解码器输出“China”
- 第四轮：给解码器模块输入“<start> I love China” 和 编码器的输出结果，解码器输出“<end>”，至此完成。

第二轮时，假设解码器没有正确预测出“Iove”，而是得到了“want”。如果没有采用teacher force，在第三轮时，解码器模块输入的就是“<start> I want”。如果采用了 teacher force，第三轮时，解码器模块输入的仍然是“<start> I love”。通过这样的方法可以有效的避免因中间预测错误而对后续序列的预测，从而加快训练速度。

Transformer采用这个方法，为并行化训练提供了可能，因为每个时刻的输入不再依赖上一时刻的输出，而是依赖正确的样本，而正确的样本在训练集中已经全量提供了。值得注意的一点是：**Decoder的并行化仅在训练阶段，在测试阶段，因为我们没有正确的目标语句，t时刻的输入必然依赖t-1时刻的输出。**

由于Transformer在训练时是将正确序列全部提供给decoder的self attention层，为避免看到t时刻以后的正确样本，确保预测仅依赖于已生成的输出词元，使用掩码(mask)将t时刻query与t时刻之后的key的联系摘除<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/masked_scores.jpeg" alt="img" style="zoom: 67%;" />

## 2.1 基于位置的前馈网络positionwise FNN

Position-wise FFN由两个全连接层组成。*(b:batch size;n:序列长度;d:词元feature长度)*FNN将输入形状由(b,n,d)变换成(bn,d)。输出形状由(bn,d)变换成(b,n,ffn_num_outputs)。由于n是不定长的，因此在展平时选择(bn,d)而非类似图像变成(c,hw)

等价于两层核窗口为1的一维卷积层(1×1卷积的作用：1.**降维或升维**2.**加入非线性**，提升网络的表达能力)。1×1卷积对每个像素融合其通道信息，而这里的FNN对每个词元融合其feature信息(d)，并且使用同一个多层感知机（MLP），这就是称前馈网络是*基于位置的*（positionwise）的原因。此外还将词元的特征维度由d变换成ffn_num_outputs。

~~~python
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
~~~

~~~python
ffn = PositionWiseFFN(4, 4, 8)
ffn.eval()
ffn(torch.ones((2, 3, 4)))[0] #输出变成(2,3,8)
'''
tensor([[ 0.1142,  0.6240,  0.4109, -0.0074, -0.0121, -0.4988, -0.0082, -0.1232],
        [ 0.1142,  0.6240,  0.4109, -0.0074, -0.0121, -0.4988, -0.0082, -0.1232],
        [ 0.1142,  0.6240,  0.4109, -0.0074, -0.0121, -0.4988, -0.0082, -0.1232]],
       grad_fn=<SelectBackward0>)
'''
~~~

Multi-Head Attention的内部结构中进行的主要都是矩阵乘法，即**进行的都是线性变换**。而线性变换的学习能力是不如非线性变化的强的，所以Multi-Head Attention的输出尽管利用了Attention机制，学习到了每个word的新representation表达，但是这种representation的表达能力可能并不强，我们仍然希望可以**通过激活函数的方式，来强化representation的表达能力**。

这也是为什么**在Attention层后加了一个Layer Normalizaiton层**，通过对representation进行标准化处理，将数据移动到激活函数的作用区域，可以使得ReLU激活函数更好的发挥作用。

[Feed Forward](https://blog.csdn.net/Urbanears/article/details/98742013)

## 2.2层归一化Layer Normalizaiton

批归一化batch normalizaiton：给一批数据(batch_size:b,序列长度:len,词元特征长度:d)，对于每个特征元素进行归一化，即每次对(b,len)大小的数据作均值为0方差为1的变换，做d次。然而此种方法在NLP应用中len是变化的，如果序列长度变化比较大，每个batch内计算的均值和方差的抖动就会比较大。另外模型在预测时使用记录的全局均值和方差，对于部分序列长度特别大或特别小的情况可能不适用。

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/batch_norm_in_nlp_1.png" alt="屏幕截图 2022-04-02 132226" style="zoom: 33%;" />

而层归一化Layer Normalizaiton：对于(b,len,d)的数据，每次对(len,d)大小的数据做归一化，做b次。由于层归一化是对于一个样本(单个序列)做归一化，抖动相对来说会更小，同时因为是在每个样本内部进行归一化，也不需要为了预测而记录全局均值和方差。

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/layer_norm.png" alt="image-20220402133701490" style="zoom: 33%;" />

[知乎问题：transformer 为什么使用 layer normalization，而不是其他的归一化方法？](https://www.zhihu.com/question/395811291)

## 2.3Encoder与Decoder间的信息传递

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/Information_transfer.png" alt="屏幕截图 2022-04-02 150911" style="zoom:50%;" />

编码器中的输出$y_1,...,y_n$作为解码器中第$i$个Transformer块中编码器－解码器注意力（encoder-decoder attention）层的key和value，该层的query来自目标序列。这意味着编码器和解码器中块的个数==(此处存疑，论文原文“which performs multi-head attention over the output of the **encoder stack**.”参考哈佛NLP团队实现的源码，Decoder块的第二个attention层K,V使用的是Encoder的最后一个块的输出，因此编码器和解码器中块的个数不必相同)==和输出维度是一样的。

