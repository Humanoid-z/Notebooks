2017 年由Google 机器翻译团队发表的此篇论文完全抛弃了RNN和CNN等网络结构，而仅仅采用Attention机制来进行机器翻译任务，并且取得了很好的效果，自注意力机制此后成为了研究热点。Transformer 开创了继 MLP 、CNN和 RNN之后的第四大类模型。部分学者建议将Transformer作为基础模型。

# 摘要

2017年主流的序列转录模型大多基于复杂的包含一个编码器和一个解码器的循环/卷积神经网络。性能好的模型通常会在编码器和解码器之间使用注意力机制。这篇文章设计的Transformer模型**仅采用Attention机制**，完全抛弃了RNN和CNN。该模型在两个机器翻译任务上的实验表明，Transformer在质量上更优，同时具有更强的**并行性**，需要**更少的训练时间**。实验结果：英 - 德：提高 2 BLEU；英 - 法：SOTA，41.8 BLEU，只需8GPUs的3.5天的训练。此外该模型能很好地**泛化到其它任务**。

# 7 结论

Transformer是第一个完全基于注意力的序列转换模型，用多头自注意替换了编码器-解码器架构中最常用的循环层。对于翻译任务，Transformer的训练速度比基于循环或卷积层的体系结构快得多。

计划将**纯注意力模型应用在其他任务**：图片、音频、视频；使生成不那么时序化

# 1 导言

RNN，LSTM，GRU在2017年是序列模型的最优方法。RNN：隐藏状态$h_t$根据上个隐藏状态$h_{t-1}$和$t$位置输入计算，缺点：（训练时）难以并行

attention在序列模型上的应用：允许对依赖关系建模而**不考虑它们在输入或输出序列中的距离**。然而，大多数情况下attention是与RNN结合使用的。

本文提出的Transformer避免了循环结构，完全依赖于注意力机制来绘制输入和输出之间的全局依赖关系。支持更高的并行化，在8个P100 GPU上经过12个小时的训练后，可以在翻译质量上达到最优水平。

# 2 相关工作(Background)

使用CNN替换RNN来减少时序计算的工作：Extended Neural GPU, ByteNet，ConvS2S都使用CNN作为基本block，并行计算输入和输出所有位置的隐藏状态。这些模型学习两个任意输入或输出位置的关系所需的计算随距离增长，难以学习长距离位置之间的关系。而Transformer上述学习所需的计算为常数，但会由于注意力加权位置的平均化(?)影响结果，==多头注意力机制==可以抵消这一影响。（模拟CNN使用多个卷积核识别不同的模式）

自我注意Self-attention是一种将序列的不同位置联系起来以计算出序列的一种表示形式的注意力机制（即新的representation中每个位置都和原序列所有位置相关）。Self-attention在阅读理解、抽象摘要、文本蕴涵和学习任务无关的句子表征等任务中都得到了成功的应用

还提到17年比较热门的memory nework

就作者所知Transformer是第一个完全依赖于Self-attention来计算其输入和输出表示的序列转换模型，而不使用RNNs或CNN。

# 3 模型架构

大多数好的的序列转换模型都有一个encoder-decoder结构。encoder将原始输入序列(符号表示)$(x_1,...,x_n)$映射成序列$\mathbf{z}=(z_1,...,z_n)$(一串tensor)。decoder拿到$\mathbf{z}$后时序地生成输出序列$(y_1,...,y_m)$(符号表示)，每一步都是自回归的，在生成下一个符号时，将之前生成的符号作为额外的输入。

Transformer遵循encoder-decoder架构，编码器和解码器是将self-attention层和基于位置的前馈网络层堆叠起来的结构。如下图

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/Transformer_en.svg" alt="../_images/transformer.svg" style="zoom: 50%;" />

## 3.1 编码器和解码器栈

Encoder：由6个相同的encoder block堆叠。每个block有2个子层。第一个子层是**多头自注意力层**，第二层是简单的基于位置的前馈网络层(FFN)。每个子层都使用残差连接，接着进行**层归一化**。即每个子层的输出为：$LayerNorm(x+Sublayer(x))$，为了便于进行残差连接，每层输出维度$d_{model}=512$

Layer Norm：见[笔记04-Transformer 2.2](D:\OneDrive - stmail.ujs.edu.cn\Notebooks\深度学习\deeplearning_ai_books-master\笔记\笔记04-Transformer.md)

Decoder：由6个相同的decoder block堆叠。在encoder block的结构上加入第三个子层：**基于Encoder输出的多头注意力机制层**。每个子层和encoder block一样有着残差连接和层归一化。（**为了训练的并行化实现**，见[笔记04-Transformer 2.1](D:\OneDrive - stmail.ujs.edu.cn\Notebooks\深度学习\deeplearning_ai_books-master\笔记\笔记04-Transformer.md)）在decoder block中的自注意子层添加**mask机制**，确保位置$i$的预测只能依赖于$i$之前的已知输出，以防止后续位置对当前位置的影响。

## 3.2 注意力机制

注意力函数可以描述为：将一个查询和一组键值对映射到输出，其中查询、键、值和输出都是向量。输出是值的加权和，每个值的权重由查询与相应键的compatibility函数计算得来。

### 3.2.1 缩放点积注意力

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/Scaled%20Dot-Product%20Attention_1.png" alt="image-20220410224459935" style="zoom: 67%;" />

见[笔记04-Transformer 1.3.2](D:\OneDrive - stmail.ujs.edu.cn\Notebooks\深度学习\deeplearning_ai_books-master\笔记\笔记04-Transformer.md)

### 3.2.2 多头注意力机制

<img src="https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/Multi-Head%20Attention.png" alt="image-20220410225224321" style="zoom:50%;" />

见[笔记04-Transformer 1.6](D:\OneDrive - stmail.ujs.edu.cn\Notebooks\深度学习\deeplearning_ai_books-master\笔记\笔记04-Transformer.md)

### 3.2.3 本模型中注意力机制的应用

Transformer以三种不同的方式使用多头注意力：

- 在"encoder-decoder attention"层，查询来自上一个的decoder block，键和值来自encoder的输出。这样能使decoder的每个位置都能考虑到输入序列的所有位置。
- encoder的自注意力层。该层所有的键、值和查询都来自同一个地方，即前一encoder block的输出。这样每个位置可以考虑前一encoder block的所有位置。
- decoder的自注意力层也让每个位置可以考虑前一decoder block中该位置之前并包括该位置的部分。为了防止信息在解码器中向左流动，以保持自回归特性，通过mask(设置为$-∞$)softmax输入中与非法连接对应的值来实现缩放点积注意。

## 3.3 基于位置的前馈网络

该网络包括两个线性转换，中间有一个ReLU激活。等价于做两次1*1卷积。

$FFN(x)=max(0,xW_1+b_1)W_2+b_2$

输入输出维数$d_{model} = 512$，内层维数$d_{ff} = 2048$。

见[笔记04-Transformer 2.1](D:\OneDrive - stmail.ujs.edu.cn\Notebooks\深度学习\deeplearning_ai_books-master\笔记\笔记04-Transformer.md)

## 3.4 嵌入层和Softmax

与其他序列转换模型类似，使用学习过的嵌入将输入标记和输出符号转换为维数$d_{model}$的向量。还使用线性变换和softmax将解码器输出转换为预测的下一个符号概率。在模型中，共享嵌入层的权值矩阵。在嵌入层中，把权重乘以$\sqrt{d_{model}}$（embedding层默认进行L2规范化，如pytorch的实现：

```python
torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None,max_norm=None,norm_type=2.0,scale_grad_by_freq=False, sparse=False,  _weight=None)
```

$d_{model}$越大，嵌入向量每个元素越小，但之后还需要加上位置编码positional encoding。 为了使 embedding 和  positional encosing 的 scale 差不多，让权重乘以$\sqrt{d_{model}}$）

## 3.5位置编码

见[笔记04-Transformer 1.5.2](D:\OneDrive - stmail.ujs.edu.cn\Notebooks\深度学习\deeplearning_ai_books-master\笔记\笔记04-Transformer.md)

作者也尝试过可学习的positional embeddings，但是这种方法与三角函数的位置编码有着几乎相同的效果，因此选择后者因为它可以让模型泛化应用到比训练序列更长的序列长度。

# 4 为什么用自注意力

本章将Self-Attention层的各个方面与RNN和CNN相比较：

1. 总计算复杂度。
2. 可以并行化的计算量。（由所需的最小顺序操作数衡量）
3. 长距离依赖关系之间的路径长度（信息从一个位置走到另外一个位置要走多少步）。路径越短，学习长距离依赖关系就越容易。

![image-20220411124400697](https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/compare%20with%20cnn%20rnn%20self_att.png)

# 5 实验Training

## 5.1训练数据和批处理

数据集：WMT 2014 English-German dataset（450万英德语句对）和WMT
2014 English-French dataset（3600万个英法语句对，32000大小字典）

batch_num=25000 tokens

## 5.2 硬件及训练策略

8*NVIDIA P100 GPU

base models：0.4 seconds / batch, 100, 000 steps or 12 hours 

big models: 1s / step, 300, 000 steps, 3.5 days

## 5.3 Optimizer

Adam：$β_1=0.9,β_2=0.98,\epsilon=10^{-9}$

学习率衰减：$learning\ rate=d^{-0.5}_{model}*min(step\_num^{-0.5},step\_num*warmup\_steps^{-1.5})$

$warmup\_steps=4000$

## 5.4正则化

残差Dropout：每个子层的输出在进入残差连接之前和 layer norm 前都使用dropout。embedding和positional encoding的相加也使用dropout。$dropout=0.1$

Label Smoothing：$\epsilon_{ls}=0.1$ 对于正确的值，softmax的输出很难逼近于1，Label Smoothing让正确的值只需要到0.1，剩下的值是 0.9 / 字典大小。Label Smoothing损害了困惑度Perplexity，但提高了准确性和BLEU得分。

# 面试题

1. Transformer为何使用多头注意力机制？（为什么不使用一个头）
2. Transformer为什么Q和K使用不同的权重矩阵生成，为何不能使用同一个值进行自身的点乘？ （注意和第一个问题的区别）
3. Transformer计算attention的时候为何选择点乘而不是加法？两者计算复杂度和效果上有什么区别？
4. 为什么在进行softmax之前需要对attention进行scaled（为什么除以dk的平方根），并使用公式推导进行讲解
5. 在计算attention score的时候如何对padding做mask操作？
6. 为什么在进行多头注意力的时候需要对每个head进行降维？（可以参考上面一个问题）
7. 大概讲一下Transformer的Encoder模块？
8. 为何在获取输入词向量之后需要对矩阵乘以embedding size的开方？意义是什么？
9. 简单介绍一下Transformer的位置编码？有什么意义和优缺点？
10. 你还了解哪些关于位置编码的技术，各自的优缺点是什么？
11. 简单讲一下Transformer中的残差结构以及意义。
12. 为什么transformer块使用LayerNorm而不是BatchNorm？LayerNorm 在Transformer的位置是哪里？
13. 简答讲一下BatchNorm技术，以及它的优缺点。
14. 简单描述一下Transformer中的前馈神经网络？使用了什么激活函数？相关优缺点？
15. Encoder端和Decoder端是如何进行交互的？（在这里可以问一下关于seq2seq的attention知识）
16. Decoder阶段的多头自注意力和encoder的多头自注意力有什么区别？（为什么需要decoder自注意力需要进行 sequence mask)
17. Transformer的并行化提现在哪个地方？Decoder端可以做并行化吗？
18. Transformer训练的时候学习率是如何设定的？Dropout是如何设定的，位置在哪里？Dropout 在测试的需要有什么需要注意的吗？
19. 解码端的残差结构有没有把后续未被看见的mask信息添加进来，造成信息的泄露。