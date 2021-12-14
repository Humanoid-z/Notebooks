# TensorFlow实现Softmax Regression识别手写数字

MNIST是一个非常简单的机器视觉数据集，它由几万张28像素×28像素的手写数字组成，这些图片只包含灰度值信息。任务就是对这些手写数字的图片进行分类，转成0~9一共10类。

~~~python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/"," one_hot=True)
~~~

mnist数据集的训练集有55000个样本,测试集有10000个样本，同时验证集有5000个样本。每一个样本都有它对应的标注信息，即 label。我们将在训练集上训练模型,在验证集上检验效果并决定何时完成训练,最后我们在测试集评测模型的效果（可通过准确率、召回率、F1-score等评测)。

~~~python
print(mnist.train.images.shape,mnist.train.lables.shape);
print(mnist.test.images.shape,mnist.test.lables.shape);
print(mnist.validation.images.shape,mnist.validation.lables.shape);
~~~

图像是28像素×28像素大小的灰度图片，如图所示。空白部分全部为0，有笔迹的地方根据颜色深浅有0到1之间的取值。

![image-20210613220309292](https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/20210613220309.png)

同时可以发现每个样本有784维的特征，也就是28×28个点的展开成1维的结果（28×28=784)。因此，这里丢弃了图片的二维结构方面的信息,只是把一张图片变成一个很长的1维向量。因为这个数据集的分类任务比较简单，不需要建立复杂的模型，所以简化了问题，丢弃空间结构的信息。后面的章节将使用卷积神经网络对空间结构信息进行利用，并取得更高的准确率。将图片展开成1维向量时，顺序并不重要，只要每一张图片都是用同样的顺序进行展开的就可以。

我们的训练数据的特征是一个55000x784 的 Tensor，第一个维度是图片的编号，第二个维度是图片中像素点的编号，如图3-3所示。

![image-20210613220553770](https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/20210613220553.png)

同时训练的数据Label是一个55000×10的Tensor，如图3-4所示

![image-20210613220642568](https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/20210613220642.png)

这里是对10个种类进行了**one-hot编码**(将离散型特征使用one-hot编码，会让特征之间的距离计算更加合理。)，Label是一个10维的向量，只有1个值为1，,其余为0。比如数字0，对应的Label就是[1,0,0,0,0,0,0,0,0.0]，数字5对应的Label就是[0,0,0,0,0,1,0,0,0,0]，数字n就代表对应位置的值为1。

使用Softmax Regression算法训练手写数字识别的分类模型。我们的数字都是0~9之间的，所以一共有10个类别，当我们的模型对一张图片进行预测时, Softmax Regression会对每一种类别估算一个概率，最后取概率最大的那个数字作为模型的输出结果。

当我们处理多分类任务时，通常需要使用Softmax Regression模型。即使后面章节的卷积神经网络或者循环神经网络,如果是分类模型,最后一层也同样是Softmax Regression.它的工作原理很简单，将可以判定为某类的特征相加，然后将这些特征转化为判定是这一类的概率。上述特征可以通过一些简单的方法得到，比如**对所有像素求一个加权和**，而权重是模型根据数据自动学习、训练出来的。比如某个像素的灰度值大代表很可能是数字n时，这个像素的权重就很大;反之，如果某个像素的灰度值大代表不太可能是数字n时，这个像素的权重就可能是负的。图3-5所示为这样的一些特征,其中明亮区域代表负的权重，灰暗区域代表正的权重。

![image-20210613222636491](https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/20210613222636.png)

~~~python
import tensorflow as tf;
sess = tf.InteractiveSession();
x = tf.placeholder(tf.float32,[None,784]);
//None代表不限条目的输入
W = tf.Variable(tf.zeros([784,10]));
b = tf.Variable(tf.zeros([10]));
y = tf.nn.sortmax(tf.matmul(x,W)+b);
~~~

为了训练模型，我们需要定义一个loss function来描述模型对问题的分类精度。Loss越小，代表模型的分类结果与真实值的偏差越小，也就是说模型越精确。我们一开始给模型填充了全零的参数，这样模型会有一个初始的 loss，而训练的目的是不断将这个loss减小，直到达到一个全局最优或者局部最优解。对多分类问题，通常使用cross-entropy作为loss function。Cross-entropy最早出自信息论( Information Theory )中的信息嫡(与压缩比率等有关)，然后被用到很多地方，包括通信、纠错码、博弈论、机器学习等。Cross-entropy的定义如下，其中y是预测的概率分布，y'是真实的概率分布（即Label的one-hot编码)，通常可以用它来判断模型对真实概率分布估计的准确程度。

![image-20210614121057580](https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/20210614121057.png)

~~~python
y_ = tf.placeholder(tf.float32,[None,10]);
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices = [1]));
~~~

现在我们有了算法Softmax Regression的定义,又有了损失函数cross-entropy的定义，只需要再定义一个优化算法即可开始训练。我们采用常见的随机梯度下降SGD( StochasticGradient Descent )。定义好优化算法后,TensorFlow就可以根据我们定义的整个计算图(我们前面定义的各个公式已经自动构成了计算图）自动求导，并根据反向传播(（BackPropagation)算法进行训练，在每一轮迭代时更新参数来减小loss。在后台TensorFlow会自动添加许多运算操作(Operation)来实现刚才提到的反向传播和梯度下降，而给我们提供的就是一个封装好的优化器，只需要每轮迭代时feed数据给它就好。我们直接调用tf.train.GradientDescentOptimizer，并设置学习速率为0.5，优化目标设定为cross-entropy,得到进行训练的操作 train_step。当然，TensorFlow中也有很多其他的优化器，使用起来也非常方便，只需要修改函数名即可。

~~~py
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entrop)
~~~

