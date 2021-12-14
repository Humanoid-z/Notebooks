# 环境

源代码：https://github.com/nfmcclure/tensorflow_cookbook

Tensorflow：https://www.tensorflow.org/

pip install tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple/



```
conda install tensorflow
conda list
conda install cudatoolkit=11.0 -c http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/win-64/
conda install cudnn=7.6.5 -c http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/win-64/
```

# TensorFlow算法的一般流程

1. 导入或生成数据集
2. 转换和规范化数据
3. 把数据集划分为训练集、测试集和验证集
4. 设置算法参数(超参数)
5. 初始化变量和占位符
6. 规定模型结构
7. 声明损失函数
8. 初始化并训练模型
9. 评估模型
10. 调整超参数
11. 部署/预测新的结果



# 张量tensor的声明

1. 确定的张量

~~~python
zero_tsr = tf.zeros([row_dim, col_dim])
ones_tsr = tf.ones([row_dim, col_dim])
filled_tsr = tf.fill([row_dim, col_dim], 42)
constant_tsr = tf.constant([1,2,3])
~~~

2. 相同大小的张量

~~~python
zeros_similar = tf.zeros_like(constant_tsr)
ones_similar = tf.ones_like(constant_tsr)
~~~

3. 张量序列

~~~python
linear_tsr = tf.linspace(start=0, stop=1, start=3)//[0.0, 0.5, 1.0]
integer_seq_tsr = tf.range(start=6, limit=15, delta=3)//[6, 9, 12]
~~~

4. 随机张量

~~~python
randunif_tsr = tf.random_uniform([row_dim, col_dim],minval=0, maxval=1)//uniform distribution
randnorm_tsr = tf.random_normal([row_dim, col_dim],mean=0.0, stddev=1.0)//normal distribution
runcnorm_tsr = tf.truncated_normal([row_dim, col_dim],mean=0.0, stddev=1.0)
//randomizing entries of arrays
shuffled_output = tf.random_shuffle(input_tensor)
cropped_output = tf.random_crop(input_tensor, crop_size)
//randomly cropping an image of size (height, width, 3)
cropped_image = tf.random_crop(my_image, [height/2, width/2,3])
~~~

# 占位符和变量

变量是算法的参数，TensorFlow跟踪如何改变变量来优化算法。占位符是允许输入特定类型和形状的数据的对象，并依赖于计算图的结果，例如计算的预期结果。

实例化一个变量主要使用`Variable()`方法，以张量为输入，变量为输出。

初始化是将具有相应方法的变量放到计算图上。

~~~python
my_var = tf.Variable(tf.zeros([2,3]))
sess = tf.Session()
initialize_op = tf.global_variables_initializer ()
sess.run(initialize_op)
~~~

占位符从session中的`feed_dict`参数获取数据.要将占位符放入计算图中，要先将至少一个操作赋给占位符。下例中声明x是一个占位符，定义y是x的一个操作(仅返回x)，然后创建数据赋给x，运行定义的操作。

~~~python
sess = tf.Session()
x = tf.placeholder(tf.float32, shape=[2,2])
y = tf.identity(x)
x_vals = np.random.rand(2,2)
sess.run(y, feed_dict={x: x_vals})
# Note that sess.run(x, feed_dict={x: x_vals}) will result in a selfreferencing error.
~~~

如果有变量的初始化依赖其它变量初始化的结果，我们需要按顺序初始化：

~~~python
sess = tf.Session()
first_var = tf.Variable(tf.zeros([2,3]))
sess.run(first_var.initializer)
second_var = tf.Variable(tf.zeros_like(first_var))
# Depends on first_var
sess.run(second_var.initializer)
~~~

# 矩阵

1. 创建矩阵

~~~python
identity_matrix = tf.diag([1.0, 1.0, 1.0])
A = tf.truncated_normal([2, 3])
B = tf.fill([2,3], 5.0)
C = tf.random_uniform([3,2])
D = tf.convert_to_tensor(np.array([[1., 2., 3.],[-3., -7.,-1.],[0., 5., -2.]]))
print(sess.run(identity_matrix))
[[ 1. 0. 0.]
 [ 0. 1. 0.]
 [ 0. 0. 1.]]
print(sess.run(A))
[[ 0.96751703 0.11397751 -0.3438891 ]
 [-0.10132604 -0.8432678 0.29810596]]
print(sess.run(B))
[[ 5. 5. 5.]
[ 5. 5. 5.]]
print(sess.run(C))
[[ 0.33184157 0.08907614]
 [ 0.53189191 0.67605299]
 [ 0.95889051 0.67061249]]
print(sess.run(D))
[[ 1. 2. 3.]
 [-3. -7. -1.]
 [ 0. 5. -2.]]
~~~

2. 加减法运算

~~~python
print(sess.run(A+B))
[[ 4.61596632 5.39771316 4.4325695 ]
[ 3.26702736 5.14477345 4.98265553]]
print(sess.run(B-B))
[[ 0. 0. 0.]
[ 0. 0. 0.]]
Multiplication
print(sess.run(tf.matmul(B, identity_matrix)))
[[ 5. 5. 5.]
[ 5. 5. 5.]]
~~~

3. 转置

~~~python
print(sess.run(tf.transpose(C)))
[[ 0.67124544 0.26766731 0.99068872]
[ 0.25006068 0.86560275 0.58411312]]
//重新初始化矩阵C获得了不同的结果
~~~

4. 行列式

~~~python
print(sess.run(tf.matrix_determinant(D)))
-38.0
~~~

5. 求逆矩阵Inverse

~~~python
print(sess.run(tf.matrix_inverse(D)))
[[-0.5 -0.5 -0.5 ]
[ 0.15789474 0.05263158 0.21052632]
[ 0.39473684 0.13157895 0.02631579]]
~~~

6. 矩阵分解

~~~python
print(sess.run(tf.cholesky(identity_matrix)))
[[ 1. 0. 1.]
[ 0. 1. 0.]
[ 0. 0. 1.]]
~~~

7. 特征值、特征向量

~~~python
print(sess.run(tf.self_adjoint_eig(D))
[[-10.65907521 -0.22750691 2.88658212]
[ 0.21749542 0.63250104 -0.74339638]
[ 0.84526515 0.2587998 0.46749277]
[ -0.4880805 0.73004459 0.47834331]]
//第一行是特征值，下面是特征向量
~~~

# 声明操作

1. 除法

~~~python
print(sess.run(tf.div(3,4)))
0
print(sess.run(tf.truediv(3,4)))
0.75
print(sess.run(tf.floordiv(3.0,4.0)))
0.0
~~~

2. 取模

~~~python
print(sess.run(tf.mod(22.0, 5.0)))
2.0
~~~

3. 向量积

~~~python
print(sess.run(tf.cross([1., 0., 0.], [0., 1., 0.])))
[ 0. 0. 1.0]
~~~

4. 常用数学函数表

   ![image-20210610151001719](https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/20210610151001.png)

5. 专业数学函数表

   ![image-20210610151109248](https://raw.githubusercontent.com/SNIKCHS/MDImage/main/img/20210610151109.png)

自定义函数

~~~python
def custom_polynomial(value):
	return(tf.sub(3 * tf.square(value), value) + 10)
print(sess.run(custom_polynomial(11)))
~~~

# 激活函数

1. 线性整流函数(ReLU)

~~~python
print(sess.run(tf.nn.relu([-3., 3., 10.])))
[ 0. 3. 10.]
~~~

2. 为ReLU增加上限

~~~python
print(sess.run(tf.nn.relu6([-3., 3., 10.])))
[ 0. 3. 6.]
~~~

3. S形函数 Logistic函数是常见的S形函数，形为`1/(1+exp(-x))`

~~~python
print(sess.run(tf.nn.sigmoid([-1., 0., 1.])))
[ 0.26894143 0.5 0.7310586 ]
~~~

一些激活函数不是以0为中心的，比如`sigmoid`，会导致神经网络收敛较慢，这要求我们在使用大多数计算图算法前将数据**零均值化**。

4. hyper tangent

~~~python
print(sess.run(tf.nn.tanh([-1., 0., 1.])))
[-0.76159418 0. 0.76159418 ]
~~~

5. softsign `x/(abs(x) + 1)`

~~~python
print(sess.run(tf.nn.softsign([-1., 0., -1.])))
[-0.5 0. 0.5]
~~~

6. softplus `log(exp(x) + 1)`

~~~python
print(sess.run(tf.nn.softplus([-1., 0., -1.])))
[ 0.31326166 0.69314718 1.31326163]
~~~

7. 指数线性单元 Exponential Linear Unit (ELU) 

   `(exp(x)+1) if x < 0 else x`

~~~python
print(sess.run(tf.nn.elu([-1., 0., -1.])))
[-0.63212055 0. 1. ]
~~~

使用激活函数在神经网络或其他计算图中引入非线性

# 数据源

1. 鸢尾花数据集

   使用py加载

~~~python
from sklearn import datasets
iris = datasets.load_iris()
print(len(iris.data))
150
print(len(iris.target))
150
print(iris.target[0]) # Sepal length, Sepal width, Petal length,
Petal width
[ 5.1 3.5 1.4 0.2]
print(set(iris.target)) # I. setosa, I. virginica, I. versicolor
{0, 1, 2}
~~~

2. 出生体重数据

~~~python
import requests
birthdata_url = 'https://www.umass.edu/statdata/statdata/data/
lowbwt.dat'
birth_file = requests.get(birthdata_url)
birth_data = birth_file.text.split('\'r\n') [5:]
birth_header = [x for x in birth_data[0].split( '') if len(x)>=1]
birth_data = [[float(x) for x in y.split( ')'' if len(x)>=1] for y in birth_data[1:] if len(y)>=1]
print(len(birth_data))
189
print(len(birth_data[0]))
~~~

3. 波士顿住房数据

~~~python
import requests
housing_url = 'https://archive.ics.uci.edu/ml/machine-learningdatabases/
housing/housing.data'
housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV0']
housing_file = requests.get(housing_url)
housing_data = [[float(x) for x in y.split( '') if len(x)>=1] for y in housing_file.text.split('\n') if len(y)>=1]
print(len(housing_data))
506
print(len(housing_data[0]))
~~~

4. MNIST 手写数据集

~~~python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/"," one_hot=True)
print(len(mnist.train.images))
55000
print(len(mnist.test.images))
10000
print(len(mnist.validation.images))
5000
print(mnist.train.labels[1,:]) # The first label is a 3'''
[ 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
~~~

5. Spam-ham文本数据

~~~python
import requests
import io
from zipfile import ZipFile
zip_url = 'http://archive.ics.uci.edu/ml/machine-learningdatabases/00228/smsspamcollection.zip'
r = requests.get(zip_url)
z = ZipFile(io.BytesIO(r.content))
file = z.read('SMSSpamCollection')
text_data = file.decode()
text_data = text_data.encode('ascii',errors='ignore')
text_data = text_data.decode().split('\n')
text_data = [x.split('\t') for x in text_data if len(x)>=1]
[text_data_target, text_data_train] = [list(x) for x in zip(*text_
data)]
print(len(text_data_train))
5574
print(set(text_data_target))
{'ham', 'spam'}
print(text_data_train[1])
Ok lar... Joking wif u oni...
~~~

6. 电影评论数据

~~~python
import requests
import io
import tarfile
movie_data_url = 'http://www.cs.cornell.edu/people/pabo/moviereview-
data/rt-polaritydata.tar.gz'
r = requests.get(movie_data_url)
# Stream data into temp object
stream_data = io.BytesIO(r.content)
tmp = io.BytesIO()
while True:
s = stream_data.read(16384)
if not s:
break
tmp.write(s)
stream_data.close()
tmp.seek(0)
# Extract tar file
tar_file = tarfile.open(fileobj=tmp, mode="r:gz")
pos = tar_file.extractfile('rt'-polaritydata/rt-polarity.pos')
neg = tar_file.extractfile('rt'-polaritydata/rt-polarity.neg')
# Save pos/neg reviews (Also deal with encoding)
pos_data = []
for line in pos:
pos_data.append(line.decode('ISO'-8859-1').
encode('ascii',errors='ignore').decode())
neg_data = []
for line in neg:
neg_data.append(line.decode('ISO'-8859-1').
encode('ascii',errors='ignore').decode())
tar_file.close()
print(len(pos_data))
5331
print(len(neg_data))
5331
# Print out first negative review
print(neg_data[0])
simplistic , silly and tedious .
~~~

7. CIFAR-10图像数据

`http://www.cs.toronto.edu/~kriz/cifar.html`

8. 莎士比亚作品文本资料

~~~python
import requests
shakespeare_url = 'http://www.gutenberg.org/cache/epub/100/pg100.
txt'
# Get Shakespeare text
response = requests.get(shakespeare_url)
shakespeare_file = response.content
# Decode binary into string
shakespeare_text = shakespeare_file.decode('utf-8')
# Drop first few descriptive paragraphs.
shakespeare_text = shakespeare_text[7675:]
print(len(shakespeare_text)) # Number of characters
5582212
~~~

9. 英德句子翻译数据

~~~python
import requests
import io
from zipfile import ZipFile
sentence_url = 'http://www.manythings.org/anki/deu-eng.zip'
r = requests.get(sentence_url)
z = ZipFile(io.BytesIO(r.content))
file = z.read('deu.txt''')
# Format Data
eng_ger_data = file.decode()
eng_ger_data = eng_ger_data.encode('ascii''',errors='ignore''')
eng_ger_data = eng_ger_data.decode().split('\n''')
eng_ger_data = [x.split('\t''') for x in eng_ger_data if len(x)>=1]
[english_sentence, german_sentence] = [list(x) for x in zip(*eng_
ger_data)]
print(len(english_sentence))
137673
print(len(german_sentence))
137673
print(eng_ger_data[10])
['I won!, 'Ich habe gewonnen!']
~~~

# 其它资源

TensorFlow Python API：

`https://www.tensorflow.org/api_docs/python`

TensorFlow官方教程:

`https://www.tensorflow.org/tutorials/index.html`