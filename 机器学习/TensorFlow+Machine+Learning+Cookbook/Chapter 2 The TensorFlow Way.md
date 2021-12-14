# 1.计算图中的运算

第一步都是加载TensorFlow以及创建session

~~~python
import tensorflow as tf
sess = tf.Session()
~~~

1. 声明张量和占位符。创建numpy数组填入操作

~~~python
import numpy as np
x_vals = np.array([1., 3., 5., 7., 9.])
x_data = tf.placeholder(tf.float32)
m_const = tf.constant(3.)
my_product = tf.mul(x_data, m_const)
for x_val in x_vals:
print(sess.run(my_product, feed_dict={x_data: x_val}))
3.0
9.0
15.0
21.0
27.0
~~~

# 2.分层嵌套操作

在同一个计算图上放多个运算

1. 创建需要填入的数据和相应的占位符

~~~python
my_array = np.array([[1., 3., 5., 7., 9.],
					[-2., 0., 2., 4., 6.],
					[-6., -3., 0., 3., 6.]])
x_vals = np.array([my_array, my_array + 1])
x_data = tf.placeholder(tf.float32, shape=(3, 5))
~~~

2. 创建用来矩阵乘法和加法的常量

~~~python
m1 = tf.constant([[1.],[0.],[-1.],[2.],[4.]])
m2 = tf.constant([[2.]])
a1 = tf.constant([[10.]])
~~~

3. 声明操作并把他们加入计算图

~~~python
prod1 = tf.matmul(x_data, m1)
prod2 = tf.matmul(prod1, m2)
add1 = tf.add(prod2, a1)
~~~

4. 把数据填入图中

~~~python
for x_val in x_vals:
print(sess.run(add1, feed_dict={x_data: x_val}))
[[ 102.]
 [ 66.]
 [ 58.]]
[[ 114.]
 [ 78.]
 [ 70.]]
~~~

如果对数据形状不清楚，可以指定该维度为none。

`x_data = tf.placeholder(tf.float32, shape=(3,None))`

# 使用多层

介绍如何连接有数据流通的各层

~~~python
import tensorflow as tf
import numpy as np
sess = tf.Session()
~~~

1. 用numpy创建一个4*4像素的2D图像，4个参数分布式图像的数量、高度、宽度和通道数。

~~~python
x_shape = [1, 4, 4, 1]
x_val = np.random.uniform(size=x_shape)
~~~

2. 创建占位符。

~~~python
x_data = tf.placeholder(tf.float32, shape=x_shape)
~~~

3. 