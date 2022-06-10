# Autograd

grad在反向传播过程中是累加的，这意味着每一次运行反向传播，梯度都会累加之前的梯度，所以一般在反向传播之前需把梯度清零。

~~~python
x.grad.data.zero_()
~~~

# pytorch 中矩阵乘法

**1. 二维矩阵乘法 torch.mm()**

```text
torch.mm(mat1, mat2, out=None)
```

该函数一般只用来计算两个二维矩阵的矩阵乘法，并且不支持broadcast操作。

**2. 三维带batch的矩阵乘法 torch.bmm()**

由于神经网络训练一般采用mini-batch，经常输入的时三维带batch的矩阵，所以提供

```text
torch.bmm(bmat1, bmat2, out=None)
```

该函数的两个输入必须是三维矩阵并且第一维相同（表示Batch维度）， 不支持broadcast操作

**3. 多维矩阵乘法 torch.matmul()**

```text
torch.matmul(input, other, out=None)
```

支持broadcast操作，使用起来比较复杂。针对多维数据 matmul() 乘法，可以认为该乘法使用使用两个参数的后两个维度来计算，其他的维度都可以认为是batch维度。

**4. 矩阵逐元素(Element-wise)乘法 torch.mul()**

```text
torch.mul(mat1, other, out=None)
```

其中 other 乘数可以是标量，也可以是任意维度的矩阵， 只要满足最终相乘是可以broadcast的即可。