# numpy :Python进行科学计算的基本软件包

## numpy.ndarray类型数组

```python
Array.shape #(1,209) type(train_set_y_orig.shape)==<class 'tuple'>
Array.shape[0]==1
Array.shape[1]==209

type(Array[0,10])==<class 'numpy.int64'>
type(Array[:,10])==<class 'numpy.ndarray'>
```

## axis

axis=i，则numpy沿着第i个下标变化的方向进行操作

```python
a= np.array([[1,2],[3,4]])  
a.sum(axis = 0)
>>>array([4, 6])
```

```python
A = np.random.randn(4,3)
B = np.sum(A, axis = 1, keepdims = True)
# B.shape = (4, 1) （keepdims = True）确保A.shape是（4,1）而不是（4，）
```

## np.round()

对小数四舍五入，当近似的小数为5时，则前一位近似为偶数。

```python
>>> np.around([.5, 1.5, 2.5, 3.5, 4.5]) # rounds to nearest even value
array([ 0.,  2.,  2.,  4.,  4.])
```

## np.random

```python
np.random.randn(d0,d1,…,dn)
#返回指定维度的array，具有标准正态分布，以0为均值、以1为标准差
```

