# Tensor给部分元素赋值

**scatter()参数**

- dim：指定轴方向，以二维Tensor为例：

- - dim=0表示逐列进行填充
  - dim=1表示逐行进行填充

- index：LongTensor，按照轴方向，在源Tensor中需要填充的位置

- src：用来进行填充的值：

- - src为一个数时，用这个数替换所有index位置上的值
  - src为一个Tensor时，其shape必须与index一致，src中的元素会按顺序填充至对应index的位置上

给不同行、不同列的元素赋不同的值:

```python
id = torch.tensor([[0],[2],[1],[3],[1]])
u = torch.zeros((id.shape[0],4))
u = u.scatter(1,id,1)
print(u)

tensor([[1., 0., 0., 0.],
        [0., 0., 1., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 1.],
        [0., 1., 0., 0.]])
```

