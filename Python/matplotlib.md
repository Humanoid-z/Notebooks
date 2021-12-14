# matplotlib:在Python中绘制图表

1. plt.imshow不显示图像的问题

   解决方法:在后面加一句：plt.show()

   原理：plt.imshow()函数负责对图像进行处理，并显示其格式，而plt.show()则是将plt.imshow()处理后的函数显示出来。

## scatter()

```
matplotlib.pyplot.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, edgecolors=None, hold=None, data=None, **kwargs)
```

绘制x与y的散点图,点的大小按==s==缩放;==c==控制标记颜色映射，长度与x，y相同，根据值的不同使得（x,y）参数对表现为不同的颜色。可以理解为某个散点的type，根据type的不同画出不同颜色

# 问题

1. 同时画点和线

   ```
   plt.figure()  #同时画点和线
   plt.plot(X,Y)
   plt.scatter(X,Y)
   ```

2. 正常显示中文和负号

   ```python
   plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文
   plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
   ```

   