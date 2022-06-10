# ValueError: too many values to unpack

接收函数返回结果的变量数大于返回变量数，如：

```python
class net(nn.Module):
    def forward(self):
        ...
        return dec_output[0]

Y_hat, _ = net(X, dec_input, X_valid_len)
```