TreeSet中的元素将按照升序排列，缺省是按照自然排序进行排列

Integer能排序(有默认顺序), String能排序(有默认顺序), 自定义的类存储的时候出现异常(没有顺序)

TreeSet是一个有序集合，TreeSet中的元素将按照升序排列，缺省是按照自然排序进行排列，意味着TreeSet中的元素要实现Comparable接口。或者有一个自定义的比较器。

我们可以在构造TreeSet对象时，传递实现Comparator接口的比较器对象。