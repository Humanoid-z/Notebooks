# 布局管理器

## 布局类型

- 线性布局（LinearLayout）
- 相对布局（RelativeLayout）
- 表格布局（TableLayout）
- 网格布局（GridLayout）
- 绝对布局（AbsoluteLayout）
- 帧布局（FrameLayout）
- 约束性布局（ConstraintLayout）扁平化布局

### 线性布局

线性布局是Android中较为常用的布局方式，它使用<LinearLayout>标签，主要分为水平线性布局和垂直线性布局。

# Android控件

## TextView（文本框）

TextView直接继承了View，它还是EditText和Button两个UI组件类的父类。TextView的作用就是在界面上显示文字，通过在布局文件当中或者在Activity中修改文字的内容。

## EditText（输入框）

EditText与TextView非常的相似，许多XML属性都能共用，与TextView的最大区别就是EditText能够接受用户的输入。EditText的重要属性就是inputType,该属性相当于Html的<input…/>元素的type属性，用于将EditText设置为指定类型的输入组件，如手机号、密码、日期等。还有一个属性是当提示用户当前文本框要输入的内容是什么，使用android:hint=“”来提示用户，当用户点击文本框这些文字就会消失。