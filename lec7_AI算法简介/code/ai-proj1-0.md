
# 项目1： 统计字母频率
给定一个字符串，统计其字母出现的频率


```python
# Tip:
# - 导入string库
# - 创建一个变量，命名为alphabet
# - 使用string的数据属性ascii_letters获取大小写英文字符，存在alphabet中


```


```python
sentence = 'Heal the world, Make it a better place.For you and for me, And the entire human race'
count_letters = {}

# Tip:
# - count_letters 字典应该包含 sentence中所有大小写字母出现的次数
# 显示出该字典

```


```python
# Tip:
# 将上述功能写成函数，记为counter， 实现同样的功能


```


```python
address = "  Four score and seven years ago our fathers brought forth  \\\nNow we are engaged in a great civil war  testing whether that nation  or any nation so conceived                                   and so dedicated         \n\n                                                                                                                                upon this continent  \\\nNow we are engaged in a great civil war  testing whether that nation  or any nation so conceived   can long endure. We are met on a great battle...   \n\n                                                                                                                                       a new nation  \\\nNow we are engaged in a great civil war  testing whether that nation  or any nation so conceived   as a final resting place for those who died here   \n\n                                                                                                                      conceived in liberty  \\\nNow we are engaged in a great civil war  testing whether that nation  or any nation so conceived   that the nation might live. This we may   \n\n                                                                                                  and dedicated to the proposition that all men are created equal.  \nNow we are engaged in a great civil war  testing whether that nation  or any nation so conceived                               in all propriety do."
```


```python
# Tip:
# 使用counter函数统计上述字符串的字母频率


```


```python
# Tip:
# 求address中出现频率最高的字母


```

# 项目 2： 蒙特卡罗方法计算$\pi$
已知：
$$\frac{正方形的内切圆的面积}{正方形的面积} = \frac{\pi}{4}$$
用近似的方法求$\pi$


```python
# Tip:
# 使用math库，计算pi/4,并打印输出


```


```python
# Tip:
# 使用random.uniform()定义一个函数rand, 产生一个-1到1之间的float型的数

import random
random.seed(1) # This line fixes the value called by your function,
               # and is used for answer-checking.


```


```python
import math

# Tip
# - 编写一个函数distance，计算两个点x和y之间的距离
# - 使用你的函数计算点x=(0,0)，y=(1,1)之间的距离
# - 开方 math.sqrt

```


```python
import random, math
random.seed(1)
# Tip: 
# - 编写函数判断点x是否在单位圆内
# - 调用函数in_circle判断点(0,0)是否在以(0,0)为原点的单位圆内


   
```



```python
R = 10000

# Tip:
# - 统计有多少个点落在单位圆内
# - 计算 落在单位圆内点数 的比例 

```


```python
# 计算误差： 求pi/4 与计算值之间的差异

```



