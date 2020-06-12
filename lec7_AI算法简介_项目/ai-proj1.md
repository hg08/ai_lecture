
# 项目1： 统计字母频
给定一个字符串，统计其字母出现的频率


```python
# Tip:
# - 导入string库
# - 创建一个变量，命名为alphabet
# - 使用string的数据属性ascii_letters获取大小写英文字符，存在alphabet中

# import string
# alphabet = string.ascii_letters
# print(alphabet)
```


```python
sentence = 'Heal the world, Make it a better place.For you and for me, And the entire human race'
count_letters = {}

# Tip:
# - count_letters 字典应该包含 sentence中所有大小写字母出现的次数


# for letter in sentence:
#     if letter != ' ':
#         count_letters[letter] = 0

# for letter in sentence:
#     if letter != ' ':
#         count_letters[letter] += 1

# print(count_letters)
```


```python
# Tip:
# 将上述功能写成函数，记为counter

# def counter(input_string):
#     count_letters = {}
#     for letter in input_string:
#         if letter in alphabet:
#             count_letters[letter] = 0
#     for letter in input_string:
#         if letter in alphabet:
#             count_letters[letter] += 1
#     return count_letters

# counter(sentence)
```


```python
address = "  Four score and seven years ago our fathers brought forth  \\\nNow we are engaged in a great civil war  testing whether that nation  or any nation so conceived                                   and so dedicated         \n\n                                                                                                                                upon this continent  \\\nNow we are engaged in a great civil war  testing whether that nation  or any nation so conceived   can long endure. We are met on a great battle...   \n\n                                                                                                                                       a new nation  \\\nNow we are engaged in a great civil war  testing whether that nation  or any nation so conceived   as a final resting place for those who died here   \n\n                                                                                                                      conceived in liberty  \\\nNow we are engaged in a great civil war  testing whether that nation  or any nation so conceived   that the nation might live. This we may   \n\n                                                                                                  and dedicated to the proposition that all men are created equal.  \nNow we are engaged in a great civil war  testing whether that nation  or any nation so conceived                               in all propriety do."
```


```python
# Tip:
# 使用counter函数统计上述字符串的字母频率

# address_count = counter(address)
# print(address_count)
```


```python
# Tip:
# 求address中出现频率最高的字母

# most_frequent_letter = ''
# Max = 0
# for letter in address_count:
#     if address_count[letter] > Max:
#         Max = address_count[letter]
#         most_frequent_letter = letter
# print(most_frequent_letter)
```

# 项目 2： 蒙特卡罗方法计算$\pi$
已知：
$$\frac{正方形的内切圆的面积}{正方形的面积} = \frac{\pi}{4}$$
用近似的方法求$\pi$


```python
# Tip:
# 使用math库，计算pi/4,并打印输出

# import math
# pi = math.pi
# print(pi/4)

```


```python
# Tip:
# 使用random.uniform()定义一个函数rand, 产生一个-1到1之间的float型的数

import random
random.seed(1) # This line fixes the value called by your function,
               # and is used for answer-checking.

# def rand():
#     # define `rand` here!
#     res = random.uniform(-1,1)
#     return res
    
# rand()
```


```python
import math

# Tip
# - 编写一个函数distance，计算两个点x和y之间的距离
# - 使用你的函数计算点x=(0,0)，y=(1,1)之间的距离
# - 开方 math.sqrt

# def distance(x, y):
#     res = math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
#     return res
   
# print(distance((0,0),(1,1)))
```


```python
import random, math
random.seed(1)
# Tip: 
# - 编写函数判断点x是否在单位圆内
# - 调用函数in_circle判断点(0,0)是否在以(0,0)为原点的单位圆内

def in_circle(x, origin = [0]*2):
   # Define your function here!
#     if distance(x, origin) < 1:
#         return True
#     else:
#         return False

# print(in_circle((1,1),(0,0)))
   
```

    False



```python
R = 10000

# Tip:
# - 统计有多少个点落在单位圆内
# - 计算 落在单位圆内点数 的比例 

# inside = []
# for i in range(R):
#     point = [rand(), rand()]
#     # append inside here!
#     inside.append(in_circle(point))

# print(sum(inside) / R)
```


```python
# 计算误差

# print(sum(inside) / R - math.pi/4 )
```

# 项目3： 滑动平均
一串数据可能包含很多毛刺（由噪声引起），但其实这串数据在现实生活中是平滑的。其中一个平滑数据的操作便是滑动平均。


```python
import random

random.seed(1)

def moving_window_average(x, n_neighbors=1):
    n = len(x)
    width = n_neighbors*2 + 1
    x = [x[0]]*n_neighbors + x + [x[-1]]*n_neighbors
    # To complete the function,
    # return a list of the mean of values from i to i+width for all values i from 0 to n-1.
    li = []
    for i in range(n):
        mean = 0
        for j in range(width):
            mean += x[i+j]
        mean = mean/width
        li.append(mean)
    return li    
        
x=[0,10,5,3,1,5]
print(moving_window_average(x, 1))
```

    [3.3333333333333335, 5.0, 6.0, 3.0, 3.0, 3.6666666666666665]



```python
import random

random.seed(1) # This line fixes the value called by your function,
               # and is used for answer-checking.
    
# write your code here!
# x = []
# for i in range(1000):
#     x.append(random.random())

x = [random.random() for i in range(1000)]
Y = [x]
for i in range(1,10):
    Y.append(moving_window_average(x,i))

```


```python
# write your code here!
ranges = []
for li in Y:
    ranges.append((max(li))-min(li))
print(ranges)

```

    [0.9973152343362711, 0.9128390185520854, 0.801645771909397, 0.7137391224212468, 0.6230146948375028, 0.5042284086774562, 0.5071013753101629, 0.4590090496908159, 0.44659549539083265, 0.4433696944090051]

