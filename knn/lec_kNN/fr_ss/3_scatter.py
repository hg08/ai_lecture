# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 14:42:52 2018

@author: huang
"""

"""
1. 课程内容
1.1	   了解人脸校正和编码的基本过程.
1.2.	K-近邻算法.
1.3.	学习使用mglearn, skimage, matplotlib, numpy等Python库.

"""
# 训练集{X, y}. 展示其特正的散点图
import mglearn
import numpy as np
import matplotlib.pyplot as plt


# 生成数据集: 身高和体重.
X = np.array([[1.55, 41],[1.56, 48],[1.6, 51],[1.7, 65], [1.72, 60],[1.8, 72]])
y = np.array([1, 1, 1, 0, 0, 0])

# 作图
mglearn.discrete_scatter(X[:,0], X[:,1],y)
#plt.scatter([1.55, 1.56, 1.6, 1.7, 1.72, 1.80],[41,48,51,65,60,72], c='b', marker='v')
plt.legend(["Men", "Women"],loc = 4)
plt.xlabel("Height")
plt.ylabel("Weight")

print(X)
print("X.shape:{}".format(X.shape))
plt.show()



#已知的点 [1, 0, 0.5] 离数据集中哪一个点最近?
from sklearn.neighbors import NearestNeighbors
s = [[0,0.5, 0], [0,0,0], [1,1,5],[10,10,0]]
neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(s)
print(neigh.kneighbors([[1,0,0.5]], return_distance=True))

X = [[0,5,6],[8,7,0]]
print(neigh.kneighbors(X, return_distance=True))


