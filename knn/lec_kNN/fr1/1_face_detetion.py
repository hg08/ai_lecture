# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 20:14:43 2018

@author: huang
"""

"""
1. 课程内容
1.1  理解机器学习基本概念,监督学习和非监督学习的区别.
1.2  数据集,训练样本. (了解)
1.3	 了解人脸识别的实质(监督学习中的分类算法).
1.4  Hog的计算和Python实现. (了解)
1.5 学习使用skimage, matplotlib, numpy等Python库.

2.
    机器学习, 是指 ”使计算机模拟或实现人类的学习行为，以从数据中获取新的知识或技能，
重新组织已有的知识结构使之不断改善自身的性能”.
     分析: 模拟人类行为
     新的技能
     不断改善
   
3. 数据集概念
训练集, 测试集, 样本

4. 数据集的表示:
数据集: D:{(X^((1) ),y^((1) ) ),…,(X^((N) ),y^((N) ) )}
X, 用来表示特征. 
y, 用来表示标签.

5. 机器学习的分类
监督学习,非监督学习.

6. 分类和回归  
  分类算法的目标是从(已有的)数个可能的分类标签中预测一个样本所属的分类标签.标签具有不连续性.
  回归算法的目标是预测一个连续的数(浮点型).  
    

二. 人脸识别的步骤概述
"""

# 1导入库函数
from skimage import io, color
from skimage.feature import hog
import matplotlib.pyplot as plt


# 2导入图片, 简化处理
image = io.imread("eg.png")
image = color.rgb2gray(image)

"""
3 计算HOG
arry: HOG as a 1-D array
hog_image:  HOG for visualization 可视化HOG图
pixels(像素), cells(方格), block(区间)
区间: bin
"""
array, hog_image = hog(image, visualise=True) 

#4. 作图
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
ax1.imshow(image, cmap=plt.cm.gray)
ax2.imshow(hog_image, cmap=plt.cm.gray)
plt.show()


 















