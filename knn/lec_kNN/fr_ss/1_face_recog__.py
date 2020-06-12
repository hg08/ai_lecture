# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 11:03:19 2018

@author: huang


1.	课程: 第一节. 人脸探测
    1.	理解机器学习基本概念,监督学习和非监督学习的区别.
    2.	数据集,训练样本. (了解)
    3.	了解人脸识别的实质(监督学习中的分类算法).
    4.	Hog的计算和Python实现. (了解)
    5.	学习使用skimage, matplotlib, numpy等Python库.


2. 机器学习
    机器学习, 是指 ”使计算机模拟或实现人类的学习行为，
以从数据中获取新的知识或技能，重新组织已有的知识结构使之
不断改善自身的性能” 这样一种过程.
    计算机, 模拟人类, 不断完善自身.

    Arthur Samuel给出了机器学习最早的定义: 机器学习是”不用具体地编程,
而让计算机具有学习能力的一个研究领域.

3.监督学习和非监督学习

4. 数据集的表示
 X, 特征
 y,标签

5. 监督学习的流程

6. 分类和回归算法
    分类算法的目标是从(已有的)数个可能的分类标签中预测一个样本所属的分类标签.
    回归算法的目标是预测一个连续的数(浮点型).

7. 人脸识别的步骤
  7.1 人脸探测
 举例. 用skimage 来计算图像的HOG(方向梯度直方图).

"""
#1. 导入库函数
from skimage import io, color
from skimage.feature import hog
import matplotlib.pyplot as plt


#2. 导入图片
image = io.imread("eg.png")
image = color.rgb2gray(image)


#3. 计算HOG
#hog()返回值
# 1-array HOG
# hog_image (可用于显示HOG图)
arr, hog_image = hog(image,visualise=True, orientations=4)

#4. 作图
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(8,4))
ax1.imshow(image, cmap=plt.cm.gray)
ax2.imshow(hog_image,cmap=plt.cm.gray)
plt.show()









