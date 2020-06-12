#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate cos_similarity
"""
# 导入模块
import numpy as np

# 实现表示文档A,B,C的矢量a,b,c
v1 = np.array([1,2,0,2,1])
v2 = np.array([1,3,0,1,3])
v3 = np.array([0,2,0,1,1])

# 计算文档A,B之间的相似度（其实是计算两个矢量夹角之余弦值）
o12=np.dot(v1,v2)/(np.linalg.norm(v1)*(np.linalg.norm(v2)))
o13=np.dot(v1,v3)/(np.linalg.norm(v1)*(np.linalg.norm(v3)))
o23=np.dot(v2,v3)/(np.linalg.norm(v2)*(np.linalg.norm(v3)))

print("o12:",o12)
print("o13:",o13)
print("o23:",o23)

#输出
#0.929669680201
