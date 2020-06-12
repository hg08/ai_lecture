#导入模块
import numpy as np

#创建矢量
  #学生代码
a = np.mat([0,1])
b = np.mat([[0],[1]])
Z = np.mat([[1,0],[0, -1]])

#计算矩阵乘法
  #学生代码
print("AB=",a*b)
print("Zc=",Z*b)
