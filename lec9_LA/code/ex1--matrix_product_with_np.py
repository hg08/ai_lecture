##====================================================================
#习题1:已知二维矢量x=[1,0]^T, 二维旋转矩阵A, 求A与x的乘积和A的逆矩阵：A^{-1}
##====================================================================

#========
#导入模块
#========
import numpy as np
import math

##=======================
#用np数组来实现矩阵和矢量
##=======================
def mat_A(theta):
    c = math.cos(theta)
    s = math.sin(theta)
    A = np.array([[c, -s],[s, c]])
    return A

x = np.array([1, 0])

#============
# 为theta赋值
#============
# 方法１
#theta = 3.1416/6 = 0.524
#theta = 0.524

# 方法2
# theta = input("输入一个角度值:\n")

# 方法3
# 自己定义get()函数，来获取参数(可取默认值)
def get(msg,default=0):
    res = input(msg)
    if res =='':
        res = default
    return res

theta = get("输入一个角度值:\n",default=30)


# 单位转换
theta = (float(theta)/180)*math.pi
print("角度(rad):",theta)

#=================================
#运用np自带矩阵乘法np.dot计算Ax的值
#=================================
matrix_product = np.dot(mat_A(theta),x)

print(matrix_product)
