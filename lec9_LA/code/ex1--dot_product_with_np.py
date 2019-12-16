##=================================================
#习题1:已知二维矢量a,b,c． 求a与c的夹角，b与c的夹角
##=================================================
import numpy as np
import math

#用np数组来实现矢量
a = np.array([4, 0])
b = np.array([-4, 0])
c = np.array([4,3])

#定义矢量之点积(直接用现成的函数np.dot)
ac = np.dot(a,c)
bc = np.dot(b,c)

#定义矢量之模(norm)
def norm(vec):
    s = sum([vec[i]*vec[i] for i in range(len(vec))])
    return math.sqrt(s)

#计算矢量a,c之间的夹角
cos_theta1 = np.dot(a,c)/(norm(a)*norm(c))
print("ac夹角的余弦:",cos_theta1)

theta1= math.acos(cos_theta1)
print("a与c的夹角(rad):",theta1)

theta1_deg = theta1 * 180/math.pi
print("a与c的夹角(°)：",theta1_deg)
