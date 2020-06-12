##=================================================
#习题1:已知二维矢量a,b,c． 求a与c的夹角，b与c的夹角
##=================================================
import math

#用列表来实现矢量
a = [4, 0]
b = [-4, 0]
c = [4, 3]

#定义矢量之点积
def dot(vec1, vec2):
    res = 0
    for i in range(len(vec1)):
        res += vec1[i] * vec2[i]
    return res

#定义矢量之模(norm)
def norm(vec):
    s = sum([vec[i]*vec[i] for i in range(len(vec))])
    return math.sqrt(s)

#计算矢量a,c之间的夹角
cos_theta1 = dot(a,c)/(norm(a)*norm(c))
print("ac夹角的余弦:",cos_theta1)
theta1= math.acos(cos_theta1)
print("a与c的夹角(rad):",theta1)

#单位换算
theta1_deg = theta1 * 180/math.pi
print("a与c的夹角(°)：",theta1_deg)
