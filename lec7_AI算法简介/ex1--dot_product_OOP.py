##=================================================
#习题1:已知二维矢量a,b,c． 求a与c的夹角，b与c的夹角
##=================================================
import math
import numpy as np
#实现矢量
    #学生代码

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def set(self, x, y):
        self.x = x
        self.y = y

    def calNorm(self):
        return math.sqrt(self.x**2 + self.y**2)

    def getArray(self):
        return np.array([self.x, self.y])


#定义矢量之点积
def dot(vec1, vec2):
    #学生代码
    return vec1.x*vec2.x + vec1.y*vec2.y

#定义矢量之模(norm)
def norm(vec):
    #学生代码
    return vec.calNorm()

#计算矢量a,c之间的夹角
def calAngle(vec1,vec2):
    #学生代码
    return math.acos (dot(vec1,vec2)/(vec1.calNorm()*vec2.calNorm()))

#单位换算
def rad2deg(rad):
    #学生代码
    return 180.*rad/math.pi

if __name__ == '__main__':
    vec1 = Vector(4,0)
    vec2 = Vector(-4,0)
    vec3 = Vector(4,3)
    print("vec1 = {},\nvec2 = {}, \nvec3 = {}".format(vec1.getArray(),
          vec2.getArray(), vec3.getArray()))
    print("dot(vec1,vec2) = {}".format(dot(vec1,vec2)))
    print("|vec1| = {},\n|vec2| = {}, \n|vec3| = {}".format(vec1.calNorm(),
          vec2.calNorm(), vec3.calNorm()))
    print("calAngle(vec1, vec2) = {},\ncalAngle(vec2, vec3) = {}, \
          \ncalAngle(vec3, vec21) = {}".format(calAngle(vec1,vec2),
          calAngle(vec2,vec3),calAngle(vec3,vec1)))

    print("Angle(vec1, vec2) = {},\nAngle(vec2, vec3) = {}, \
          \nAngle(vec3, vec21) = {}".format(rad2deg(calAngle(vec1,vec2)),
          rad2deg(calAngle(vec2,vec3)),rad2deg(calAngle(vec3,vec1))))