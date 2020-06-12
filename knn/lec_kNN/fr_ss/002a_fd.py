# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 16:46:09 2018

@author: huang
"""
"""1. 课程内容
  1.1 人脸探测
  1.2 人脸标识
  1.3 skimage和dlib库
2. Dlib库函数.
"""
"""0. 导入库函数"""
from skimage import io
import dlib

"""1. 获取图片, 将图片存进一个数组"""
file_name = "li_00.jpg"
image = io.imread(file_name)

"""2.建立一个HOG人脸探测器．(Dlib)
Python API: http://dlib.net/python/index.html 
返回:探测器对象."""
detector = dlib.get_frontal_face_detector() 

"""3. 在图片上运行人脸探测器"""
detected_faces = detector(image,1)
"""输出一个数组,每个元素是一个人脸对象"""
print("发现{}张人脸,于图片{}.".format(len(detected_faces), file_name))

"""4. 建立窗口对象,放入图片"""
win = dlib.image_window()
win.set_image(image)

"""5. 遍历图片中的每一张脸,画出人脸边界盒子
for循环:"""
for box in detected_faces:
    win.add_overlay(box)    
    dlib.hit_enter_to_continue()
for i, box in enumerate(detected_faces):
    print("第{}张人脸, 位置: {}".format(i,box.left(), box.right() ))
    """每出现一张人脸,就打印一次该人脸的信息"""
    win.add_overlay(box)
    """保持图像"""
    dlib.hit_enter_to_continue()
    




    

 

