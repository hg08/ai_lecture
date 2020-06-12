# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:00:36 2018
@author: huang
"""
from skimage import io
import dlib
"""参考: http://dlib.net/python/"""

"""
1.获取图片"""
file_name = "li_00.jpg"
image = io.imread(file_name)

"""
2. 创建HOG人脸探测器"""
detector = dlib.get_frontal_face_detector()

"""
3.对图片运行探测器"""
detected_faces = detector(image, 1)
print("发现{}张人脸,于{}图片.".format(len(detected_faces), file_name))
"""
5. 导入训练好的人脸标识模型"""
model = "shape_predictor_68_face_landmarks.dat"
"""http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"""
"""
4. 提取68点特征"""
predictor = dlib.shape_predictor(model)
#6.显示在窗口
win = dlib.image_window()
win.set_image(image)
#7.对每张人脸做同样的操作
for box in detected_faces:
    win.add_overlay(box)
    landmarks = predictor(image, box)
    print(landmarks)
    win.add_overlay(landmarks)