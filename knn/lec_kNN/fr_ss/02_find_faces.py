
import dlib
from skimage import io

#1. 获取图像名称file_name; 将图片存进一个数组
file_name = "li_00.jpg"
image = io.imread(file_name)

# 2.建立一个HOG人脸探测器. 方法：用内置的dlib类
detector = dlib.get_frontal_face_detector()


# 3.在图片数据上运行HOG人脸探测器.
# 结果：人脸边界盒子.
detected_faces = detector(image,1)
print("在文件{}中发现{}张人脸".format(file_name, len(detected_faces)))

# 4. 建立窗口对象
win = dlib.image_window()
# 在桌面打开一个窗口以显示图片
win.set_image(image)

# 5. 遍历照片中的每一张人脸, 画出人脸边界盒子 (可以用for循环实现)

# detected_faces: 人脸对象的集合,它的每一个元素,就是一个人脸对象(矩形框),
# 它具有边界坐标: top, left, right and bottom. 

for 循环开始: 
    ## 学生代码 ##
    # 要求,打印出人脸序号, 和人脸的上,左,右,下边沿的位置.
    # 例如, 用box 表示detected_faces集合的元素,那么 box的上边缘的位置为 box.top()



    
    #(以下代码也在for循环内部)
    # 在发现的每张人脸周围画一个盒子
    win.add_overlay(box)
    # 保持图像
    dlib.hit_enter_to_continue()
