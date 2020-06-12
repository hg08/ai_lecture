
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

# 5. 遍历照片中的每一张人脸, 画出人脸边界盒子
for i, box in enumerate(detected_faces):

    # Detected faces are returned as an object with the coordinates
    # of the top, left, right and bottom edges
    print("- 人脸 No.{}, 位置: {} Top: {} Right: {} Bottom: {}".format(i, box.left(), box.top(),
          box.right(), box.bottom()))

    # 在发现的每张人脸周围画一个盒子
    win.add_overlay(box)
    # 保持图像
    dlib.hit_enter_to_continue()
