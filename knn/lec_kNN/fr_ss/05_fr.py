# -*- coding: utf-8 -*-
#file:05_face_recog_knn.py

#===========
#1.导入模块
#===========

import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition as fr
from face_recognition.cli import image_files_in_folder
from sklearn import neighbors

#=========
#2.函数定义
#=========
#训练
def train(train_dir, model_save_path='trained_knn_model.clf', n_neighbors=3,
          knn_algo='ball_tree'):
    """
    训练一个KNN分类器.
    :return: KNN分类器.
    """
    #建立数据集 X,y
    X = []
    y = []
    #对每一个照片操作
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir,class_dir)):
            continue
        #对每一张照片
        for img_path in image_files_in_folder(os.path.join(train_dir,class_dir)):
            image = fr.load_image_file(img_path)
            boxes = fr.face_locations(image)

            #对每张照片编码
            X.append(fr.face_encodings(image,known_face_locations=boxes)[0])
            y.append(class_dir)

    #决定k
    if n_neighbors is None:
        n_neighbors = 3


    #训练出分类器
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_clf.fit(X,y)


    #保存分类器
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf,f)


    return knn_clf


#预测
def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.45):
    """
    利用KNN分离器识别给定照片中的人脸
    :return: [(人名1, 边界盒子1), ...]
    """
    if knn_clf is None and model_path is None:
        raise Exception("必须提供KNN分类器:可选方式为 knn_clf 或 model_path")

    # 加载训练好的KNN模型(如果有)
    # rb 表示要读入二进制数据
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # 加载图片,发现人脸的位置
    X_img = fr.load_image_file(X_img_path)
    X_face_locations = fr.face_locations(X_img)


    # 对测试图片中的人脸编码
    encodings = fr.face_encodings(X_img,
                known_face_locations=X_face_locations)

    # 利用KNN model 找出与测试人脸最匹配的人脸
    # encodings: 128个人脸特征构成的向量
    closest_distances = knn_clf.kneighbors(encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold
                   for i in range(len(X_face_locations))]

    # 预言类别,并 remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc)
            for pred, loc, rec in zip(knn_clf.predict(encodings),
                                      X_face_locations, are_matches)]

# 结果显示出来
def show_names_on_image(img_path, predictions):
    """
    人脸识别可视化.
    :param img_path: 待识别图片的位置
    :param predictions:预测的结果
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # 用Pillow模块画出人脸边界盒子
        draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 255))

        # pillow里可能生成非UTF-8格式,所以这里做如下转换
        name = name.encode("UTF-8")
        name = name.decode("ascii") #L add

        # 在人脸下写下名字,作为标签
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)),
                       fill=(255, 0, 255), outline=(255, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255))

        # 追加名字到列表li_names
        li_names.append(name)

    # 从内存删除draw
    del draw

    # 显示结果图
    pil_image.show()


#========
#统计分析
#========
# 为了打印名字的集合
li_names = []

# 计算总人数
def count(train_dir):
    """
    Counts the total number of the set.
    """
    path = train_dir
    count = 0
    for fn in os.listdir(path): #fn 表示的是文件名
            count = count + 1
    return count

# 获取所有名字的列表
def list_all(train_dir):
    """
    Determine the list of all names.
    """
    path = train_dir
    result = []
    for fn in os.listdir(path): #fn 表示的是文件名
            result.append(fn)
    return result

# 输出结果
def stat_output():
    s_list = set(li_names)
    s_list_all = set(list_all("examples/train"))
    if "unknown" in s_list:
        s_list.remove("unknown")

    tot_num = count("examples/train")
    s_absent = set(s_list_all - s_list)
    print("\n")
    print("==========================*\n")
    print("全体名单:",s_list_all)
    print("已到名单:",s_list)
    print("应到人数:",tot_num)
    print("已到人数:",len(s_list))
    print("出勤率:{:.2f}".format(float(len(s_list))/float(tot_num)))
    print("未到:",s_absent)


# 运行
if __name__ == "__main__":
    #1.训练分类器
    train("examples/train", model_save_path="trained_knn_model.clf",
          n_neighbors=3)
    print("训练完成!")
    #2. 利用训练好的分类器,对新照片进行预测
    for image_file in os.listdir("examples/test"):
        full_file_path = os.path.join("examples/test", image_file)

        # 利用分类器,找出所有人脸
        #def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.35):
        predictions = predict(full_file_path, model_path="trained_knn_model.clf")
        #打印出
        for name, (top,right,bottom,left) in predictions:
            #print("发现{}.位置:{}, 右边:{}.".format(name, top,right))
            print("name")
        #显示出名字
        #def show_names_on_image(img_path, predictions):
        show_names_on_image(os.path.join("examples/test", image_file), predictions)

    #统计结果
    stat_output()









