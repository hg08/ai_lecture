# encoding=utf8
##coding:utf-8
#======================
#Author: Huang Gang
#Date: 2018/03/30
#======================



# for solving the "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xce..."
import sys
#  because of the Illusive setdefaultencoding function. It is deleted at Python
# startup since it should never have been a part of a proper release in the
# first place, apparently

#reload(sys)
#sys.setdefaultencoding('utf8')



#=================
#imporpted modules
#=================
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
#from face_recognition.face_recognition_cli import image_files_in_folder
from face_recognition.cli import image_files_in_folder
import cv2
import shutil
import numpy as np
import time

# for detecting encoding type of strings
import chardet

# qt related module
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton

# QtWidgets模块包含创造经典桌面风格的用户界面提供了一套UI元素的类.

class Person(QDialog):
    def __init__(self, *args, **kwargs):
        super(Person, self).__init__(*args, **kwargs)
        self.initUi()

    def initUi(self):
        self.resize(450, 300)
        # self.center()
        self.setWindowTitle('人脸识别')

        main_layout = QVBoxLayout()
        header_layout = QVBoxLayout()
        play_layout = QHBoxLayout()

        # 状态信息
        self.status = QLabel('正在采集人脸', self)

        btn1 = QPushButton('开始', self)
        btn2 = QPushButton('结果显示', self)
        btn3 = QPushButton('退出', self)

        btn1.clicked.connect(self.result)
        btn2.clicked.connect(lambda:face_recog.run(5))
        btn3.clicked.connect(face_recog.logout)

        header_layout.addWidget(self.status)

        play_layout.addWidget(btn1)
        play_layout.addWidget(btn2)
        play_layout.addWidget(btn3)

        main_layout.addLayout(header_layout)

        play_layout.addStretch(1)
        main_layout.addLayout(play_layout)


        self.setLayout(main_layout)

        self.show()

    def result(self):
        self.status.setText('aaabbb')


# the class face_recog
class face_recog():
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

    #========================
    #Definitions of functions
    #========================
    # To count the total number of people
    def count(train_dir):
        """
        Counts the total number of the set.
        """
        path = train_dir
        count = 0
        for fn in os.listdir(path): #fn 表示的是文件名
                count = count + 1
        return count

    # To obtain the list of all names
    def list_all(train_dir):
        """
        Determine the list of all names.
        """
        path = train_dir
        result = []
        for fn in os.listdir(path): #fn 表示的是文件名
                result.append(fn)
        return result

    def train(train_dir, model_save_path='trained_knn_model.clf', n_neighbors=3, knn_algo='ball_tree', verbose=False):
        """
        Trains a k-nearest neighbors classifier for face recognition.
        :param train_dir: directory that contains a sub-directory for each known person, with its name.
         (View in source code to see train_dir example tree structure)
        :param model_save_path: (optional) path to save model on disk
        :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
        :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
        :param verbose: verbosity of training
        :return: returns knn classifier that was trained on the given data.
        """
        X = []
        y = []

        # Loop through each person in the training set
        for class_dir in os.listdir(train_dir):
            if not os.path.isdir(os.path.join(train_dir, class_dir)):
                continue

            # Loop through each training image for the current person
            for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
                image = face_recognition.load_image_file(img_path)
                face_bounding_boxes = face_recognition.face_locations(image)

                if len(face_bounding_boxes) != 1:
                    # If there are no people (or too many people) in a training image, skip the image.
                    if verbose:
                        print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
                else:
                    # Add face encoding for current image to the training set
                    X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                    y.append(class_dir)

        # Determine how many neighbors to use for weighting in the KNN classifier
        if n_neighbors is None:
            n_neighbors = int(round(math.sqrt(len(X))))
            if verbose:
                print("Chose n_neighbors automatically:", n_neighbors)

        # Create and train the KNN classifier
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
        knn_clf.fit(X, y)

        # Save the trained KNN classifier
        if model_save_path is not None:
            with open(model_save_path, 'wb') as f:
                pickle.dump(knn_clf, f)

        return knn_clf


    def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.35):
        """
        Recognizes faces in given image using a trained KNN classifier
        :param X_img_path: path to image to be recognized
        :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
        :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
        :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance of mis-classifying an unknown person as a known one. L: If the train set is large enough, we can reduce the value of distance_thredhold.
        :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
            For faces of unrecognized persons, the name 'unknown' will be returned.
        """
        if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in face_recog.ALLOWED_EXTENSIONS:
            raise Exception("Invalid image path: {}".format(X_img_path))

        if knn_clf is None and model_path is None:
            raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

        # Load a trained KNN model (if one was passed in)
        if knn_clf is None:
            with open(model_path, 'rb') as f:
                knn_clf = pickle.load(f)

        # Load image file and find face locations
        X_img = face_recognition.load_image_file(X_img_path)
        X_face_locations = face_recognition.face_locations(X_img)

        # If no faces are found in the image, return an empty result.
        if len(X_face_locations) == 0:
            return []

        # Find encodings for faces in the test iamge
        faces_encodings = face_recognition.face_encodings(X_img,
                    known_face_locations=X_face_locations)

        # Use the KNN model to find the best matches for the test face
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in
                       range(len(X_face_locations))]

        # Predict classes and remove classifications that aren't within the threshold
        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
                zip(knn_clf.predict(faces_encodings), X_face_locations,
                are_matches)]

    # For printing the set of names
    li_names = []

    def show_prediction_labels_on_image(img_path, predictions):
        """
        Shows the face recognition results visually.
        :param img_path: path to image to be recognized
        :param predictions: results of the predict function
        :return:
        """
        pil_image = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(pil_image)

        for name, (top, right, bottom, left) in predictions:
            # Draw a box around the face using the Pillow module
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

            # There's a bug in Pillow where it blows up with non-UTF-8 text
            # when using the default bitmap font

            name =  bytes(name, encoding = "utf8")

            en_coding = chardet.detect(name)
            en_coding = en_coding['encoding']
            if en_coding=="ascii":
                print("Eng is detected")
            else:
                print("Other is detected")
                name = str(name, encoding = "utf-8")
                #name = name.encode("UTF-8")
                #name = name.decode("utf-8") #L add

            # Draw a label with a name below the face
            text_width, text_height = draw.textsize(name)
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
            print(type(name),name)
            if type(name) == 'numpy.str_':
                draw.text((left + 6, bottom - text_height - 5), np.unicode_(name), fill=(255, 255, 255, 255))
            else:
                draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

            # Append name to the list
            face_recog.li_names.append(name)

        # Remove the drawing library from memory as per the Pillow docs
        del draw

        # Display the resulting image
        pil_image.show()

    def logout(self):
        #app.exit()
        sys.exit(app.exec_())


    def take_photo(self):
        # to remove the picture "im.png"
        pic_name = "./im.png"
        try:
            os.remove(pic_name)
        except:
            pass

        # to delete im*.png in test directory
        from pathlib import Path
        try:
            for p in Path("./examples/test/").glob("im*.png"):
                p.unlink()
        except:
            pass

        # STEP 0: To get picture from camera
        count_pic = 0
        cap = cv2.VideoCapture(0)
        cap.set(3,640) #设置分辨率
        cap.set(4,480)
        #for i in range(10):
        #    ret,frame = cap.read(0)
        while 1:
            ret,frame = cap.read(0)
            time.sleep(2)
            #if not ret:
            #    continue
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            pil = Image.fromarray(gray)
            height,width = pil.size #The order is essential here
            pix = np.array(pil.getdata()).reshape(pil.size[1], pil.size[0], -1)
            cv2.imwrite(pic_name,pix)
            shutil.copyfile(pic_name,"im{}.png".format(str(count_pic))) # copyfile(src,dst)
            shutil.move( "im{}.png".format(str(count_pic)), "examples/test")
            count_pic = count_pic + 1
            print("拍摄完成!下一位.")
            #time.sleep(1)
            if count_pic > 20:
                break
            # When everything done, release the capture
        print("Releasing the capture ...")
        cap.release()
        #app.exec_()

    #if __name__ == "__main__":
    def run(num_test=3):
        # to remove the picture "im.png"
        pic_name = "./im.png"
        try:
            os.remove(pic_name)
        except:
            pass

        # to delete im*.png in test directory
        from pathlib import Path
        try:
            for p in Path("./examples/test/").glob("im*.png"):
                p.unlink()
        except:
            pass

        # STEP 0: To get picture from camera
        count_pic = 0
        cap = cv2.VideoCapture(0)
        time.sleep(1)
        #for i in range(500):
        #    ret,frame = cap.read(0)
        while 1:
            ret,frame = cap.read(0)
            if not ret:
                continue
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            pil = Image.fromarray(gray)
            height,width = pil.size #The order is essential here
            pix = np.array(pil.getdata()).reshape(pil.size[1], pil.size[0], -1)
            cv2.imwrite(pic_name,pix)
            shutil.copyfile(pic_name,"im{}.png".format(str(count_pic))) # copyfile(src,dst)
            shutil.move( "im{}.png".format(str(count_pic)), "examples/test")
            count_pic = count_pic + 1
            print("拍摄完成!下一位.")
            #Person.status.setText('拍照完成! 下一位.')
            time.sleep(1)
            if count_pic > num_test:
                break
        # When everything done, release the capture
        print("Releasing the capture ...")
        cap.release()
        #print("to destroyallwindows..")
        #cv2.destroyAllwindows()

        # STEP 1: Train the KNN classifier and save it to disk
        # Once the model is trained and saved, you can skip this step next time.
        print("Training KNN classifier...")
        classifier = face_recog.train("examples/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
        print("Training complete!")

        # STEP 2: Using the trained classifier, make predictions for unknown images
        for image_file in os.listdir("examples/test"):
            full_file_path = os.path.join("examples/test", image_file)

            print("Looking for faces in {}".format(image_file))

            # Find all people in the image using a trained classifier model
            # Note: You can pass in either a classifier file name or a classifier model instance
            predictions = face_recog.predict(full_file_path, model_path="trained_knn_model.clf")

            # Print results on the console
            for name, (top, right, bottom, left) in predictions:
                print("- Found {} at ({}, {})".format(name, left, top))

            # Display results overlaid on an image
            print(type(image_file),image_file)
            face_recog.show_prediction_labels_on_image(os.path.join("examples/test", image_file), predictions)

        # Print the statistical data
        s_list = set(face_recog.li_names)
        s_list_all = set(face_recog.list_all("examples/train"))
        if "unknown" in s_list:
            s_list.remove("unknown")

        tot_num = face_recog.count("examples/train")
        s_absent = set(s_list_all - s_list)
        print("\n")
        print("============================\n")
        print("全体名单:",s_list_all)
        print("已到:",s_list)
        print("总人数:",tot_num)
        print("已到人数:",len(s_list))
        print("出勤率:{:.2f}".format(float(len(s_list))/float(tot_num)))
        print("未到:",s_absent)

if __name__ == '__main__':
    app = QApplication([])

    win = Person()

    app.exec_()
