from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton

class Person(QDialog):
    def __init__(self, *args, **kwargs):
        super(Person, self).__init__(*args, **kwargs)
        self.initUi()
    
    def initUi(self):
        self.resize(450, 300)
        # self.center()
        self.setWindowTitle('hello world')

        main_layout = QVBoxLayout()
        header_layout = QVBoxLayout()
        play_layout = QHBoxLayout()

        # 状态信息
        self.status = QLabel('正在采集人脸', self)

        btn1 = QPushButton('识别开始', self)
        btn2 = QPushButton('结果显示', self)

        btn1.clicked.connect(self.result)

        header_layout.addWidget(self.status)

        play_layout.addWidget(btn1)
        play_layout.addWidget(btn2)

        main_layout.addLayout(header_layout)

        play_layout.addStretch(1)
        main_layout.addLayout(play_layout)


        self.setLayout(main_layout)

        self.show()
    
    def result(self):
        self.status.setText('aaaaa')


if __name__ == '__main__':
    app = QApplication([])

    win = Person()

    app.exec_()


