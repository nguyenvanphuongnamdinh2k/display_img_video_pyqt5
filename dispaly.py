from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QFont, QStandardItemModel, QStandardItem, QIcon, QPixmap
from PyQt5.QtWidgets import QMainWindow, QListView, QVBoxLayout, QLabel
import sys
from phuong import Ui_MainWindow
import cv2
from PyQt5.QtCore import QTimer, QTime, QDateTime, Qt, QMimeDatabase
import os

class display(QMainWindow):
    def __init__(self):
        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)
        #######################################################
        self.timer_video = QTimer()
        self.timer_video.timeout.connect(self.show_video_frame)
        #####################################################
        self.path_save = "img"
        self.kiemtra()
        #############################################################
        self.file_system_model = QStandardItemModel(self.uic.list_img)
        self.uic.list_img.setModel(self.file_system_model)
        self.timer_list_img = QTimer()
        self.timer_list_img.timeout.connect(self.displayImages)
        self.timer_list_img.start(1000)
        ################################################################
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        # Hiển thị QListView trong giao diện
        layout = QVBoxLayout()
        layout.addWidget(self.uic.list_img)
        layout.addWidget(self.image_label)
        self.setLayout(layout)
        ##############################################################
        self.uic.IMGNG.clicked.connect(self.file_img_ng)
        ##################################################
        self.dem = 0
        self.cap = cv2.VideoCapture()
        self.uic.name_folder.setFont(QFont("MS Shell Dlg 2", 15))
        self.uic.name_folder.insertHtml('<b>'+self.path_save+'</b>')
        self.uic.name_folder.setAlignment(Qt.AlignCenter)
        self.uic.button_img.clicked.connect(self.show_img)
        self.uic.button_video.clicked.connect(self.show_video)
        self.uic.butto_camera.clicked.connect(self.camera)
        self.uic.capture.clicked.connect(self.capture)
        self.uic.list_img.doubleClicked.connect(self.displaySelectedImage)
        self.uic.hour.setDigitCount(8) # 8 cột
        self.uic.day.setDigitCount(19)  #19 cột
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_lcd)
        self.timer.start(1000)  # Cập nhật mỗi giây
    def file_img_ng(self):
        img_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "open images", self.path_save, "*.jpg;;*.png;;All Files(*)")
        img = cv2.imread(img_name)
        cv2.imshow(str(img_name),img)
        cv2.waitKey(0)
    def displayImages(self):
        # Lấy danh sách các file trong folder
        files = os.listdir(self.path_save)
        self.file_system_model.clear()
        # Hiển thị các file ảnh trong model
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                item = QStandardItem(QIcon(self.path_save + "/" + filename), filename)
                item.setData(self.path_save + "/" + filename, Qt.UserRole)
                self.file_system_model.appendRow(item)

    def kiemtra(self):
        if not os.path.exists(self.path_save):
            os.mkdir(self.path_save)
    def update_lcd(self):
        # Lấy thời gian hiện tại
        current_time = QTime.currentTime()
        # Chuyển đổi thời gian thành chuỗi "hh:mm:ss"
        time_str = current_time.toString('hh:mm:ss')
        # Hiển thị thời gian trên QLCDNumber
        self.uic.hour.display(time_str)
        now = QDateTime.currentDateTime()  # Lấy thời gian hiện tại
        self.uic.day.display(now.toString('dd-MM-yyyy hh:mm:ss'))  # Hiển thị ngày và thời gian trên QLCDNumber
    def show_img(self):
        img_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "open images", "", "*.jpg;;*.png;;All Files(*)")
        if not img_name:
            return
        else:
            img = cv2.imread(img_name)
            doc,ngang,_= img.shape
            self.result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            self.QtImg = QtGui.QImage(
                self.result.data, self.result.shape[1], self.result.shape[0], QtGui.QImage.Format_RGB32)
            self.uic.display.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
    def show_video(self):
        video,_ = QtWidgets.QFileDialog.getOpenFileName(None,"open video","","*.mp4;;All File(*)")
        if not video:
            return
        else:
            self.cap.open(video)
            self.timer_video.start(1)
    def camera(self):
        flag = self.cap.open(0)
        if flag ==False:
            return
        else:
            self.timer_video.start(1)
    def capture(self):
        self.dem+=1
        ret,frame = self.cap.read()
        cv2.imwrite(os.path.join(self.path_save, str(self.dem) + '.jpg'), frame)
    def show_video_frame(self):
        flag, img = self.cap.read()
        self.result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                 QtGui.QImage.Format_RGB888)
        self.uic.display.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def displaySelectedImage(self, index):
        # Lấy đường dẫn đến file ảnh được chọn
        path = self.file_system_model.index(index.row(), 0, index.parent()).data(Qt.UserRole)
        # Kiểm tra nếu file là ảnh thì hiển thị lên màn hình
        if os.path.isfile(path) and (path.endswith(".jpg") or path.endswith(".png")):
            img = cv2.imread(path)
            cv2.imshow(path,img)
            cv2.waitKey(0)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_win = display()
    main_win.show()
    sys.exit(app.exec_())
