from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QFont, QStandardItemModel, QStandardItem, QIcon, QPixmap
from PyQt5.QtWidgets import QMainWindow, QListView, QVBoxLayout, QLabel
import sys
from phuong import Ui_MainWindow
import cv2
from PyQt5.QtCore import QTimer, QTime, QDateTime, Qt, QMimeDatabase
import os
import sys
import cv2
import argparse

import torch
import numpy as np
import torch.backends.cudnn as cudnn
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from PyQt5 import QtCore, QtGui, QtWidgets
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_boxes, check_imshow
from utils.augmentations import letterbox
from utils.plots import Annotator,colors
import socket
from models.common import DetectMultiBackend
class display(QMainWindow):
    def __init__(self):
        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)
        #######################################################

        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str,
                            default='yolov5x.pt', help='model.pt path(s)')
        # file/folder, 0 for webcam
        # rtsp://admin:admin111@192.168.1.108:554/cam/realmonitor?channel=1&subtype=00
        parser.add_argument('--source', type=str,
                            default=0, help='source')
        parser.add_argument('--img-size', type=int,
                            default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float,
                            default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float,
                            default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='',
                            help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument(
            '--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true',
                            help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true',
                            help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true',
                            help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, default='0',
                            help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument(
            '--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true',
                            help='augmented inference')
        parser.add_argument('--update', action='store_true',
                            help='update all models')
        parser.add_argument('--project', default='runs/detect',
                            help='save results to project/name')
        parser.add_argument('--name', default='exp',
                            help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true',
                            help='existing project/name ok, do not increment')
        self.opt = parser.parse_args()
        print(self.opt)

        source, weights,  save_txt, imgsz = self.opt.source, self.opt.weights, self.opt.save_txt, self.opt.img_size

        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        cudnn.benchmark = True
        # Load model
        self.model = model = DetectMultiBackend(weights, device=self.device, dnn=None, data='data/coco128.yaml',
                                                fp16=False)
        stride, names, pt = model.stride, model.names, model.pt
        self.imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16
        # Get names and colors
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names



        ############################################################################
        self.path_save = "img"
        self.path_save_camera_thuong = "img_camera_thuong"
        self.kiemtra()
        #############################################################
        self.file_system_model = QStandardItemModel(self.uic.list_img)
        self.uic.list_img.setModel(self.file_system_model)
        self.timer_list_img = QTimer()
        self.timer_list_img.timeout.connect(self.displayImages)
        self.timer_list_img.start(1000)
        self.uic.list_img.doubleClicked.connect(self.displaySelectedImage)
        ################################################################
        self.file_system_model_camera_thuong = QStandardItemModel(self.uic.list_img_camer_thuong)
        self.uic.list_img_camer_thuong.setModel(self.file_system_model_camera_thuong)
        self.timer_list_img_camera_thuong = QTimer()
        self.timer_list_img_camera_thuong.timeout.connect(self.displayImages_camera_thuong)
        self.timer_list_img_camera_thuong.start(1000)
        self.uic.list_img_camer_thuong.doubleClicked.connect(self.displaySelectedImage_camera_thuong)
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
        self.dem_camera_thuong = 0
        self.cap = cv2.VideoCapture()
        self.timer_video = QTimer()
        self.timer_video.timeout.connect(self.show_video_frame)
        self.uic.name_folder.setFont(QFont("MS Shell Dlg 2", 15))
        self.uic.name_folder.insertHtml('<b>'+self.path_save+'</b>')
        self.uic.name_folder.setAlignment(Qt.AlignCenter)
        self.uic.button_img.clicked.connect(self.show_img)
        self.uic.actionfile_img.triggered.connect(self.show_img)
        self.uic.actionfile_video.triggered.connect(self.show_video)
        self.uic.button_video.clicked.connect(self.show_video)
        self.uic.butto_camera.clicked.connect(self.camera)
        self.uic.capture.clicked.connect(self.capture)

        self.uic.hour.setDigitCount(8) # 8 cột
        self.uic.day.setDigitCount(19)  #19 cột
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_lcd)
        self.timer.start(1000)  # Cập nhật mỗi giây

        ##########################################################
        self.uic.camer_thuong.clicked.connect(self.camera_thuong)
        self.timer_video_thuong = QTimer()
        self.timer_video_thuong.timeout.connect(self.show_camera_thuong)
        self.uic.capture_thuong.clicked.connect(self.chup_anh_detect)
    def file_img_ng(self):
        img_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "open images", self.path_save, "*.jpg;;*.png;;All Files(*)")
        if not img_name:
            return
        else:
            img = cv2.imread(img_name)
            cv2.imshow(str(img_name),img)
            cv2.waitKey(0)


    def kiemtra(self):
        if not os.path.exists(self.path_save):
            os.mkdir(self.path_save)
        else:
            for file in os.listdir(self.path_save):
                os.remove(os.path.join(self.path_save,file))
        if not os.path.exists(self.path_save_camera_thuong):
            os.mkdir(self.path_save_camera_thuong)
        else:
            for file in os.listdir(self.path_save_camera_thuong):
                os.remove(os.path.join(self.path_save_camera_thuong,file))
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
        img, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "open images", "", "*.jpg;;*.png;;All Files(*)")
        if not img:
            print("k co anh")
        else:
            img = cv2.imread(img)
            showing = img
            annotator = Annotator(im=showing, line_width=3)
            with torch.no_grad():
                img = letterbox(img, new_shape=self.opt.img_size)[0]
                # Convert
                # BGR to RGB, to 3x416x416
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                pred = self.model(img, augment=self.opt.augment)[0]
                # Apply NMS
                pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                           agnostic=self.opt.agnostic_nms)
                for i, det in enumerate(pred):
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(
                            img.shape[2:], det[:, :4], showing.shape).round()

                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            annotator.box_label(xyxy, label=label,
                                                color=colors(int(cls),True))
            self.result = cv2.cvtColor(showing, cv2.COLOR_BGR2BGRA)
            self.result = cv2.resize(self.result,(899,579))
            self.QtImg = QtGui.QImage(
                self.result.data, self.result.shape[1], self.result.shape[0], QtGui.QImage.Format_RGB32)
            self.uic.display.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
    def show_video(self):
        if not self.timer_video.isActive():
            video,_ = QtWidgets.QFileDialog.getOpenFileName(None,"open video","","*.mp4;;All File(*)")
            if not video:
                return
            else:
                self.cap.open(video)
                self.timer_video.start(1)
                self.uic.button_video.setText(u'turn off video')
                self.uic.butto_camera.setDisabled(True)
                self.uic.button_img.setDisabled(True)
        else:
            self.timer_video.stop()
            self.cap.release()
            self.uic.display.clear()
            self.uic.button_video.setText(u'turn on video')
            self.uic.butto_camera.setDisabled(False)
            self.uic.button_img.setDisabled(False)
    def camera(self):
        if not self.timer_video.isActive():
            flag = self.cap.open(0)

            if flag ==False:
                return
            else:
                self.timer_video.start(1)
                self.uic.butto_camera.setText(u"turn off camera")
                self.uic.button_video.setDisabled(True)
                self.uic.button_img.setDisabled(True)
        else:
            self.timer_video.stop()
            self.cap.release()
            self.uic.display.clear()
            self.uic.butto_camera.setText(u'turn on camera')
            self.uic.button_video.setDisabled(False)
            self.uic.button_img.setDisabled(False)


    def capture(self):
        self.dem+=1
        cv2.imwrite(os.path.join(self.path_save, str(self.dem) + '.jpg'), self.showimg)
    def show_video_frame(self):
        # print("aaaaaaaaaaaaaaa")
        flag, img = self.cap.read()
        # print(f"img : {img}")
        if img is not None:
            self.showimg = img
            annotator = Annotator(self.showimg, line_width=3)
            with torch.no_grad():
                img = letterbox(img, new_shape=self.opt.img_size)[0]
                # Convert
                # BGR to RGB, to 3x416x416
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                pred = self.model(img, augment=self.opt.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=None,
                                           agnostic=self.opt.agnostic_nms)
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(
                            img.shape[2:], det[:, :4], self.showimg.shape).round()
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            # print(label)
                            annotator.box_label(
                                xyxy, label=label, color=colors(int(cls), True))
            self.result = cv2.cvtColor(self.showimg, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                     QtGui.QImage.Format_RGB888)
            self.uic.display.setPixmap(QtGui.QPixmap.fromImage(showImage))
    #################################################################################################
    def displayImages(self):
        # Lấy danh sách các file trong folder
        files = os.listdir(self.path_save)
        self.file_system_model.clear() # xóa dữ liệu cũ
        # Hiển thị các file ảnh trong model
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                item = QStandardItem(QIcon(self.path_save + "/" + filename), filename)
                item.setData(self.path_save + "/" + filename, Qt.UserRole)
                self.file_system_model.appendRow(item)

    def displaySelectedImage(self, index):
        # Lấy đường dẫn đến file ảnh được chọn
        path = self.file_system_model.index(index.row(), 0, index.parent()).data(Qt.UserRole)
        if not path:
            return
        else:
        # Kiểm tra nếu file là ảnh thì hiển thị lên màn hình
            if os.path.isfile(path) and (path.endswith(".jpg") or path.endswith(".png")):
                img = cv2.imread(path)
                cv2.imshow(path,img)
                cv2.waitKey(0)

    #################################################################################################
    def displayImages_camera_thuong(self):
        files = os.listdir(self.path_save_camera_thuong)
        self.file_system_model_camera_thuong.clear()  # xóa dữ liệu cũ
        # Hiển thị các file ảnh trong model
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                item = QStandardItem(QIcon(self.path_save_camera_thuong + "/" + filename), filename)
                item.setData(self.path_save_camera_thuong + "/" + filename, Qt.UserRole)
                self.file_system_model_camera_thuong.appendRow(item)

    def displaySelectedImage_camera_thuong(self, index):
        # Lấy đường dẫn đến file ảnh được chọn
        path = self.file_system_model_camera_thuong.index(index.row(), 0, index.parent()).data(Qt.UserRole)
        if not path:
            return
        else:
            # Kiểm tra nếu file là ảnh thì hiển thị lên màn hình
            if os.path.isfile(path) and (path.endswith(".jpg") or path.endswith(".png")):
                img = cv2.imread(path)
                cv2.imshow(path, img)
                cv2.waitKey(0)
    def camera_thuong(self):
        if not self.timer_video_thuong.isActive():
            flag = self.cap.open(0)

            if flag == False:
                return
            else:
                self.timer_video_thuong.start(1)
                self.uic.camer_thuong.setText(u"turn off camera")
                self.uic.button_video.setDisabled(True)
                self.uic.button_img.setDisabled(True)
        else:
            self.timer_video_thuong.stop()
            self.cap.release()
            self.uic.display.clear()
            self.uic.camer_thuong.setText(u'turn on camera')
            self.uic.button_video.setDisabled(False)
            self.uic.button_img.setDisabled(False)

    def show_camera_thuong(self):
        ok,self.frame = self.cap.read()
        self.result = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                 QtGui.QImage.Format_RGB888)
        self.uic.display.setPixmap(QtGui.QPixmap.fromImage(showImage))
    def chup_anh_detect(self):
        if self.frame is not None:
            self.dem_camera_thuong +=1
            self.showimg_camera_thuong = self.frame
            annotator = Annotator(self.showimg_camera_thuong, line_width=3)
            with torch.no_grad():
                img = letterbox(self.frame, new_shape=self.opt.img_size)[0]
                # Convert
                # BGR to RGB, to 3x416x416
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                pred = self.model(img, augment=self.opt.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=None,
                                           agnostic=self.opt.agnostic_nms)
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(
                            img.shape[2:], det[:, :4], self.showimg_camera_thuong.shape).round()
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            # print(label)
                            annotator.box_label(
                                xyxy, label=label, color=colors(int(cls), True))
            cv2.imwrite(os.path.join(self.path_save_camera_thuong,str(self.dem_camera_thuong)+'.jpg'),self.showimg_camera_thuong)
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_win = display()
    main_win.show()
    sys.exit(app.exec_())
