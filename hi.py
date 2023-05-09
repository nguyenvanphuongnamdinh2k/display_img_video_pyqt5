import sys
import cv2
import argparse
import random
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

class yolov5():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str,
                            default='yolov5s.pt', help='model.pt path(s)')
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


    def run(self):
        img = cv2.imread("data/images/bus.jpg")
        showing = img
        annotator = Annotator(im=showing,line_width=3)
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
                                            color=self.colors[int(cls)])
            cv2.imshow(str(showing),showing)
            cv2.waitKey(0)

    def camera(self):
        self.cap = cv2.VideoCapture(0)
        name_list = []
        while True:
            flag, img = self.cap.read()
            if img is not None:
                showimg = img
                annotator = Annotator(showimg, line_width=3)
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
                                img.shape[2:], det[:, :4], showimg.shape).round()
                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                label = '%s %.2f' % (self.names[int(cls)], conf)

                                name_list.append(self.names[int(cls)])
                                print(label)
                                annotator.box_label(
                                    xyxy, label=label, color=colors(int(cls),True))
                        cv2.imshow("camera",showimg)
                        cv2.waitKey(1)

a = yolov5()
a.camera()
