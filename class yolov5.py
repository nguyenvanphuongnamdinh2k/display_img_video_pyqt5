import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
import torch.backends.cudnn as cudnn

class yolov5:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt',
                            help='model path or triton URL')
        parser.add_argument('--source', type=str, default=ROOT / 'data/images',
                            help='file/dir/URL/glob/screen/0(webcam)')
        parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                            help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='show results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
        self.opt = parser.parse_args()
        source, weights, view_img, save_txt, imgsz = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.imgsz
        nosave,project,name,exist_ok,device,dnn,data,half,vid_stride,augment =  self.opt.nosave, self.opt.project, self.opt.name, \
                                                                                self.opt.exist_ok, self.opt.device, self.opt.dnn, \
                                                                                self.opt.data, self.opt.half, self.opt.vid_stride, self.opt.augment
        conf_thres,iou_thres,classes,agnostic_nms,max_det = self.opt.conf_thres,self.opt.iou_thres,self.opt.classes,self.opt.agnostic_nms,self.opt.max_det
        save_crop ,line_thickness ,hide_labels, hide_conf ,save_conf = self.opt.save_crop ,self.opt.line_thickness ,self.opt.hide_labels, self.opt.hide_conf,self.opt.save_conf
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        self.webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        self.save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (self.save_dir / 'labels' if save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        self.model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = [640, 640]  # check image size

        # Dataloader
        self.bs = 1  # batch_size
        if self.webcam:
            view_img = check_imshow(warn=True)
            self.dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=self.pt, vid_stride=vid_stride)
            bs = len(self.dataset)
        elif screenshot:
            self.dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=self.pt)
        else:
            self.dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=self.pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * self.bs, [None] * self.bs

    def detect(self):
        # Run inference
        imgsz = [640, 640]
        print(f'imgsz:{imgsz}')
        print(f"model :{self.model}")
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else self.bs, 3, *imgsz))  # warmup
        print("aaaaaaaaaaaaaaa")
        self.seen, self.windows, self.dt = 0, [], (Profile(), Profile(), Profile())

        for path, im, im0s, vid_cap, s in self.dataset:

            with self.dt[0]:
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with self.dt[1]:
                visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = self.model(im, augment=self.opt.augment, visualize=visualize)

            # NMS
            with self.dt[2]:
                pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, self.opt.classes, self.opt.agnostic_nms, max_det=self.opt.max_det)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                self.seen += 1
                if self.webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), self.dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(self.dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(self.opt.save_dir / p.name)  # im.jpg
                txt_path = str(self.opt.save_dir / 'labels' / p.stem) + (
                    '' if self.dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if self.opt.save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=self.opt.line_thickness, example=str(self.model.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if self.opt.save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.opt.save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if self.opt.save_img or self.opt.save_crop or self.opt.view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if self.opt.hide_labels else (self.model.names[c] if self.opt.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if self.opt.save_crop:
                            save_one_box(xyxy, imc, file= self.opt.save_dir / 'crops' / self.model.names[c] / f'{p.stem}.jpg', BGR=True)

                # Stream results
                im0 = annotator.result()
                if  self.opt.view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond
if __name__ == '__main__':
    yolo = yolov5()
    yolo.detect()