# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 segmentation inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python segment/predict.py --weights yolov5s-seg.pt --source 0                               # webcam
                                                                  img.jpg                         # image
                                                                  vid.mp4                         # video
                                                                  screen                          # screenshot
                                                                  path/                           # directory
                                                                  'path/*.jpg'                    # glob
                                                                  'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                                  'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python segment/predict.py --weights yolov5s-seg.pt                 # PyTorch
                                          yolov5s-seg.torchscript        # TorchScript
                                          yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                          yolov5s-seg_openvino_model     # OpenVINO
                                          yolov5s-seg.engine             # TensorRT
                                          yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                          yolov5s-seg_saved_model        # TensorFlow SavedModel
                                          yolov5s-seg.pb                 # TensorFlow GraphDef
                                          yolov5s-seg.tflite             # TensorFlow Lite
                                          yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
                                          yolov5s-seg_paddle_model       # PaddlePaddle
"""
import cv2
import math
import re
import numpy as np
import argparse
import logging
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import masks2segments, process_mask
from utils.torch_utils import select_device, smart_inference_mode


logger = logging.getLogger("jinchuan.segment")
from model_decryptor import load_encrypted_torch


@smart_inference_mode()
def loadmodel(
    weights=ROOT / 'weights' / 'best.pt',  # model.pt path(s)
    data=ROOT / 'data/nietiao-seg.yaml',  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    project=ROOT / 'runs/predict-seg',  # save results to project/name
    name='exp',  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference

):

    # Directories
    #save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)

    # 支持加密模型加载 - 使用公共方法
    from model_decryptor import is_encrypted_file, load_encrypted_torch
    from models.experimental import attempt_load

    # 优先使用加密文件
    enc_path = str(weights) + '.enc'
    if os.path.exists(enc_path) and not is_encrypted_file(str(weights)):
        weights_path = enc_path
    else:
        weights_path = str(weights)

    # 检查是否为加密文件
    if is_encrypted_file(weights_path):
        logger.info("load_encrypted_segment_model", extra={"event": "load_encrypted_segment_model"})
        # 使用自定义加载函数
        loaded = load_encrypted_torch(weights_path, device=device)
        # 提取实际模型
        if isinstance(loaded, dict) and 'model' in loaded:
            model = loaded['model']
        elif isinstance(loaded, dict) and 'ema' in loaded:
            model = loaded['ema']
        else:
            model = loaded
        # 获取属性
        stride = model.stride if hasattr(model, 'stride') else 32
        if isinstance(stride, torch.Tensor):
            stride = int(stride.max())
        names = model.names if hasattr(model, 'names') else {}
        pt = True
        # CPU 模式下强制使用 FP32（不支持 FP16）
        if device.type == 'cpu' or not half:
            model.float()
        else:
            model.half()
        model.eval()
    else:
        # 使用 DetectMultiBackend 加载原始模型
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    return model,stride,pt,names
   

    


def infer(model,stride,pt,names,source,vid_stride=1,augment=False,imgsz=(640, 640),
          conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        nosave=True,  # do not save images/videos
        visualize=False,  # visualize features
        view_img=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        retina_masks=False,
        k=0.901, #像素转换为实际毫米的系数
        # x_compensation=-40.00,#若相机发生位移，对x和y加入补偿
        # y_compensation=0.00
    ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # Dataloader
    bs = 1  # batch_size
    if screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    # Run inference
    # DetectMultiBackend 有 warmup 方法，原始模型可能没有
    if hasattr(model, 'warmup'):
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    # 兼容处理：获取设备信息
    # DetectMultiBackend 有 device 属性，原始模型需要从参数获取
    if hasattr(model, 'device'):
        model_device = model.device
        model_fp16 = model.fp16 if hasattr(model, 'fp16') else False
    else:
        # 原始模型：从第一个参数获取设备
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device('cpu')
        model_fp16 = False

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model_device)
            im = im.half() if model_fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = False
            pred, proto = model(im, augment=augment, visualize=visualize)[:2]

    

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
         

        segresults=[]
        # Process predictions
        for i, det in enumerate(pred): # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # Segments
                if save_txt:
                    segments = reversed(masks2segments(masks))
                    segments = [scale_segments(im.shape[2:], x, im0.shape, normalize=True) for x in segments]

                # Mask plotting
                annotator.masks(masks,
                                colors=[colors(x, True) for x in det[:, 5]],
                                im_gpu=None if retina_masks else im[i])

                # Write results
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    if save_txt:  # Write to file
                        segj = segments[j].reshape(-1)  # (n,2) to (n*2)
                        if (xyxy[2] - xyxy[0]) > 800 or (xyxy[3] - xyxy[1]) > 800:
                            line = (cls, *segj, conf) if save_conf else (cls, *segj)  # label format
                            segresults.append(line)
                        # print('segresult',segresults)
                        #with open(f'{txt_path}.txt', 'a') as f:
                        #    f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        else:
                            continue
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        # annotator.draw.polygon(segments[j], outline=colors(c, True), width=3)
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

           
            # Save results (image with detections)
            #save_img = True
            #if save_img:
            #    if dataset.mode == 'image':
            #        cv2.imwrite(save_path, im0)

            # 读取图片
            # save_path=r'D:\Projects\JinchuanPlateDetection\JinchuanPlateDetection\bin\Debug\net462\Data\Strips\2024\01\17\210416588.png'
            # image = cv2.imread(save_path)
            # if image is None:
            #    raise ValueError("无法读取图像文件，请确保提供正确的图像路径！")
            # segresults=segresults[1:]
            # print("pred",pred)
            # print("segresults",segresults[0],type(segresults[0]))
            if(len(segresults)==0):
                return -1, 0, 0, 0, 0, 0 

            x=0
            y=0
            coordinates=[]
            for index,value in enumerate(segresults[0]):
                if(index==0):
                    continue
                if(index%2==1):
                    x=value
                else:
                    y=value
                    coordinates.append((float(x),float(y)))

            # 解析小数坐标
            #pattern = r'(\d*\.\d+)\s+(\d*\.\d+)'  # 正则表达式模式，匹配两个小数值作为一组坐标
            #matches = re.findall(pattern, segresults)
            #coordinates = [(float(x), float(y)) for x, y in matches]

            # 将相对坐标转换为绝对坐标
            w, h = cv2.imread(source).shape[1], cv2.imread(source).shape[0]
            abs_coordinates = [(int(x * w), int(y * h)) for x, y in coordinates]

            # 计算最小外接矩形
            rect = cv2.minAreaRect(np.array(abs_coordinates))
            box = np.intp(cv2.boxPoints(rect))  # 获取最小外接矩形的4个顶点坐标
            x1,y1 = box[0]
            x2,y2 = box[1]
            x3,y3 = box[2]
            x4,y4 = box[3]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            x3 = int(x3)
            y3 = int(y3)
            x4 = int(x4)
            y4 = int(y4)
            #print(box)
            #角度判断
            sorted_box = box[np.argsort(box[:,1])]
            bottom_two_coordinates = sorted_box[:2]
            x5,y5 = bottom_two_coordinates[0]
            x6,y6 = bottom_two_coordinates[1]
            x_comparison = np.greater(x6,x5)
            y_comparison = np.greater(y6,y5)

            #读取图片
            # draw_rect = Image.copy()
            # cv2.drawContours(draw_rect, [box], 0, (0, 255, 0), 2)  # 绘制轮廓最小外接矩形
            # cv2.namedWindow("enhanced",0)
            # cv2.resizeWindow("enhanced", 640, 480)
            # cv2.imshow('enhanced',draw_rect)
            # cv2.waitKey()

            center_x, center_y = rect[0]
            width, height = rect[1] 
            angle=rect[2]
            if(x_comparison and y_comparison):
                angle = angle
            else:
                if(angle == 0):
                    angle = 0.01
                else:
                    angle = (angle-90)  
            angle = 0.01 if round(angle,2)==0.00 else angle
            center_x = round((center_x-1224)*k+2.04,2)
            center_x = 0.01 if center_x==0.00 else center_x
            center_y = round(((center_y-1024)*k+2.04)*(-1),2) 
            center_y = 0.01 if center_y==0.00 else center_y  
            mmWidth = width*k
            mmHeight = height*k

            # center_x=center_x+x_compensation
            # center_y=center_y+y_compensation

            x_range = [980, 2860]
            y_range = [470, 1590]
            flag=True
            for point in box:
                x, y = point
                if x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1]:
                    flag=True
                else:
                    flag=False
            if flag:
                result={
                        "width":width,
                        "height":height,
                        "angle":angle,
                        "center_x":center_x,
                        "center_y":center_y
                    }
                logger.debug("segment_result", extra={"event": "segment_result"})
            else:
                logger.warning("segment_out_of_range", extra={"event": "segment_out_of_range"})
    

                    
    # print("segresults",segresults)
    return mmWidth, mmHeight, angle, center_x, center_y,box
def main():
    global model
    global stride
    global pt
    #global save_dir
    global names
    model,stride,pt,names=loadmodel()

    width, height, angle, center_x, center_y, box =infer(
        model=model,stride=stride,pt=pt,names=names,source=r'D:/Projects/JinchuanPlateDetection/JinchuanPlateDetection/bin/Debug/Mv/Data/Strips/2024/01/28/104241404.png'
    )
    #print(width, height, angle, center_x, center_y)

if __name__ == "__main__":
    main()
