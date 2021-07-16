import os
import subprocess
from timeit import default_timer as timer
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import cv2
import numpy as np
import torch
from PySide2.QtCore import QThread, Signal
from src.box import Box
import math

def setup_detector(weights_path: str):
    cfg_path='/home/h06607/MaskLicense/yolov4-obj.cfg'
    model=Darknet(cfg_path)
    model.print_network()
    model.load_weights(weights_path)
    if torch.cuda.is_available():
        print(f"Using {torch.cuda.get_device_name(torch.cuda.current_device())}.")
        model.cuda()
        torch.backends.cudnn.benchmark = True
    # else:
    #     print("Using CPU.")
    return model
weights_name="yolov4-objv2_1000"
# basefile_dir=os.path.split(__file__)[0].rsplit('/',1)[0]
weights_path = os.path.join("/home/h06607/MaskLicense//weights/raw_weight", f"{weights_name}.weights")
license_model = setup_detector(weights_path)
cfg_path='/home/h06607/MaskLicense/yolov4.cfg'
car_model=Darknet(cfg_path)
car_model.print_network()
car_model.load_weights(os.path.join('/home/h06607/MaskLicense/weights/raw_weight','yolov4.weights'))
if torch.cuda.is_available():
    # print(f"Using {torch.cuda.get_device_name(torch.cuda.current_device())}.")
    car_model.cuda()
    torch.backends.cudnn.benchmark = True
print("Worker created")
frame_name="/home/h06607/MaskLicense/test_frame_1.png"
frame=cv2.imread(frame_name)
width = frame.shape[1]
height = frame.shape[0]
# new_detections,cars =detect_identifiable_information(frame.copy(),width,height)
sized=cv2.resize(frame,(car_model.width,car_model.height))
cars=do_detect(car_model,sized,0.4,0.6,torch.cuda.is_available())[0]
count=0
for i in range(len(cars)):
    box=cars[i]
    x1 = max(0,int(box[0] * width))
    y1 = max(0,int(box[1] * height))
    x2 = int(box[2] * width)
    y2 = int(box[3] * height)
    cls_id=box[6]
    car_width=abs(x1-x2)
    car_height=abs(y1-y2)
    diagonal=math.sqrt(pow(car_height,2)+pow(car_width,2))        
    cls_conf=box[5]
    cls_id=box[-1]
    if(cls_id in [2,3,5,7]) and (diagonal>=100):
    # if(cls_id in [2,3,5,7]):
        # cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),5)
        count+=1
        x_mid=int(x1+car_width/2)
        y_mid=int(y1+car_height/2)
        x1=x_mid-208
        x2=x_mid+208
        y1=y_mid-208
        y2=y_mid+208
        if(y_mid<208):
            y1=0
            y2=416
        elif(y_mid>(height-208)):
            y1=height-416
            y2=height
        if(x_mid<208):
            x1=0
            x2=416
        elif(x_mid>(width-208)):
            x1=width-416
            x2=width
        crop=frame[y1:y2,x1:x2]
        resized=cv2.resize(crop,(license_model.width,license_model.height))
        plates=do_detect(license_model,crop,0.4,0.6,torch.cuda.is_available())[0]
        output_name=f"test_frame_cropV3_detect_{count}.jpg"
        plot_boxes_cv2(resized,plates,output_name,class_names=None)
        # output_name=f"test_frame_cropV2_detect_{count}.jpg"
        # cv2.imwrite(output_name,resized)

   
