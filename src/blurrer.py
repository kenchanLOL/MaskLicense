
import os
from timeit import default_timer as timer
from numpy.linalg import det
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import tool.generate_detections as gdet
import cv2
import numpy as np
import torch
from PySide2.QtCore import QThread, Signal
from src.box import Box
import math

from deep_sort_tf import nn_matching
from deep_sort_tf import preprocessing
from deep_sort_tf.detection import Detection
from deep_sort_tf.tracker import Tracker

import copy

# libs for plate bbox drawing 

import matplotlib.pyplot as plt
from _collections import deque


class VideoBlurrer(QThread):
    setMaximum = Signal(int)
    updateProgress = Signal(int)
    def __init__(self, weights_name='yolov4-objv8_6000', parameters=None):
        """
        Constructor
        :param weights_name: file name of the weights to be used
        :param parameters: all relevant paremeters for the blurring process
        3 network model will be init in the function
        1.self.detector 
            -license plate AI
            -yolov4
            -Pytorch
        2.car_detector
            -car AI
            -yolov4
            -Pytorch
            -fixed 
                -using yolov4.weight
                -pre-trained yolov4 network on coco dataset and name
        3.encoder
            -deepSORT feature map generator
            -yolov4 (unsure???)
            -Tensorflow (replaced by Pytorch may reduce runtime but need to consider GPU memory)
            -fixed 
                -using mars-small128.pb
                -from https://github.com/theAIGuysCode/yolov4-deepsort
        """
        super(VideoBlurrer, self).__init__()
        self.parameters = parameters
        self.detections = []
        basefile_dir=os.path.split(__file__)[0].rsplit('/',1)[0]

        # load license plate detection networks
        weights_path = os.path.join(basefile_dir+"/weights/raw_weight", f"{weights_name}.weights")
        self.detector = setup_detector(basefile_dir,weights_path)

        # load car detection networks
        cfg_path=basefile_dir+'/yolov4.cfg'
        self.car_detector=Darknet(cfg_path)
        self.car_detector.load_weights(os.path.join(basefile_dir+'/weights/raw_weight','yolov4.weights'))
        if torch.cuda.is_available():
            print(f"Using {torch.cuda.get_device_name(torch.cuda.current_device())}.")
            self.car_detector.cuda()
            torch.backends.cudnn.benchmark = True
        else:
            print("Using CPU.")
        
        self.result = {"success": False, "elapsed_time": 0}
        print("Worker created")

        # load feature map networks
        # max_cosine _distance=> for feature map matching 
        # nn_budget=> unknowmn
        # nms_max_overlap=> non-maximum suppression IOU threshold
        max_cosine_distance = 0.5
        nn_budget = None
        self.nms_max_overlap = 0.8

        # Extractor is the one used by deepSORT Pytorch version 
        # self.encoder=Extractor(model_path= basefile_dir+'/deep_sort/deep/checkpoint/ckpt.t7',use_cuda=torch.cuda.is_available())

        model_filename = basefile_dir+'/model_data_tf/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)

    def apply_blur(self, frame, new_detections):
        """
        Apply Gaussian blur to regions of interests
        :param frame: input image
        :param new_detections: list of newly detected faces and plates
        :return: processed image
        *** very common error message***
        "Traceback (most recent call last):
        File "/home/h06607/Masking_related/MaskLicense_kalman/src/blurrer.py", line 506, in run
    
        File "/home/h06607/Masking_related/MaskLicense_kalman/src/blurrer.py", line 139, in apply_blur
            frame[outer_box.coords_as_slices()] = cv2.blur(
        cv2.error: OpenCV(4.5.2) /tmp/pip-req-build-sl2aelck/opencv/modules/imgproc/src/box_filter.dispatch.cpp:446: error: (-215:Assertion failed) !_src.empty() in function 'boxFilter'"
        
        Reason:
        -> wrong detection box size input
            -x/y_min >x/y_max
            -detection width/height <=0
            -bbox > frame width/height
        """
        # gather inputs from self.parameters
        # commented codes are for frame memory
        blur_size = self.parameters["blur_size"]
        # blur_memory = self.parameters["blur_memory"]
        # roi_multi = self.parameters["roi_multi"]
        # gather and process all currently relevant detections
        # self.detections = [[x[0], x[1] + 1] for x in self.detections if
        #                    x[1] <= blur_memory] 
        # throw out outdated detections, increase age by 1

        # apply ROI scaling (tbh it is the most brute force method to cover some specific plate (e.g cross-border licenses) better
        # but it does not work well in general cases)
        # if u need to use it, ignore the suggestion on removing ROI ratio in main.py

        # for detection in new_detections:
        #     scaled_detection = detection.scale(frame.shape, roi_multi)
        #     self.detections.append([scaled_detection, 0])
        
        # prepare copy and mask
        temp = frame.copy()
        mask = np.full((frame.shape[0], frame.shape[1], 1), 0, dtype=np.uint8)
        # print("========================================================")
        # print("blur box:")
        for detection in new_detections:
            # two-fold blurring: softer blur on the edge of the box to look smoother and less abrupt
            outer_box = detection

            # Dont pass extremely small bbox here as it may round off to zero after scaling and return error in blur function later
            inner_box = detection.scale(frame.shape, 0.8)
            # print(detection)
            frame[outer_box.coords_as_slices()] = cv2.blur(
                frame[outer_box.coords_as_slices()], 
                (blur_size, blur_size))
            frame[inner_box.coords_as_slices()] = cv2.blur(
                frame[inner_box.coords_as_slices()],
                (blur_size * 2 + 1, blur_size * 2 + 1))
            cv2.rectangle

        # print("========================================================")
        mask_inverted = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(frame, frame, mask=mask_inverted)
        blurred = cv2.bitwise_and(temp, temp, mask=mask)
        self.detections=[]
        return cv2.add(background, blurred)

    def detect_identifiable_information(self, image: np.array,car_count):
        """
        Run plate and car detection on an input image 
        :param image: input image
        :return: detected faces and plates
        ****************************************************************************
        Workflow: (variable names) 
        Car detection-> (cars) 
            - car bboxes
            - dtype:[x1,y1,x2,y2]

        Crop and resize based on car bbox-> (crop)
            - Cropped image
        License Plate detection->(detected_plate) 
            - plate bboxes
            - dtype:[x1,y1,x2,y2]

        bboxes convertion->(detections)
            - dtype:Detection(bbox,score.feature)
            - bbox:[x1,y1,width,height] (Top Left Width Height=tlwh)
            - score:conf level
            - feature: nparray of feature map generated by encoder AI

        deepSORT tracking->(self.trackers.tracks) 
            - motion tracks of different plate objects 
            - dtype: [Track(),Track().....]
            

        bbox convertion->(confirmed plates)
            - plates bbox that belongs to confirmed tracks (tracks that are still
            in the frame and detected by AI or not exceeding the max age)
            - track.tlbr()==Top Left Bottom Right
            - dtype:[x1,y1,x2,y2]
    
        suppressing wrong bbox->(res)
            - dtype:Box(x1,y1,x2,y2,conf,class)
        ****************************************************************************

        """
        car_detector=self.car_detector
        license_model=self.detector
        #  all image height/width must be resized to multiples of 416 
        #  but large image may cause CUDA memory out of space
        sized=cv2.resize(image,(car_detector.width,car_detector.height))
        confirmed_plates=[]
        detected_plates=[]
        """"
        return format of do_detect():
        [
            max confidence anchor box->
            *** all coordinates are between [0,1] ***
            [
                [ 
                    x_min(x1), y_min(y1), x_max(x2), y_max(y2), 
                    max Confidence level of class, same conf value (repeated value, seems to be useless in our case but),
                    class id of max conf level
                ],
                [car2],.....
            ],
            
            #  Incompleted by the author
            [bboxs detected by other anchor boxes (i guess)], .... 
        ]

        """
        cars=do_detect(car_detector,sized,0.4,0.6,torch.cuda.is_available())[0]
        width = image.shape[1]
        height = image.shape[0]
        
        for i in range(len(cars)):
            
            box=cars[i]
            #Bounding box(bbox) coordinates conversion 
            x1 = max(0,int(box[0] * width))
            y1 = max(0,int(box[1] * height))
            x2 = min(width,int(box[2] * width))
            y2 = min(height,int(box[3] * height))
            # determine the car size 
            car_width=abs(x1-x2)
            car_height=abs(y1-y2)
            diagonal=math.sqrt(pow(car_height,2)+pow(car_width,2))
            cls_id=box[-1]
            '''
            Cropping method:
            small car
                -get the mid point of car bbox
                -crop 416*416
            big car
                -get the mid point of car bbox
                -crop (n*416)*(n*416)
                -resize to 416*416
                -*** dont directly resize big car bbox to 416*416 (deformmation of license plate-> undetected)
            (TBH i think the small car case is just a subclass of big car when n=1, but i dont have time to check and debug it)
            ** for version 9 plate detection AI, there are some extremely large false positive bounding box, so I use IOU to suppress it
            ** this imply that the training dataset still need improvements to reduce false positive
            '''
            if(cls_id in [2,3,5,7]) and (diagonal>=100):
                # crop image for cars size in range of (10*10 to 416*416)
                if (diagonal<=588):
                    car_count+=1
                    bbox_x_mid=int(x1+car_width/2)
                    bbox_y_mid=int(y1+car_height/2)
                    bbox_x1=bbox_x_mid-208
                    bbox_x2=bbox_x_mid+208
                    bbox_y1=bbox_y_mid-208
                    bbox_y2=bbox_y_mid+208
                    if(bbox_y_mid<208):
                        bbox_y1=0
                        bbox_y2=416
                    elif(bbox_y_mid>(height-208)):
                        bbox_y1=height-416
                        bbox_y2=height
                    if(bbox_x_mid<208):
                        bbox_x1=0
                        bbox_x2=416
                    elif(bbox_x_mid>(width-208)):
                        bbox_x1=width-416
                        bbox_x2=width
                    crop=image[bbox_y1:bbox_y2,bbox_x1:bbox_x2]
                    crop=cv2.resize(crop,(license_model.width,license_model.height))
                    plates=do_detect(license_model,crop,0.6,0.6,torch.cuda.is_available())[0]
                    for plate in plates:
                        plate[0] = bbox_x1+max(0,plate[0]) * license_model.width
                        plate[1] = bbox_y1+max(0,plate[1]) * license_model.height
                        plate[2] = bbox_x1+min(1,plate[2]) * license_model.width
                        plate[3] = bbox_y1+min(1,plate[3]) * license_model.height
                        plate_width=abs(plate[2]-plate[0])
                        plate_height=abs(plate[3]-plate[1])
                        plate_size=plate_width*plate_height
                        iou=plate_size/(license_model.width*license_model.height*pow(1,2))
                        if(iou>1/9):
                            continue
                        detected_plates.append(plate)
                    
                        

                else:
                    car_count+=1
                    max_side_length=max(car_width,car_height)
                    crop_power=math.ceil(max_side_length//416)
                    bbox_x_mid=int(x1+car_width/2)
                    bbox_y_mid=int(y1+car_height/2)
                    bbox_x1=bbox_x_mid-208*crop_power
                    bbox_x2=bbox_x_mid+208*crop_power
                    bbox_y1=bbox_y_mid-208*crop_power
                    bbox_y2=bbox_y_mid+208*crop_power
                    if(bbox_y_mid<208*crop_power):
                        bbox_y1=0
                        bbox_y2=416*crop_power
                    elif(bbox_y_mid>(height-208*crop_power)):
                        bbox_y1=height-416*crop_power
                        bbox_y2=height
                    if(bbox_x_mid<208*crop_power):
                        bbox_x1=0
                        bbox_x2=416*crop_power
                    elif(bbox_x_mid>(width-208*crop_power)):
                        bbox_x1=width-416*crop_power
                        bbox_x2=width
                    crop=image[bbox_y1:bbox_y2,bbox_x1:bbox_x2]
                    crop=cv2.resize(crop,(license_model.width,license_model.height))
                    plates=do_detect(license_model,crop,0.6,0.6,torch.cuda.is_available())[0]
                    # print("========================================================")
                    # print(f'plates of large car {car_count}')
                    # for plate in plates:
                    #     plate_x1 = max(0,int(plate[0] * license_model.width))
                    #     plate_y1 = max(0,int(plate[1] * license_model.height))
                    #     plate_x2 = int(plate[2] * license_model.width)
                    #     plate_y2 = int(plate[3] * license_model.height)
                    #     cv2.rectangle(crop,(plate_x1,plate_y1),(plate_x2,plate_y2),(255,0,0),5)

                    #     cv2.imwrite(f'car_{car_count}.jpg',crop)
                    for plate in plates:
                        plate[0] = bbox_x1+max(0,plate[0]) * license_model.width*crop_power
                        plate[1] = bbox_y1+max(0,plate[1]) * license_model.height*crop_power
                        plate[2] = bbox_x1+min(1,plate[2]) * license_model.width*crop_power
                        plate[3] = bbox_y1+min(1,plate[3]) * license_model.height*crop_power
                        plate_width=abs(plate[2]-plate[0])
                        plate_height=abs(plate[3]-plate[1])
                        plate_size=plate_width*plate_height
                        iou=plate_size/(license_model.width*license_model.height*pow(crop_power,2))
                        if(iou>1/9):
                            continue
                        detected_plates.append(plate)
        # print("========================================================")
        # print("Plate bbox:")
        # for plate in detected_plates:
        #     print(plate)
        #     plate_width=abs(plate[2]-plate[0])
        #     plate_height=abs(plate[3]-plate[1])
        #     plate_size=plate_width*plate_height
        #     print(plate_size)
        # print("========================================================")
        
        # implementation of deepSORT
        bboxes=[]
        scores=[]
        detected_plates_copy=copy.deepcopy(detected_plates)
        for plate in detected_plates_copy:
            plate[3]=int(abs(plate[3]-plate[1]))
            plate[2]=int(abs(plate[2]-plate[0]))
            bboxes.append(plate[:4])
            scores.append(plate[-2])        
        bboxes=np.array(bboxes)
        scores=np.array(scores)
        features = self.encoder(image, bboxes)
        detections=[Detection(bbox,score,"plate",feature) for bbox,score,feature in 
        zip(bboxes,scores,features) ]
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        # print("========================================================")
        # print("detected boxs :")
        # for box in detections:
        #     print(box.to_tlbr())
        # print("========================================================")
        self.tracker.predict()
        self.tracker.update(detections)
        for track in self.tracker.tracks:
            bbox=track.to_tlbr()
            # when the prediction is out of the image, change the state of track to TrackState.Delete
            if not track.is_confirmed():
                continue
            
            confirmed_plates.append(bbox)
        res=[]
        # print("========================================================")
        # print("confirmed boxs :")
        for box in confirmed_plates:
            # print(box)
            if (box[0]<0) or (box[1]<0) or (box[2]<0) or (box[3]<0):
                continue
            box_width=(box[2]-box[0])
            box_height=(box[3]-box[1])
            if(box_width<=3) or (box_height<=3) or (box[2]>width) or (box[3]>height) or ((pow(box_width,2)+pow(box_height,2))<200):
                continue
            else:
                res.append(Box(box[0],box[1],box[2],box[3],1,'plate'))

        # print("========================================================")
        # print(res)
        return res,car_count,cars

    def run(self):
        """
        Write a copy of the input video stripped of identifiable information, i.e. faces and license plates
        """
        iters=self.parameters["input_path_iter_list"]
        print('File_name,Num_of_frames,Avg_Num_of_car,Time_used')
        timer_total=timer()
        for iter in iters:
            while (iter.hasNext()):
                car_count=0
                self.parameters['input_path_Cur']=iter.next()
                # reset success and start timer
                self.result["success"] = False
                start = timer()
                # gather inputs from self.parameters
                
                input_path = self.parameters["input_path_Cur"]
                # print('path name: '+input_path)
                fname=input_path.rsplit('/',1)[-1]
                # temp_output = f"{os.path.splitext(self.parameters['output_path'])[0]}_copy{os.path.splitext(self.parameters['output_path'])[1]}"
                output_path = self.parameters["output_path"]+'/'+fname
                # print(output_path)
                # threshold = self.parameters["threshold"]
                # print(f"Worker of {fname} started")
                # customize detector
                # self.detector.conf = threshold

                # open video file
                cap = cv2.VideoCapture(input_path)

                # get the height and width of each frame
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                # print("video size:")
                # print(width,height)
                # save the video to a file
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height)) 
                # update GUI's progress bar on its maximum frames
                self.setMaximum.emit(length)

                if cap.isOpened() == False:
                    # print(f'error at video {fname}')
                    print(str(fname)+','+str(length)+','+'-1')
                    continue
                # loop through video
                current_frame = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    
                    if ret == True:
                        last=car_count
                        # t1=time.time()
                        new_detections,car_count,cars = self.detect_identifiable_information(frame.copy(),last)
                        # print car bbox
                        # for i in range(len(cars)):
                        #     box=cars[i]
                        #     x1 = max(0,int(box[0] * width))
                        #     y1 = max(0,int(box[1] * height))
                        #     x2 = int(box[2] * width)
                        #     y2 = int(box[3] * height)
                        #     cls_id=box[6]
                        #     if(cls_id in [2,3,5,7]):
                        #         cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),5)
                        
                        # print plates bbox, id , size 
                        # pts = [deque(maxlen=30) for _ in range(1000)]
                        # cmap = plt.cm.get_cmap('tab20b')
                        # colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]
                        # for track in self.tracker.tracks:
                        #         if not track.is_confirmed():
                        #             continue
                        #         bbox = track.to_tlbr()
                        #         class_name= track.get_class()
                        #         color = colors[int(track.track_id) % len(colors)]
                        #         color = [i * 255 for i in color]
                        #         plate_width=abs(bbox[2]-bbox[0])
                        #         plate_height=abs(bbox[3]-bbox[1])
                        #         plate_size=plate_width*plate_height
                        #         cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2)
                        #         cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)
                        #                     +len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                        #         cv2.putText(frame, class_name+"-"+str(track.track_id)+"-"+str(plate_size), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75,
                        #                     (255, 255, 255), 2)
                        #         center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
                        #         pts[track.track_id].append(center)
                        # fps = 1./(time.time()-t1)
                        # cv2.putText(frame, "FPS: {:.2f}".format(fps), (0,30), 0, 1, (0,0,255), 2)
                        frame = self.apply_blur(frame, new_detections)
                        writer.write(frame)
                        # print('Car in frame '+str(current_frame))
                        # Num_car=car_count-last
                        # print(Num_car)

                    else:
                        break
                    current_frame += 1
                    self.updateProgress.emit(current_frame)
                
                
                self.detections = []
                cap.release()
                writer.release()
                ## store sucess and elapsed time
                self.result["success"] = True
                used_time=timer() - start
                avg_car=car_count/length
                print(str(fname)+','+str(length)+','+str(avg_car)+','+str(used_time))
        self.result["success"] = True
        self.result["elapsed_time"] = timer()-timer_total



def setup_detector(basefile_dir,weights_path: str):
    """
    Load YOLOv5 detector from torch hub and update the detector with this repo's weights
    :param weights_path: path to .pt file with this repo's weights
    :return: initialized yolov5 detector
    """
    # model = torch.hub.load('ultralytics/yolov5', 'custom', weights_path)
    cfg_path=basefile_dir+'/yolov4-obj.cfg'
    model=Darknet(cfg_path)
    model.print_network()
    model.load_weights(weights_path)
    if torch.cuda.is_available():
        print(f"Using {torch.cuda.get_device_name(torch.cuda.current_device())}.")
        model.cuda()
        torch.backends.cudnn.benchmark = True
    else:
        print("Using CPU.")
    return model
