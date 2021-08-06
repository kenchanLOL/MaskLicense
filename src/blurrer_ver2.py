import os
from timeit import default_timer as timer

from numpy.linalg import det
from deep_sort import tracker
from deep_sort import detection
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
# from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import simple_tracker
import copy
# import operator
import matplotlib.pyplot as plt
from _collections import deque
class VideoBlurrer(QThread):
    setMaximum = Signal(int)
    updateProgress = Signal(int)
    def __init__(self, weights_name, parameters=None):
        """
        Constructor
        :param weights_name: file name of the weights to be used
        :param parameters: all relevant paremeters for the blurring process
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
        self.car_model=Darknet(cfg_path)
        self.car_model.load_weights(os.path.join(basefile_dir+'/weights/raw_weight','yolov4.weights'))
        if torch.cuda.is_available():
            print(f"Using {torch.cuda.get_device_name(torch.cuda.current_device())}.")
            self.car_model.cuda()
            torch.backends.cudnn.benchmark = True
        else:
            print("Using CPU.")
        
        self.result = {"success": False, "elapsed_time": 0}
        print("Worker created")
        max_cosine_distance = 0.5
        nn_budget = None
        self.nms_max_overlap = 0.7
        model_filename = 'model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)
        self.car_history={}
        # self.tracker2=simple_tracker()
        self.plate_history={}

    def apply_blur(self, frame, new_detections):
        """
        Apply Gaussian blur to regions of interests
        :param frame: input image
        :param new_detections: list of newly detected faces and plates
        :return: processed image
        """
        # gather inputs from self.parameters
        blur_size = self.parameters["blur_size"]
        # blur_memory = self.parameters["blur_memory"]
        roi_multi = self.parameters["roi_multi"]

        # gather and process all currently relevant detections
        # self.detections = [[x[0], x[1] + 1] for x in self.detections if
        #                    x[1] <= blur_memory]  # throw out outdated detections, increase age by 1
        for detection in new_detections:
            scaled_detection = detection.scale(frame.shape, roi_multi)
            self.detections.append([scaled_detection, 0])
        # prepare copy and mask
        temp = frame.copy()
        mask = np.full((frame.shape[0], frame.shape[1], 1), 0, dtype=np.uint8)
        print("========================================================")
        print("blur box:")
        for detection in [x[0] for x in self.detections]:
            # detection.x_min=max(0,detection.x_min)
            # detection.y_min=max(0,detection.y_min)
            # detection.x_max=min(frame.shape[0],detection.x_max)
            # detection.y_max=min(frame.shape[1],detection.y_max)
            # if(math.sqrt(pow(detection.x_max-detection.x_min,2)+pow(detection.y_max-detection.y_max,2))<15):
            # two-fold blurring: softer blur on the edge of the box to look smoother and less abrupt
            outer_box = detection
            # Dont pass extremely small bbox here as it may round off to zero after scaling and return error in blur function later 
            inner_box = detection.scale(frame.shape, 0.8)
            print(detection)
            # if detection.kind == "plate":
            frame[outer_box.coords_as_slices()] = cv2.blur(
                frame[outer_box.coords_as_slices()], 
                (blur_size, blur_size))
            frame[inner_box.coords_as_slices()] = cv2.blur(
                frame[inner_box.coords_as_slices()],
                (blur_size * 2 + 1, blur_size * 2 + 1))
            cv2.rectangle

            # else:
            #     raise ValueError(f"Detection kind not supported: {detection.kind}")

        print("========================================================")
        mask_inverted = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(frame, frame, mask=mask_inverted)
        blurred = cv2.bitwise_and(temp, temp, mask=mask)
        self.detections=[]
        return cv2.add(background, blurred)

    # def convert_box_tlbr(boxes,width,height,car_x1,car_y1):
    #     for box in boxes:
    #         box[0] = car_x1+max(0,box[0]) * width
    #         box[1] = car_y1+max(0,box[1]) * height
    #         box[2] = car_x1+min(1,box[2]) * width
    #         box[3] = car_y1+min(1,box[3]) * height
            

    def detect_identifiable_information(self, image: np.array,car_count):
        """
        Run plate and face detection on an input image
        :param image: input image
        :return: detected faces and plates
        """
        car_model=self.car_model
        license_model=self.detector
        sized=cv2.resize(image,(car_model.width,car_model.height))
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
            [bboxs detected by other anchor boxes], .... 
        ]

        """

        cars=do_detect(car_model,sized,0.4,0.6,torch.cuda.is_available())[0]
        width = image.shape[1]
        height = image.shape[0]
        removed=[]
        for idx,car in enumerate(cars):
            car[0] = max(0,car[0]) * width
            car[1] = max(0,car[1]) * height
            car[2] = min(1,car[2]) * width
            car[3] = min(1,car[3]) * height

            # determine the car size 
            car_width=int(abs(car[2]-car[0]))
            car_height=int(abs(car[3]-car[1]))
            diagonal=math.sqrt(pow(car_height,2)+pow(car_width,2))
            cls_id=car[-1]       
            if not (cls_id in [2,3,5,7] or diagonal<=25):
                removed.append(idx)
        cars=[c for i,c in enumerate(cars) if i not in removed]
        bboxes=[]
        scores=[]
        for car in cars:
            car[3]=int(abs(car[3]-car[1]))
            car[2]=int(abs(car[2]-car[0]))
            bboxes.append(car[:4])
            scores.append(car[-2])
        bboxes=np.array(bboxes)
        scores=np.array(scores)
        features=self.encoder(image,bboxes)
        detections=[Detection(bbox,score,"car",feature) for bbox,score,feature in 
        zip(bboxes,scores,features) ]
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        print("========================================================")
        print("detections: ")
        for detection in detections:
            print(detection.to_tlbr())
        print("========================================================")
        
        self.tracker.predict()
        self.tracker.update(detections)
        last_cars=copy.deepcopy(self.car_history)
        self.car_history={}
        for track in self.tracker.tracks:
            
            if not track.is_confirmed() or track.time_since_update >1:
                continue
            if len(last_cars)>0 and track.track_id in last_cars.keys():
                bbox=track.to_tlbr()
                delta_x1=bbox[0]-last_cars[track.track_id][0]
                delta_y1=bbox[1]-last_cars[track.track_id][1]
                delta_x2=bbox[2]-last_cars[track.track_id][2]
                delta_y2=bbox[3]-last_cars[track.track_id][3]
                self.car_history[track.track_id]=bbox.extend([delta_x1,delta_y1,delta_x2,delta_y2])
            else:
                bbox=track.to_tlbr()
                self.car_history[track.track_id]=bbox.extend([0,0,0,0])
        # for track_idx,detection_idx in self.tracker.matches:
        #     self.confirmed_cars[track_idx].extend([detection_idx])

        for key,car in self.car_history:
            car_idx=key
            # determine the car size 
                # crop image for cars size in range of (10*10 to 416*416)
            car_width=int(abs(car[2]-car[0]))
            car_height=int(abs(car[3]-car[1]))
            diagonal=math.sqrt(pow(car_height,2)+pow(car_width,2))
            if (diagonal<=588):
                car_count+=1
                bbox_x_mid=int(car[0]+car_width/2)
                bbox_y_mid=int(car[1]+car_height/2)
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
                plates=do_detect(license_model,crop,0.4,0.6,torch.cuda.is_available())[0]
                if (len(plates>0)):
                    temp=[]
                    for plate in plates:
                        plate[0] = bbox_x1+max(0,plate[0]) * license_model.width
                        plate[1] = bbox_y1+max(0,plate[1]) * license_model.height
                        plate[2] = bbox_x1+min(1,plate[2]) * license_model.width
                        plate[3] = bbox_y1+min(1,plate[3]) * license_model.height
                        
                        temp.append(plate)
                    self.plate_history[car_idx]=temp
                    
                else:
                    if car_idx not in self.plate_history:
                        continue
                    else:
                        temp=[]
                        for plate in self.plate_history[car_idx]:
                            plate[0] += delta_x1
                            plate[1] += delta_y1
                            plate[2] += delta_x2
                            plate[3] += delta_y2
                            temp.append(plate)
                        self.plate_history[car_idx]=temp



            else:
                car_count+=1
                bbox_x_mid=int(car[0]+car_width/2)
                bbox_y_mid=int(car[1]+car_height/2)
                bbox_x1=bbox_x_mid-208
                bbox_x2=bbox_x_mid+208
                bbox_y1=bbox_y_mid-208
                bbox_y2=bbox_y_mid+208
                crop=image[bbox_y1:bbox_y2,bbox_x1:bbox_x2]
                crop=cv2.resize(crop,(license_model.width,license_model.height))
                plates=do_detect(license_model,crop,0.4,0.6,torch.cuda.is_available())[0]
                if (len(plates>0)):
                    temp=[]
                    for plate in plates:
                        plate[0] = bbox_x1+max(0,plate[0]) * license_model.width
                        plate[1] = bbox_y1+max(0,plate[1]) * license_model.height
                        plate[2] = bbox_x1+min(1,plate[2]) * license_model.width
                        plate[3] = bbox_y1+min(1,plate[3]) * license_model.height
                        temp.append(plate)
                    self.plate_history[car_idx]=temp
                    
                else:
                    if car_idx not in self.plate_history:
                        continue
                    else:
                        temp=[]
                        for plate in self.plate_history[car_idx]:
                            plate[0] += delta_x1
                            plate[1] += delta_y1
                            plate[2] += delta_x2
                            plate[3] += delta_y2
                            temp.append(plate)
                        self.plate_history[car_idx]=temp

        print("========================================================")
        print("detected boxs :")
        for box in detections:
            print(box.to_tlbr())
        print("========================================================")

        # res=[]
        # print("========================================================")
        # print("confirmed boxs :")
        # for box in confirmed_plates:
        #     print(box)
        #     res.append(Box(box[0],box[1],box[2],box[3],1,'plate'))
        # print("========================================================")
        res=[]
        for plate in plates:
            res.append(Box(box[0],box[1],box[2],box[3],1,'plate'))
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
                        t1=time.time()
                        new_detections,car_count,cars = self.detect_identifiable_information(frame.copy(),last)
                        for i in range(len(cars)):
                            box=cars[i]
                            x1 = max(0,int(box[0] * width))
                            y1 = max(0,int(box[1] * height))
                            x2 = int(box[2] * width)
                            y2 = int(box[3] * height)
                            cls_id=box[6]
                            if(cls_id in [2,3,5,7]):
                                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),5)
                                                      
                        pts = [deque(maxlen=30) for _ in range(1000)]
                        cmap = plt.cm.get_cmap('tab20b')
                        colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]
                        for track in self.tracker.tracks:
                                if not track.is_confirmed() or track.time_since_update >1:
                                    continue
                                bbox = track.to_tlbr()
                                class_name= track.get_class()
                                color = colors[int(track.track_id) % len(colors)]
                                color = [i * 255 for i in color]

                                cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2)
                                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)
                                            +len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                                cv2.putText(frame, class_name+"-"+str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75,
                                            (255, 255, 255), 2)

                                center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
                                pts[track.track_id].append(center)

                        fps = 1./(time.time()-t1)
                        cv2.putText(frame, "FPS: {:.2f}".format(fps), (0,30), 0, 1, (0,0,255), 2)
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
