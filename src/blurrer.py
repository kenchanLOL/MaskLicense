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
        weights_path = os.path.join(basefile_dir+"/weights/raw_weight", f"{weights_name}.weights")
        self.detector = setup_detector(basefile_dir,weights_path)
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

    def apply_blur(self, frame, new_detections):
        """
        Apply Gaussian blur to regions of interests
        :param frame: input image
        :param new_detections: list of newly detected faces and plates
        :return: processed image
        """
        # gather inputs from self.parameters
        blur_size = self.parameters["blur_size"]
        blur_memory = self.parameters["blur_memory"]
        roi_multi = self.parameters["roi_multi"]

        # gather and process all currently relevant detections
        self.detections = [[x[0], x[1] + 1] for x in self.detections if
                           x[1] <= blur_memory]  # throw out outdated detections, increase age by 1
        for detection in new_detections:
            scaled_detection = detection.scale(frame.shape, roi_multi)
            self.detections.append([scaled_detection, 0])
        # prepare copy and mask
        temp = frame.copy()
        mask = np.full((frame.shape[0], frame.shape[1], 1), 0, dtype=np.uint8)
        # print("blur box:")
        for detection in [x[0] for x in self.detections]:
            # two-fold blurring: softer blur on the edge of the box to look smoother and less abrupt
            outer_box = detection
            inner_box = detection.scale(frame.shape, 0.8)
            # print(outer_box,inner_box)
            if detection.kind == "plate":
                # blur in-place on frame
                # print(outer_box)
                # print(inner_box)
                frame[outer_box.coords_as_slices()] = cv2.blur(
                    frame[outer_box.coords_as_slices()], 
                    (blur_size, blur_size))
                frame[inner_box.coords_as_slices()] = cv2.blur(
                    frame[inner_box.coords_as_slices()],
                    (blur_size * 2 + 1, blur_size * 2 + 1))
                cv2.rectangle


            # elif detection.kind == "face":
            #     center, axes = detection.ellipse_coordinates()
            #     # blur rectangle around face
            #     temp[outer_box.coords_as_slices()] = cv2.blur(
            #         temp[outer_box.coords_as_slices()], (blur_size * 2 + 1, blur_size * 2 + 1))
            #     # add ellipse to mask
            #     cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)

            else:
                raise ValueError(f"Detection kind not supported: {detection.kind}")

        # apply mask to blur faces too
        mask_inverted = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(frame, frame, mask=mask_inverted)
        blurred = cv2.bitwise_and(temp, temp, mask=mask)
        return cv2.add(background, blurred)

    def detect_identifiable_information(self, image: np.array,width,height,car_count):
        """
        Run plate and face detection on an input image
        :param image: input image
        :return: detected faces and plates
        """
        # scale = self.parameters["inference_size"]
        # results = self.detector(image, size=scale)
        car_model=self.car_model
        license_model=self.detector
        sized=cv2.resize(image,(car_model.width,car_model.height))
        # basefile_dir=os.path.split(__file__)[0].rsplit('/',1)[0]
        # class_name=load_class_names(basefile_dir+'/coco.names')
        cars=do_detect(car_model,sized,0.4,0.6,torch.cuda.is_available())[0]
#         print("frame size:")
#         print(width,height)
        
        width = image.shape[1]
        height = image.shape[0]
        new_plates=[]
        # print("licnese size:")
        # print(license_model.width,license_model.height)
        
#         print("Cars:")
        for i in range(len(cars)):
            
            box=cars[i]
            x1 = max(0,int(box[0] * width))
            y1 = max(0,int(box[1] * height))
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)
            car_width=abs(x1-x2)
            car_height=abs(y1-y2)
            diagonal=math.sqrt(pow(car_height,2)+pow(car_width,2))        
            cls_conf=box[5]
            cls_id=box[-1]
            if(cls_id in [2,3,5,7]) and (diagonal>=100):
                car_count+=1
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
                crop=image[y1:y2,x1:x2]
                crop=cv2.resize(crop,(license_model.width,license_model.height))
                plates=do_detect(license_model,crop,0.4,0.6,torch.cuda.is_available())[0]
                for k in range(len(plates)):
                    new_plates.append(
                        Box((x1+plates[k][0]*416),
                        (y1+plates[k][1]*416),
                        (x1+plates[k][2]*416),
                        (y1+plates[k][3]*416),
                        cls_conf,
                        'plate' ))
        return new_plates,cars,car_count

    def run(self):
        """
        Write a copy of the input video stripped of identifiable information, i.e. faces and license plates
        """
        iters=self.parameters["input_path_iter_list"]
        print('File_name,Num_of_frames,Avg_Num_of_car,Time_used')
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
                        width = frame.shape[1]
                        height = frame.shape[0]
                        last=car_count
                        new_detections,cars,car_count = self.detect_identifiable_information(frame.copy(),width,height,last)
                        for i in range(len(cars)):
                            box=cars[i]
                            x1 = max(0,int(box[0] * width))
                            y1 = max(0,int(box[1] * height))
                            x2 = int(box[2] * width)
                            y2 = int(box[3] * height)
                            cls_id=box[6]
                            if(cls_id in [2,3,5,7]):
                                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),5)
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

                ## copy over audio stream from original video to edited video
                #ffmpeg_exe = os.getenv("FFMPEG_BINARY")
                #subprocess.run(
                #    [ffmpeg_exe, "-y", "-i", temp_output, "-i", input_path, "-c", "copy", "-map", "0:0", "-map", "1:1",
                #     "-shortest", output_path])

                # delete temporary output that had no audio track
                # os.remove(temp_output)

                ## store sucess and elapsed time
                self.result["success"] = True
                self.result["elapsed_time"] = timer() - start
                used_time=self.result["elapsed_time"]
                # print(car_count)
                avg_car=car_count/length
                print(str(fname)+','+str(length)+','+str(avg_car)+','+str(used_time))




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
