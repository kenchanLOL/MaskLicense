import os
import subprocess
from timeit import default_timer as timer

import cv2
import numpy as np
import torch
from PySide2.QtCore import QThread, Signal
from src.box import Box


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
        weights_path = os.path.join("./dashcamcleaner/weights", f"{weights_name}.pt")
        self.detector = setup_detector(weights_path)
        self.result = {"success": False, "elapsed_time": 0}
        print("Worker created")

    def apply_blur(self, frame: np.array, new_detections: list):
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

        for detection in [x[0] for x in self.detections]:
            # two-fold blurring: softer blur on the edge of the box to look smoother and less abrupt
            outer_box = detection
            inner_box = detection.scale(frame.shape, 0.8)

            if detection.kind == "plate":
                # blur in-place on frame
                frame[outer_box.coords_as_slices()] = cv2.blur(
                    frame[outer_box.coords_as_slices()], (blur_size, blur_size))
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

    def detect_identifiable_information(self, image: np.array):
        """
        Run plate and face detection on an input image
        :param image: input image
        :return: detected faces and plates
        """
        scale = self.parameters["inference_size"]
        results = self.detector(image, size=scale)
        boxes = []
        for res in results.xyxy[0]:
            if res[5].item()==0:
                detection_kind = "plate"
                boxes.append(Box(res[0].item(), res[1].item(), res[2].item(), res[3].item(), res[4].item(), detection_kind))
        return boxes

    def run(self):
        """
        Write a copy of the input video stripped of identifiable information, i.e. faces and license plates
        """
        iter=self.parameters["input_path_iter"]
        print('File_name,Num_of_frames,Time_used')
        while (iter.hasNext()):
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
            threshold = self.parameters["threshold"]
            # print(f"Worker of {fname} started")
            # customize detector
            self.detector.conf = threshold

            # open video file
            cap = cv2.VideoCapture(input_path)

            # get the height and width of each frame
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # save the video to a file
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            # print(f'File name:{fname}')
            # print(f'Number of frame: {length}')
            # print('Result:')
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
                    new_detections = self.detect_identifiable_information(frame.copy())
                    frame = self.apply_blur(frame, new_detections)
                    writer.write(frame)
                    print(current_frame)

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
            print(str(fname)+','+str(length)+','+str(used_time))



def setup_detector(weights_path: str):
    """
    Load YOLOv5 detector from torch hub and update the detector with this repo's weights
    :param weights_path: path to .pt file with this repo's weights
    :return: initialized yolov5 detector
    """
    model = torch.hub.load('ultralytics/yolov5', 'custom', weights_path)
    if torch.cuda.is_available():
        print(f"Using {torch.cuda.get_device_name(torch.cuda.current_device())}.")
        model.cuda()
        torch.backends.cudnn.benchmark = True
    else:
        print("Using CPU.")
    return model
