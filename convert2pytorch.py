from tool import darknet2pytorch
import torch
import os
# weight_path=input('Enter the path of the weight file:')
# cfg_path=input('Enter the path of cfg file:')
weight_path='dashcamcleaner/weights/yolov4-obj_5000.weights'
weight_data=os.path.splitext(weight_path)
cfg_path='C:/Users/apas4/darknet/build/darknet/x64/cfg/yolov4-obj.cfg'
model = darknet2pytorch.Darknet(cfg_path, inference=True)
model.load_weights(weight_path)
torch.save(model.state_dict(), str(weight_data[0]+'.pt'))       