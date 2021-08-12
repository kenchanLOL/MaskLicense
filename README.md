# MaskLiencese

## installation
for gpu and pytorch,tensorflow mixed version 

**dont pip install opencv and pyside**
```bash
  conda create -n masking python=3.8
  conda activate masking
  pip install -r requirement_gpu.txt
  conda install -c anaconda cudatoolkit
  conda install -c conda-forge opencv
  conda install -c conda-forge pyside2
```

## Parameters
**frame memory** : replaced by deepSORT object track, initially used to mask previously detected area as the AI may fail to detect the plate in some frame

**blur size** : the rate of blurring , higher the value, stronger the effect

**Detection threshold** : as its name implies, the threshold of conf level (currently is not used but u may add it back to do_detect() function)

**ROI ratio** : scale up or down the detected bounding box (also currently deactivated)

**inference size** : unused params (the original wirter use it to lower the resolution of input frame in order to increase runtime)


## Train custom yolov4 model 
credit:https://colab.research.google.com/drive/11GRCzo-yKzkntLK1PwuhE4RV9nGUYLhs#scrollTo=Fl7PsmikjCBW

### open source dataset (Open Image) -> use OIDv4 toolkit
source:https://github.com/theAIGuysCode/OIDv4_ToolKit
```bash
python main.py downloader --classes 'Vehicle registration plate' --type_csv train --limit 4000
python main.py downloader --classes 'Vehicle registration plate' --type_csv validation --limit 500
python convert_annotations.py
rm -r OID/Dataset/train/'Vehicle registration plate'/Label/
rm -r OID/Dataset/validation/'Vehicle registration plate'/Label/
```

### local car license plate dataset collection

source: flickr (https://www.flickr.com/photos/j3tourshongkong/)
the current dataset contain 1000 local car plates

**advantages**

multiple car types, different color, free to use, well-organized
**drawback:**

fixed lens, resolution, angle and location

**Original composition:**

car type | number of data
---------|---------------
private car| 500
truck|245
bus|195
other|60

**After data augmentation:(need more improvement) **
* brightness decrease
* mosaic
* shear angle (whole picture / inside bounding box)
* blur

for data augmentation, i just use roboflow to augment some of the pictures in dataset (free account limit) as a proof of work. You may use other tools to process all the images, including non-local part to generate a bigger data for AI trainning

for futhre data augementation:

- [ ] birgthness variation based on certain distribution
- [ ] mosaic +blur+ brightness 
- [ ] other data augmentation techniques mentioned in yolov4 paper e.g. cutoff 
- [ ] random combination of data augmentation techniques

**please be reminded that after shearing angle, u need to check or recalculate the bounding box coordinate. If u are using roboflow to generate dataset, u need to convert the _annotation.txt file into individal txt file**

Version Number | original |  50% draker | mosaic+shear angle +blur
---------|----------|-------------|--------------------------
ver8|:large_blue_circle:|:large_blue_circle:|:red_circle:
ver9|:large_blue_circle:|:large_blue_circle:|:large_blue_circle:


## labelling the dataset 

tools: https://github.com/tzutalin/labelImg

**tips:
1.read the instructions on the tool github
2.change the classes.txt before labelling
3.open autosave and deafult label option if u are just labelling license plate
4.choose YOLO format output before labelling

## customized darknet cfg

**If u are also new to darknet training, I recommed you to open a new folder and create all the files u need and paste them to darknet folder later

*step1:define new cfg file(veryyyyyyyy important)*

texts from tutorial page


>I recommend having batch = 64 and subdivisions = 16 for ultimate results (Personal advise: if the training fail to start, reduce the batch and sudivision size as the CUDA may be out of memory) . If you run into any issues then up subdivisions to 32.Make the rest of the changes to the cfg based on how many classes you are training your detector on.

>Note: I set my max_batches = 6000, steps = 4800, 5400, I changed the classes = 1 in the three YOLO layers and filters = 18 in the three convolutional layers before the YOLO layers.

>How to Configure Your Variables:

>width = 416

>height = 416 (these can be any multiple of 32, 416 is standard, you can sometimes improve results by making value larger like 608 but will slow down training)

>max_batches = (# of classes) * 2000 (but no less than 6000 so if you are training for 1, 2, or 3 classes it will be 6000, however detector for 5 classes would have max_batches=10000)

>steps = (80% of max_batches), (90% of max_batches) (so if your max_batches = 10000, then steps = 8000, 9000)

>filters = (# of classes + 5) * 3 (so if you are training for one class then your filters = 18, but if you are training for 4 classes then your filters = 27)

>Optional: If you run into memory issues or find the training taking a super long time. In each of the three yolo layers in the cfg, change one line from random = 1 to random = 0 to speed up training but slightly reduce accuracy of model. Will also help save memory if you run into any memory issues.

**Summary**

* change width and height to 416
* change classes to 1
* change batches to 6000 (increase this if u want to train more epoches)
* search "yolo" in text editor, there should be 3 yolo layers in total, for each layer, change the filter into 18   <=(1+5)*3
**Put the {name of model}.cfg to /darknet/cfg/**

*step2: define obj.names and obj.data*

**obj.names** : name list of classes

**obj.data** : files that show where the darknet should get data from

![image](https://user-images.githubusercontent.com/55791584/129152853-f353a1a3-5bf5-4689-8edf-304441077640.png)

**Put obj.names and obj.data into /darknet/data/

*step3: generate train.txt and test.txt*

**noticed that in last step, the darknet author defined the folder name of training as "obj" and that of testing as "test".**
1. creat 2 folders named 'obj' and 'test'
2. create a folder named 'data' and put 'obj' and 'test' into it 
3. put dataset for training into obj and dataset for testing into test
4. run test_gen.py and train_gen.py



