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

1.open source dataset (Open Image) -> use OIDv4 toolkit
source:https://github.com/theAIGuysCode/OIDv4_ToolKit
```bash
python main.py downloader --classes 'Vehicle registration plate' --type_csv train --limit 4000
python main.py downloader --classes 'Vehicle registration plate' --type_csv validation --limit 500
python convert_annotations.py
rm -r OID/Dataset/train/'Vehicle registration plate'/Label/
rm -r OID/Dataset/validation/'Vehicle registration plate'/Label/
```

2. local car license plate dataset collection
source: flickr (https://www.flickr.com/photos/j3tourshongkong/)
the current dataset contain 1000 local car plates
### advantages
multiple car types, different color, free to use, well-organized
### drawback:
fixed lens, resolution, angle and location

### Original composition:

car type | number of data
---------|---------------
private car| 500
truck|245
bus|195
other|60

### After data augmentation:(need more improvement) 
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


3. 
