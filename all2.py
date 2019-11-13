import cv2
import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import logging
import os
import sys
import numpy as np
import pandas as pd
import random
logger = logging.getLogger(__name__)
import torchvision.models as models
import torchvision
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video", type=str,
# 	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())

# extract the OpenCV version info
(major, minor) = cv2.__version__.split(".")[:2]

# if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
# function to create our object tracker
if int(major) == 3 and int(minor) < 3:
	tracker = cv2.Tracker_create(args["tracker"].upper())

# otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
# approrpiate object tracker constructor:
else:
	# initialize a dictionary that maps strings to their corresponding
	# OpenCV object tracker implementations
	OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"boosting": cv2.TrackerBoosting_create,
		"mil": cv2.TrackerMIL_create,
		"tld": cv2.TrackerTLD_create,
		"medianflow": cv2.TrackerMedianFlow_create,
		"mosse": cv2.TrackerMOSSE_create
	}

	# grab the appropriate object tracker using our dictionary of
	# OpenCV object tracker objects
	tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

# initialize the bounding box coordinates of the object we are going
# to track
initBB = None


# Opens the Video file
# cap= cv2.VideoCapture('phone.mov')
# i=0
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == False:
#         break
#     cv2.imwrite('frames/cup_'+str(i)+'.jpg',frame)
#     i+=1
 
# cap.release()


import torchvision.models as models

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
device = torch.device('cuda')

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# model.to(device)
model.eval()
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, TensorDataset

import argparse
import cv2
import numpy as np
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--threshold", type = int, default = 128,
	help = "Threshold value")
args = vars(ap.parse_args())
 
# load the image and convert it to grayscale
images = os.listdir("frames/")
i = 0
for imagename in images:
    img_path = "frames/" + imagename
    img = Image.open(img_path) # Load the image
    transform = transforms.Compose([transforms.ToTensor()]) # Defing PyTorch Transform
    img = transform(img) # Apply the transform to the image
    pred = model([img])
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
    try:
        pred_t = [pred_score.index(x) for x in pred_score if x > 0.5][-1]
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
    except:
        pred_t = []
        pred_boxes = []
        pred_class = []
    if len(pred_class) != 0:
        for class_index in range(len(pred_class)):
            imageid = imagename.split("_")[1].replace(".jpg", "")
            if pred_class[class_index] == 'cell phone':
                x0 = pred_boxes[class_index][0][0]
                y0 = pred_boxes[class_index][0][1]
                x1 = pred_boxes[class_index][1][0]
                y1 = pred_boxes[class_index][1][1]
                a = int((x0+x1)/2)
                c = int((y0+y1)/2)
                print(imagename, i)
                break
            else:
                x0 = 0
                y0 = 0
                x1 = 0
                y1 = 0
                image = cv2.imread("frames/" + imagename)
                cv2.imwrite("added/added_" + imageid + ".png", image)
                print(imagename, i)
                i = i + 1
if x0 != 0 and y0 != 0 and x1 != 0 and y1 !=0:
    for imagename in images:
        frame = cv2.imread("frames/" + imagename)
        frame = imutils.resize(frame, width=500)
        (H, W) = frame.shape[:2]
        if initBB is not None:
            (success, box) = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)
        initBB = (x0,y0,x1-x0,y1-y0)
        tracker.init(frame, initBB)

        img1 = cv2.imread('frames/' + imagename)
        img2 = cv2.imread('cola1.png')
        rows,cols,channels = img2.shape
        roi = img1[0:rows, 0:cols ]
        img2gray = cv2.bitwise_not(img2)
        img2gray = cv2.cvtColor(img2gray,cv2.COLOR_BGR2GRAY)
        ret, mask1 = cv2.threshold(img2gray, 150, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask1)
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        img2_fg = cv2.bitwise_and(img2,img2,mask = mask1)
        dst = cv2.add(img1_bg,img2_fg)
        try:
            img1[a:a+176, c:c+287 ] = cv2.add(img2_fg, img1[a:a+176, c:c+287 ])
            cv2.imwrite("added/added_" + imageid + ".png", img1)
            print(pred_class)
        except:
            print("error")
            image = cv2.imread("frames/" + imagename)
            cv2.imwrite("added/added_" + imageid + ".png", image)
            continue
    
import cv2
import numpy as np
import glob
import os
img_array = []
i = 0
for filename in os.listdir("added/"):
    try:
        filename = "added_{}.png".format(i)
        img = cv2.imread("added/" + filename)
        height, width, layers = img.shape
    # size = (1080,720)
        size = (width,height)
        img_array.append(img)
    except:
        continue
    i = i + 1
 
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('phone_added.mov',fourcc, 30, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()