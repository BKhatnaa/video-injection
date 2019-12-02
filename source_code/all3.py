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
# Opens the Video file
cap= cv2.VideoCapture('turshilt_2.mov')
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('frames/cup_'+str(i)+'.jpg',frame)
    i+=1
 
cap.release()


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
preds = []
for imagename in images:
    img_path = "frames/" + "cup_" + str(i) + ".jpg"
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
    for class_index in range(len(pred_class)):
        # imageid = imagename.split("_")[1].replace(".jpg", "")
        if pred_class[class_index] == 'cell phone':
            x0 = pred_boxes[class_index][0][0]
            y0 = pred_boxes[class_index][0][1]
            x1 = pred_boxes[class_index][1][0]
            y1 = pred_boxes[class_index][1][1]
            print("Found", x0, y0, x1, y1)
            # a = int((x0+x1)/2)
            # c = int((y0+y1)/2)
    if x0 != None and y0 != None and x1 != None and y1 != None: 
        preds.append([i, img_path, x0, y0, x1, y1])
    else:
        preds.append([i, img_path, 0, 0, 0, 0])
    print(img_path, i)
    i = i + 1
# print(preds)
df = pd.DataFrame(preds, columns=["id", "path", "x0", "y0", "x1", "y1"])
df.to_csv("prediction_2.csv", index=False)
# import cv2
# import numpy as np
# import glob
# import os
# img_array = []
# i = 0
# for filename in os.listdir("added/"):
#     try:
#         filename = "added_{}.png".format(i)
#         img = cv2.imread("added/" + filename)
#         height, width, layers = img.shape
#     # size = (1080,720)
#         size = (width,height)
#         img_array.append(img)
#     except:
#         continue
#     i = i + 1
 
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# # fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('phone_added.mov',fourcc, 30, size)
 
# for i in range(len(img_array)):
#     out.write(img_array[i])
# out.release()