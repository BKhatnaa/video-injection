import pandas as pd
import cv2
import os
import sys
import numpy as np
from PIL import Image

df = pd.read_csv("prediction_1.csv")

for row in df.values:
    img_path = row[1]
    img = Image.open(img_path)
    img1 = img.crop((row[2], row[3], row[4], row[5]))
    img1.save("cropped.png")
    img1 = cv2.imread("cropped.png")
    
    img2 = cv2.imread('cola2.png')
    
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(gray,kernel,iterations = 2)
    kernel = np.ones((4,4),np.uint8)
    dilation = cv2.dilate(erosion,kernel,iterations = 2)
    edged = cv2.Canny(dilation, 30, 200)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(cnt) for cnt in contours]
    rects = sorted(rects,key=lambda  x:x[1],reverse=True)

    areas = []
    conts = []
    for rect in rects:
        x,y,w,h = rect
        area = w * h
        conts.append([area, x, y, w, h])
        areas.append(area)
    max_area = max(areas)
    print(max_area)
    for i in conts:
        if i[0] == max_area:
            print("done", i)
            orig_img = cv2.imread(row[1])
            x = i[1]
            y = i[2]
            w = i[3]
            h = i[4]
            cv2.rectangle(img1, (x,y), (x+w, y+h), (0,255,0), 2)
            filname = "added/added_{}.jpg".format(row[0])
            cv2.imwrite(filname, img1)
    # orig_img = cv2.imread(row[1])
    # cv2.rectangle(img1, (x,y), (x+w, y+h), (0,255,0), 2)
    # cv2.imwrite("binary.jpg", img1)
    # break
    

import cv2
import numpy as np
import glob
import os
img_array = []
i = 0
for filename in os.listdir("added/"):
    filename = "added_{}.jpg".format(i)
    img = cv2.imread("added/" + filename)
    img = cv2.resize(img, (550,300))
    height, width, layers = img.shape
# size = (1080,720)
    size = (width,height)
    img_array.append(img)
    i = i + 1
 
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('display.mov',fourcc, 30, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()