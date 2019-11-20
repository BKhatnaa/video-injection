import pandas as pd
import cv2
import os
import sys
import numpy as np
from PIL import Image

df = pd.read_csv("prediction1.csv")

for row in df.values:
    img_path = row[1]
    img = Image.open(img_path)
    a = int((int(row[2]) + int(row[4]))/2)
    b = int((int(row[3]) + int(row[5]))/2)
    img1 = cv2.imread(img_path)
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
    if a != 0 and b != 0:
        try:
            img1[a:a+176, b:b+287 ] = cv2.add(img2_fg, img1[a:a+176, b:b+287 ])
            cv2.imwrite("added/added_" + str(row[0]) + ".jpg", img1)
        except Exception as e:
            print("error", e)
            try:
                a = a - 200
                b = b - 200
                img1[a:a+176, b:b+287 ] = cv2.add(img2_fg, img1[a:a+176, b:b+287])
            except:
                try:
                    a = a - 200
                    img1[a:a+176, b:b+287 ] = cv2.add(img2_fg, img1[a:a+176, b:b+287 ])
                except:
                    try:
                        b = b - 200
                        img1[a:a+176, b:b+287 ] = cv2.add(img2_fg, img1[a:a+176, b:b+287 ])
                    except:
                        image = cv2.imread(row[1])
                        cv2.imwrite("added/added_" + str(row[0]) + ".jpg", image)
                        continue
    else:
        image = cv2.imread(row[1])
        cv2.imwrite("added/added_" + str(row[0]) + ".jpg", image)
    print(row[1])

import cv2
import numpy as np
import glob
import os
img_array = []
i = 0
for filename in os.listdir("added/"):
    try:
        filename = "added_{}.jpg".format(i)
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
out = cv2.VideoWriter('phone_added1.mov',fourcc, 30, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()