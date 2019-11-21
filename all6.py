import pandas as pd
import cv2
import os
import sys
import numpy as np
from PIL import Image

df = pd.read_csv("prediction_2.csv")

for row in df.values:
    img_path = row[1]
    img = Image.open(img_path)
    img1 = img.crop((row[2], row[3], row[4], row[5]))
    img1.save("cropped.png")
    img1 = cv2.imread("cropped.png")
    
    img2 = cv2.imread('cola2.png')
    
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # gray = cv2.bilateralFilter(gray, 11, 17, 17)

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(gray,kernel,iterations = 2)
    kernel = np.ones((4,4),np.uint8)
    dilation = cv2.dilate(erosion,kernel,iterations = 2)
    # edged = cv2.Canny(dilation, 100, 200)
    # ret, thresh = cv2.threshold(gray, 150, 255, 0)
    edged = cv2.Canny(dilation, 100, 200)
    cv2.imwrite("edged.jpg", edged)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    cnts = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areas.append(area)
        cnts.append([area, cnt])
    max_area = max(areas)
    average = sum(areas)/len(areas)
    for i in range(len(cnts)):
        # if cnts[i][0] > average:
        if cnts[i][0] == max_area:
        # if cnts[i][0] > 2000:
            cnt = cnts[i][1]
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # area = cv2.contourArea(cnt)
            # approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
            # x = approx.ravel()[0]
            # y = approx.ravel()[1]
            # if len(approx) == 4:
            #     cv2.drawContours(img1, [approx], 0, (0, 255, 0), 3)
            # cv2.drawContours(img1,[box],0,(0,0,255),3)
            recta = box.tolist() 
            img2 = cv2.imread('cola2.jpeg')
            img2gray = cv2.bitwise_not(img2)
            img2gray = cv2.cvtColor(img2gray,cv2.COLOR_BGR2GRAY)
            ret, mask1 = cv2.threshold(img2gray, 150, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask1)
            img2 = cv2.bitwise_and(img2,img2,mask = mask1)

            pts_src = np.array([[0,0], [400,0], [400,400], [0,400]])
            pts_dst = np.array([recta[1], recta[2], recta[3], recta[0]])
            h, status = cv2.findHomography(pts_src, pts_dst)
            im_out = cv2.warpPerspective(img2, h, (img2.shape[1],img2.shape[0]))
            cv2.fillConvexPoly(img1, pts_dst.astype(int), 0, 16)
            img1 = cv2.resize(img1, (im_out.shape[1], im_out.shape[0]))
            # im_out = cv2.resize(im_out, (img1.shape[1], img1.shape[0]))
            im_out1 = img1 + im_out
            
            orig_img = cv2.imread(img_path)
            row[2] = int(row[2])
            row[3] = int(row[3])
            # orig_img[row[2]:row[2]+im_out1.shape[1], row[3]:row[3]+im_out1.shape[0]] = cv2.add(im_out1, orig_img[row[2]:row[2]+im_out1.shape[1], row[3]:row[3]+im_out1.shape[0]])
            orig_img[row[2]:row[2]+im_out1.shape[0], row[3]:row[3]+im_out1.shape[1]] = im_out1
            filname = "added/added_{}.jpg".format(row[0])
            # cv2.imwrite(filname, im_out)
            cv2.imwrite(filname, im_out1)
    print(row[0])
    

import cv2
import numpy as np
import glob
import os
img_array = []
i = 0
for filename in os.listdir("added/"):
    filename = "added_{}.jpg".format(i)
    img = cv2.imread("added/" + filename)
    img = cv2.resize(img, (550,250))
    height, width, layers = img.shape
# size = (1080,720)
    size = (width,height)
    img_array.append(img)
    i = i + 1
 
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('display1.mov',fourcc, 5, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()