# import the necessary packages
import cv2
 
# Opens the Video file
cap= cv2.VideoCapture('phone.mov')
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('frames/book_'+str(i)+'.jpg',frame)
    i+=1
 
cap.release()

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
    # gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    imageid = imagename.split("_")[1].replace(".jpg", "")
    image = cv2.imread("frames/" + imagename)
    gray = cv2.bitwise_not(image)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGRA2GRAY)


    # threshold the image and show it
    (T, thresh) = cv2.threshold(gray, args["threshold"], 255, cv2.THRESH_BINARY)
    (_, contours) = cv2.findContours(image = thresh, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
    # for (i, c) in enumerate(contours):
    #     print("\tSize of contour %d: %d" % (i, len(c)))
    (contours, hierarchy) = cv2.findContours(image = thresh, mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(shape = image.shape, dtype = "uint8")
    # print(contours)
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img = mask, 
            pt1 = (x, y),
            pt2 = (x + w, y + h), 
            color = (255, 255, 255), 
            thickness = -1)
    # print(threshName)
    image1 = cv2.bitwise_and(src1 = image, src2 = mask)
    image1 = cv2.bitwise_not(image1)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image1=255-image1
    # cv2.imwrite("test_gray_im.png", image1)
    nz = cv2.findNonZero(image1)
    try:
        a = int(nz[:,0,0].mean())
        b = nz[:,0,0].max()
        c = int(nz[:,0,1].mean())
        d = nz[:,0,1].max()
        c0 = (a+b)/2
        c1 = (c+d)/2
        c0 = int(c0)
        c1 = int(c1)
    except:
        a = 0
        b = 0
        c = 0
        d = 0
        c0 = 0
        c1 = 0
    # print(c0, c1)
    img1 = cv2.imread('frames/' + imagename)
    img2 = cv2.imread('cola.png')
    # I want to put logo on top-left corner, So I create a ROI
    rows,cols,channels = img2.shape
    roi = img1[0:rows, 0:cols ]
    # Now create a mask of logo and create its inverse mask also
    # img2gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    img2gray = cv2.bitwise_not(img2)
    img2gray = cv2.cvtColor(img2gray,cv2.COLOR_BGR2GRAY)

    ret, mask1 = cv2.threshold(img2gray, 150, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask1)
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask1)
    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    try:
        img1[a:a+100, c:c+100 ] = cv2.add(img2_fg, img1[a:a+100, c:c+100 ])
        cv2.imwrite("added/added_" + imageid + ".png", img1)
    except:
        cv2.imwrite("added/added_" + imageid + ".png", image)
        # print("Error")
        continue
    # img1[c0:c0+100, c1:c1+100 ] = cv2.add(img2_fg, img1[c0:c0+100, c1:c1+100 ])
    i = i + 1
    # cv2.waitKey(0)
import cv2
import numpy as np
import glob
import os
img_array = []
i = 0
for filename in os.listdir("added/"):
    filename = "added_{}.png".format(i)
    img = cv2.imread("added/" + filename)
    height, width, layers = img.shape
    # size = (1080,720)
    size = (width,height)
    img_array.append(img)
    i = i + 1
 
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('project1.mov',fourcc, 30, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()