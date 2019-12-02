# import the necessary packages
import argparse
import cv2
import numpy as np
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be thresholded")
ap.add_argument("-t", "--threshold", type = int, default = 128,
	help = "Threshold value")
args = vars(ap.parse_args())
 
# load the image and convert it to grayscale
image = cv2.imread(args["image"])
# gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
gray = cv2.bitwise_not(image)
gray = cv2.cvtColor(gray, cv2.COLOR_BGRA2GRAY)

# initialize the list of threshold methods
methods = [
	("THRESH_BINARY", cv2.THRESH_BINARY)]
	# ("THRESH_BINARY_INV", cv2.THRESH_BINARY_INV),
	# ("THRESH_TRUNC", cv2.THRESH_TRUNC),
	# ("THRESH_TOZERO", cv2.THRESH_TOZERO),
	# ("THRESH_TOZERO_INV", cv2.THRESH_TOZERO_INV)]
 
# loop over the threshold methods
for (threshName, threshMethod) in methods:
	# threshold the image and show it
    (T, thresh) = cv2.threshold(gray, args["threshold"], 255, threshMethod)
    (_, contours) = cv2.findContours(image = thresh, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
    for (i, c) in enumerate(contours):
        print("\tSize of contour %d: %d" % (i, len(c)))
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
    print(threshName)
    image = cv2.bitwise_and(src1 = image, src2 = mask)
    image1 = cv2.bitwise_not(image)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image1=255-image1
    cv2.imwrite("test_gray_im.png", image1)
    nz = cv2.findNonZero(image1)
    a = int(nz[:,0,0].mean())
    b = nz[:,0,0].max()
    c = int(nz[:,0,1].mean())
    d = nz[:,0,1].max()
    c0 = (a+b)/2
    c1 = (c+d)/2
    c0 = int(c0)
    c1 = int(c1)
    print(c0, c1)
    img1 = cv2.imread('frames/book_89.jpg')
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
    img1[a:a+100, c:c+100 ] = cv2.add(img2_fg, img1[a:a+100, c:c+100 ])
    # img1[c0:c0+100, c1:c1+100 ] = cv2.add(img2_fg, img1[c0:c0+100, c1:c1+100 ])
    cv2.imwrite("test_gray_" + threshName + ".png", img1)
	# cv2.waitKey(0)