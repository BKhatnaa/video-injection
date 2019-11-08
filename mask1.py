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
	("THRESH_BINARY", cv2.THRESH_BINARY),
	("THRESH_BINARY_INV", cv2.THRESH_BINARY_INV),
	("THRESH_TRUNC", cv2.THRESH_TRUNC),
	("THRESH_TOZERO", cv2.THRESH_TOZERO),
	("THRESH_TOZERO_INV", cv2.THRESH_TOZERO_INV)]
 
# loop over the threshold methods
for (threshName, threshMethod) in methods:
	# threshold the image and show it
    (T, thresh) = cv2.threshold(gray, args["threshold"], 255, threshMethod)
    (_, contours) = cv2.findContours(image = thresh, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
    for (i, c) in enumerate(contours):
        print("\tSize of contour %d: %d" % (i, len(c)))
    (contours, hierarchy) = cv2.findContours(image = thresh, mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(shape = image.shape, dtype = "uint8")
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img = mask, 
            pt1 = (x, y), 
            pt2 = (x + w, y + h), 
            color = (255, 255, 255), 
            thickness = -1)
    image = cv2.bitwise_and(src1 = image, src2 = mask)
    cv2.imwrite("test_gray_" + threshName + ".png", image)
	# cv2.waitKey(0)