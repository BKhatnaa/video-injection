import cv2
import numpy as np
image = cv2.imread("frames/book_89.jpg")
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.cvtColor(src = image, code = cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(src = gray, 
    ksize = (5, 5), 
    sigmaX = 0)
(t, binary) = cv2.threshold(src = blur,
    thresh = 245, 
    maxval = 255, 
    type = cv2.THRESH_BINARY)
(_, contours) = cv2.findContours(image = binary, 
    mode = cv2.RETR_EXTERNAL,
    method = cv2.CHAIN_APPROX_SIMPLE)
print("Found %d objects." % len(contours))
for (i, c) in enumerate(contours):
    print("\tSize of contour %d: %d" % (i, len(c)))
# cv2.drawContours(image = image, 
#     contours = contours, 
#     contourIdx = -1, 
#     color = (0, 0, 255), 
#     thickness = 5)
(contours, hierarchy) = cv2.findContours(image = binary, 
    mode = cv2.RETR_TREE,
    method = cv2.CHAIN_APPROX_SIMPLE)
# create all-black mask image
mask = np.zeros(shape = image.shape, dtype = "uint8")
for c in contours:
    (x, y, w, h) = cv2.boundingRect(c)

    cv2.rectangle(img = mask, 
        pt1 = (x, y), 
        pt2 = (x + w, y + h), 
        color = (255, 255, 255), 
        thickness = -1)
image = cv2.bitwise_and(src1 = image, src2 = mask)

# _, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# img_contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

# img_contours = sorted(img_contours, key=cv2.contourArea)
# for i in img_contours:
#     if cv2.contourArea(i) > 100:
#         break
# mask = np.zeros(img.shape[:2], np.uint8)

# cv2.drawContours(mask, [i],-1, 255, -1)

# new_img = cv2.bitwise_and(img, img, mask=mask)
# cv2.imwrite("test.png", new_img)
cv2.imwrite("test_gray.png", image)
# cv2.imshow("Original Image", img)

# cv2.imshow("Image with background removed", new_img)
# cv2.waitKey(0)

