import cv2
import numpy as np

img_path = "Inputs/Images/Shapes.png"
img = cv2.imread(img_path)

# Contour Detection
def get_contour(img, ori_img):
    ind = 0
    contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            roi = ori_img[y: y + h, x: x + w]
            cv2.imshow(f"Cropped Image {ind}", roi)
            ind += 1

# Convert Image to gray
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert Image to Blur
img_blur = cv2.GaussianBlur(img_gray, (7,7), 1)

# Convert Image to Canny
img_canny = cv2.Canny(img_blur, 50, 150)

# Edge Detector
kernel = np.ones((5, 5), dtype = np.uint8)
img_dilate = cv2.dilate(img_canny, kernel, iterations = 1)

get_contour(img_dilate, img)

cv2.imshow("Image", img_dilate)

cv2.waitKey(0)