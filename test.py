import cv2
import numpy as np

img_path = "Test_Images/banana3.jpg"

## Image
img = cv2.imread(img_path)

# Convert Image to Gray Image
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to Image
img_blur = cv2.GaussianBlur(img_gray, (13, 13), 1)

# Canny Edge Detection
img_canny = cv2.Canny(img_blur, 25, 75)

# Edge Detector
kernel = np.ones((9, 9), dtype = np.uint8)
img_dilate = cv2.dilate(img_canny, kernel, iterations = 2)

def get_contour(img, ori_img):
    ind = 0
    contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            roi = ori_img[y: y + h, x: x + w]
            cv2.imshow(f"Cropped Image {ind}", roi)
            cv2.imwrite("test_banana.jpg", roi)
            ind += 1
get_contour(img_dilate, img)

cv2.imshow("Image", img_dilate)
cv2.waitKey(0)