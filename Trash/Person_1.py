import cv2
import numpy as np

img_path = "Dataset\\Apple\\0_100.jpg"

def remove_background(img, threshold):
    """
    This method removes background from your image
    
    :param img: cv2 image
    :type img: np.array
    :param threshold: threshold value for cv2.threshold
    :type threshold: float
    :return: RGBA image
    :rtype: np.ndarray
    """
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshed = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
    
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    mask = cv2.drawContours(threshed, cnt, 0, (0, 255, 0), 0)
    masked_data = cv2.bitwise_and(img, img, mask=mask)

    x, y, w, h = cv2.boundingRect(cnt)
    dst = masked_data[y: y + h, x: x + w]

    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(dst_gray, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(dst)

    rgba = [r, g, b, alpha]
    dst = cv2.merge(rgba, 4)
    
    dst = cv2.cvtColor(dst, cv2.COLOR_BGRA2RGB)

    return dst

## Image
img = cv2.imread(img_path)
img = cv2.resize(img, (200, 200),interpolation = cv2.INTER_AREA)
img = remove_background(img, threshold = 225)
cv2.imshow("Orginal Image", img)

# Convert Image to Gray Image
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image", img_gray)

# Apply Gaussian Blur to Image
img_blur = cv2.GaussianBlur(img_gray, (7,7), 1)
cv2.imshow("Blur Image", img_blur)

# Canny Edge Detection
img_canny = cv2.Canny(img_blur, 50, 150)
cv2.imshow("Canny Image", img_canny)

# Edge Detector
kernel = np.ones((9, 9), dtype = np.uint8)
img_dilate = cv2.dilate(img_canny, kernel, iterations = 2)
cv2.imshow("Image", img_dilate)

cv2.waitKey(0)