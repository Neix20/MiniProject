import cv2
import numpy as np

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
img_path = "Dataset\\Apple\\0_100.jpg"
img = cv2.imread(img_path)
img = cv2.resize(img, (200, 200),interpolation = cv2.INTER_AREA)
img = remove_background(img, threshold = 225)
cv2.imshow("Orginal Image", img)

## Test Image
test_img_path = "Test_Images\\Apple\\apple1.jpg"
test_img = cv2.imread(test_img_path)
test_img = cv2.resize(test_img, (200, 200),interpolation = cv2.INTER_AREA)
test_img = remove_background(test_img, threshold = 225)
cv2.imshow("Test Image", test_img)

## Contrast Image
## Brighten Up Image
img = cv2.imread(img_path)
img = cv2.resize(img, (200, 200),interpolation = cv2.INTER_AREA)
img = remove_background(img, threshold = 225)
img = cv2.convertScaleAbs(img, alpha=1.5, beta=20)
cv2.imshow("Brightened Image", img)

## Color Enhancement
img = cv2.imread(img_path)
img = cv2.resize(img, (200, 200),interpolation = cv2.INTER_AREA)
img = remove_background(img, 225)
img = img / 255.0

r, g, b = cv2.split(img)
img_sum = r + g + b
CR, CG, CB = cv2.divide(r, img_sum), cv2.divide(g, img_sum), cv2.divide(b, img_sum)

img = cv2.merge((CR, CG, CB))
img = np.uint8(img * 255)
img = cv2.convertScaleAbs(img, alpha=1.5, beta=20)
cv2.imshow("Color Enhancement", img)

cv2.waitKey(0)