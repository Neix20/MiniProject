import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def convert_val_to_bin(x):
    if x >= 0 and x < 64:
        return 0
    elif x >= 64 and x < 128:
        return 64
    elif x >= 128 and x < 192:
        return 128
    else:
        return 192

def pipeline_1d(cv_img, convert_func):
    img_arr = np.array(cv_img)
    img_flatten = img_arr.reshape(1, -1).T
    img_squeeze = np.squeeze(img_flatten)
    img_convert = np.vectorize(convert_func)(img_squeeze)
    return img_convert

## Image
img_path = "Dataset\\Apple\\0_100.jpg"
img = cv2.imread(img_path)
img = cv2.resize(img, (200, 200),interpolation = cv2.INTER_AREA)
img = remove_background(img, threshold = 225)
img = np.array(img)
img = img.reshape(-1, 3)

tmp_df = pd.DataFrame(img, columns = ["b", "g", "r"])
plt.hist(tmp_df["b"],bins=256, color="blue")
plt.hist(tmp_df["g"],bins=256, color="green")
plt.hist(tmp_df["r"],bins=256, color="red")
plt.show()

## New Image
img = cv2.imread(img_path)
img = cv2.resize(img, (200, 200),interpolation = cv2.INTER_AREA)
img = remove_background(img, threshold = 225)
new_img = pipeline_1d(img, convert_val_to_bin)
new_img = new_img.reshape(200, 186, 3)
new_img = new_img.reshape(-1, 3)
tmp_df = pd.DataFrame(new_img, columns = ["b", "g", "r"])
plt.hist(tmp_df["r"],color="red", bins=4)
plt.hist(tmp_df["g"],color="green", bins=4)
plt.hist(tmp_df["b"],color="blue", bins=4)
plt.show()