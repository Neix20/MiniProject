import cv2
import numpy as np

img_path = "lena.jpg"
img = cv2.imread(img_path)

cv2.imshow("Images", img)
cv2.waitKey(0)