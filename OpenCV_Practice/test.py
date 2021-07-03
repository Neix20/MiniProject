import cv2
import numpy as np

# img_path = "lena.jpg"
# img = cv2.imread(img_path)

# cv2.imshow("Images", img)
# cv2.waitKey(0)

# vid_path = "Love.mp4"
# cap = cv2.VideoCapture(vid_path)

# while True:
#     success, img = cap.read()
#     cv2.imshow("Video", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# camera_id = 0
# cap = cv2.VideoCapture(camera_id)

# while True:
#     success, img = cap.read()
#     cv2.imshow("Video", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# Gray scale Image
# img_path = "Shapes.png"
# img = cv2.imread(img_path)
# cv2.imshow("Normal Image", img)

# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Grayscale Image", img_gray)

# Blur Image
# img_blur = cv2.GaussianBlur(img_gray, (7, 7), 2)
# cv2.imshow("Image Blur", img_blur)

# Edge Detector
# img_canny = cv2.Canny(img_gray, 50 ,150)
# cv2.imshow("Image Canny", img_canny)

# Thickness
# kernel = np.ones((5,5), dtype=np.uint8)
# img_dilate = cv2.dilate(img_canny, kernel, iterations = 1)
# cv2.imshow("Image Dilation", img_dilate)
# cv2.waitKey(0)

# img_path = "lena.jpg"
# img = cv2.imread(img_path)
# cv2.imshow("Normal Images", img)

# # Resize Image - cv2.resize()
# img_resize = cv2.resize(img, (256, 256))
# cv2.imshow("Resized Image", img_resize)

# # Crop Image - List Slicing
# # Row, Column
# img_crop = img[0:200, 200:400]
# cv2.imshow("Image Cropping", img_crop)

# cv2.waitKey(0)

# How to Crop Images
img_path = "Shapes.png"
img = cv2.imread(img_path)
cv2.imshow("Normal Image", img)

# Step 1: Convert Image to Gray
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 1.5: Blur Image to remove noise
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 2)

# Step 2: Use Edge Detector => Image canny
img_canny = cv2.Canny(img_blur, 50, 100)

# Step 3: Use Image Dilation => Increase Line thickness
kernel = np.ones((5, 5), dtype=np.uint8)
img_dilate = cv2.dilate(img_canny, kernel, iterations = 1)

# Step 4: Get the coordinates of the points of the image
# Contour
def get_contour(img_dilate, img):
    contour, hier = cv2.findContours(img_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for ind, cnt in enumerate(contour):
        x, y, w, h = cv2.boundingRect(cnt)
        img_crop = img[y: y + h, x : x + w]
        cv2.imshow(f"Shape {ind + 1}: ", img_crop)

get_contour(img_dilate, img)

# Step 5: Crop Image out


cv2.waitKey(0)