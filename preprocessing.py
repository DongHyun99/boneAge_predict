import cv2
import numpy as np
import matplotlib.pyplot as plt

#15555, 15588, 15507

train_dataset_path = 'bone_data/test/9983.png'

img = cv2.imread(train_dataset_path,cv2.IMREAD_GRAYSCALE)

val = 500
if img.shape[0]>img.shape[1]: # row>column
    val = 500/img.shape[1]
else: val = 500/img.shape[0]
img = cv2.resize(img, dsize=(0,0), fx= val, fy = val)
r = int(img.shape[0]/2)
c = int(img.shape[1]/2)
img = img[r-250 : r+250, c-250 : c+250]

clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
img = clahe.apply(img)
img = cv2.medianBlur(img,3)


mat, mask = cv2.threshold(img,np.average(img.flatten()),255,cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
max=0
num=0
for idx, contour in enumerate(contours):
    if cv2.contourArea(contour) >max:
       max = cv2.contourArea(contour)
       num = idx

# mask 이미지 제작
img2 = img.copy()
mat, mask2 = cv2.threshold(img,255,255,cv2.THRESH_BINARY)
mask2 = cv2.drawContours(mask2, contours, num, (255, 255, 255), -1)
mat, mask2 = cv2.threshold(mask2,254,255,cv2.THRESH_BINARY)
mask2 = cv2.GaussianBlur(mask2, (0,0), 3)
img = cv2.bitwise_or(img, img, mask=mask2)


cv2.imshow(train_dataset_path, img)
cv2.waitKey(0)
