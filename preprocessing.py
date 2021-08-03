import cv2
import numpy as np
import matplotlib.pyplot as plt

#15555, 15588

train_dataset_path = 'bone_data/train/15608.png'

img = cv2.imread(train_dataset_path,cv2.IMREAD_GRAYSCALE)

val = 500
if img.shape[0]>img.shape[1]: # row>column
    val = 500/img.shape[1]
else: val = 500/img.shape[0]
img = cv2.resize(img, dsize=(0,0), fx= val, fy = val)
r = int(img.shape[0]/2)
c = int(img.shape[1]/2)
img = img[r-250 : r+250, c-250 : c+250]
'''
# img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,19,9)
# img = cv2.fastNlMeansDenoising(img, 20,7,21)

r, mask = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
# dst = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
img = cv2.bitwise_xor(img,mask)
'''


# mat, mask = cv2.threshold(img,np.average(img.flatten()),255,cv2.THRESH_BINARY)
# img = cv2.bitwise_or(img,img,mask=mask)
# clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8,8))
# img = clahe.apply(img)
# img = cv2.medianBlur(img,3)



'''
mat, mask = cv2.threshold(img,np.average(img.flatten()),255,cv2.THRESH_BINARY_INV)
img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)

contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
max=0
num=0
for idx, contour in enumerate(contours):
    if cv2.contourArea(contour) >max:
       max = cv2.contourArea(contour)
       num = idx
result = cv2.drawContours(result, contours, num, (0, 255, 0), 3)
print(max)
'''
cv2.imshow(train_dataset_path, img)
cv2.waitKey(0)
