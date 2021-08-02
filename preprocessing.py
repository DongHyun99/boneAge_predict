import cv2
import numpy as np

#15555, 15588

train_dataset_path = 'bone_data/train/'

img = cv2.imread(train_dataset_path+'15584.png')
img = cv2.resize(img, (500,500))
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
'''
img = cv2.medianBlur(img,5)

th,img2 = cv2.threshold(img,55,255,cv2.THRESH_OTSU)

# img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,19,9)
# th, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# img = cv2.fastNlMeansDenoising(img, 20,7,21)

r, mask = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
# dst = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
img = cv2.bitwise_xor(img,mask)
'''
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_gray = clahe.apply(img_gray)

r, mask = cv2.threshold(img_gray, 40, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
max=0
num=0
for idx, contour in enumerate(contours):
    if contour.size >max:
        max = contour.size
        num = idx
img = cv2.drawContours(img, contours, idx, (0, 255, 0), 3)
print(contours)

cv2.imshow('3502', img)
cv2.waitKey(0)
