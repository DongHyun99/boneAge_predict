import cv2
import numpy as np
import matplotlib.pyplot as plt

#15555, 15588, 15507

train_dataset_path = 'bone_data/train/5322.png'

img = cv2.imread(train_dataset_path,cv2.IMREAD_GRAYSCALE)

#img = cv2.subtract(img, np.average(img.flatten())-40)
clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8,8))
img = clahe.apply(img)

val = 500
if img.shape[0]>img.shape[1]: # row>column
    val = 500/img.shape[1]
else: val = 500/img.shape[0]
img = cv2.resize(img, dsize=(0,0), fx= val, fy = val)

r = int(img.shape[0]/2)
c = int(img.shape[1]/2)
img = img[r-250 : r+250, c-250 : c+250]

# img = cv2.resize(img, (500,500))

cv2.imshow(train_dataset_path, img)
cv2.waitKey(0)