import cv2
import glob
import pandas as pd
import numpy as np

train_dataset_path = 'bone_data/train/'

img = cv2.imread(train_dataset_path+'15605.png',cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (1000,1000))

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img = clahe.apply(img)

img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,5)
img = cv2.fastNlMeansDenoising(img, None, 60, 7, 21)
cv2.imshow('Binary', img)
cv2.waitKey(0)