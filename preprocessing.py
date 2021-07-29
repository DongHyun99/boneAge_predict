import cv2
import glob
import pandas as pd
import numpy as np
from PIL import Image

train_dataset_path = 'bone_data/train/'




img = cv2.imread(train_dataset_path+'3502.png',cv2.IMREAD_GRAYSCALE)

img = cv2.medianBlur(img,5)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img = clahe.apply(img)
# img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,19,9)
# th, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# img = cv2.fastNlMeansDenoising(img, 20,7,21)

img = cv2.resize(img, (1000,1000))

cv2.imshow('3502', img)
cv2.waitKey(0)
