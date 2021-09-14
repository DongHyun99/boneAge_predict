import cv2
import numpy as np
import time
from numpy.core.fromnumeric import clip
import pandas as pd
import PIL.Image as Image
from pandas.core.arrays.sparse import dtype
start = time.time()

#15555, 15588, 15507

train_dataset_path = 'bone_data/1.jpg'
kernel_sharpen_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
kernel_sharpen_2 = np.array([[-1, -1, -1, -1, -1],
                         [-1, 2, 2, 2, -1],
                         [-1, 2, 9, 2, -1],
                         [-1, 2, 2, 2, -1],
                         [-1, -1, -1, -1, -1]]) / 9.0
kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]])/8.0

img = cv2.imread(train_dataset_path,cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (img.shape[1]*4, img.shape[0]*4))
clahe = cv2.createCLAHE(clipLimit=15)
img = clahe.apply(img)

df = 1.035**np.asarray(img, dtype='int32')
df= np.where(df>255,255,df)
img = np.array(df, dtype=np.uint8)

ret, img = cv2.threshold(img,100,255,cv2.THRESH_BINARY_INV)
dst = cv2.distanceTransform(img, cv2.DIST_L2, 5)
# 거리 값을 0 ~ 255 범위로 정규화 ---②
dst = (dst/(dst.max()-dst.min()) * 255).astype(np.uint8)

#img=cv2.subtract(img, np.average(img.flatten()))
#clahe = cv2.createCLAHE(clipLimit=15)
#img = clahe.apply(img)
#img = cv2.bilateralFilter(img, 5 ,75,75)
#img = cv2.filter2D(img,-1,kernel_sharpen_3)
#df = 1.05**np.asarray(img, dtype='int32')
#df= np.where(df>255,255,df)
#img = np.array(df, dtype=np.uint8)





print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
cv2.imshow(train_dataset_path, dst)
cv2.waitKey(0)
