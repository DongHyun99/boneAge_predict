import cv2
import numpy as np
import time
from numpy.core.fromnumeric import clip
import pandas as pd
import PIL.Image as Image
from pandas.core.arrays.sparse import dtype
start = time.time()

#15555, 15588, 15507

train_dataset_path = 'bone_data/test/4438.png'
kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]])/8.0

img = cv2.imread(train_dataset_path,cv2.IMREAD_GRAYSCALE)
img=cv2.subtract(img, np.average(img.flatten()))
clahe = cv2.createCLAHE(clipLimit=15)
img = clahe.apply(img)
#img = cv2.bilateralFilter(img, 5 ,75,75)
#img = cv2.filter2D(img,-1,kernel_sharpen_3)
#df = 1.05**np.asarray(img, dtype='int32')
#df= np.where(df>255,255,df)
#img = np.array(df, dtype=np.uint8)



sub = img.shape[0]-img.shape[1]
if sub < 0:
    img = cv2.copyMakeBorder(img, int(-sub/2), int(-sub/2), 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
else:
    img = cv2.copyMakeBorder(img, 0, 0, int(sub/2), int(sub/2), cv2.BORDER_CONSTANT, value=[0,0,0]) 

img = cv2.resize(img, (500,500))

print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
cv2.imshow(train_dataset_path, img)
cv2.waitKey(0)
