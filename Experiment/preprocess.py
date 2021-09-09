import cv2
import numpy as np
import time
start = time.time()

#15555, 15588, 15507

train_dataset_path = 'bone_data/test/4420.png'
kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]])/8.0

img = cv2.imread(train_dataset_path,cv2.IMREAD_GRAYSCALE)

img = cv2.filter2D(img,-1,kernel_sharpen_3)
clahe = cv2.createCLAHE()
img = clahe.apply(img)
img = cv2.bilateralFilter(img, 9 ,75,75)


sub = img.shape[0]-img.shape[1]
if sub < 0:
    img = cv2.copyMakeBorder(img, int(-sub/2), int(-sub/2), 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
else:
    img = cv2.copyMakeBorder(img, 0, 0, int(sub/2), int(sub/2), cv2.BORDER_CONSTANT, value=[0,0,0]) 

img = cv2.resize(img, (500,500))


print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
cv2.imshow(train_dataset_path, img)
cv2.waitKey(0)
