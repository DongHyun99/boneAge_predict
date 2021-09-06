import cv2
import numpy as np

#15555, 15588, 15507

train_dataset_path = 'bone_data/test/4476.png'

img = cv2.imread(train_dataset_path,cv2.IMREAD_GRAYSCALE)

clahe = cv2.createCLAHE()
img = clahe.apply(img)


sub = img.shape[0]-img.shape[1]
if sub < 0:
    img = cv2.copyMakeBorder(img, int(-sub/2), int(-sub/2), 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
else:
    img = cv2.copyMakeBorder(img, 0, 0, int(sub/2), int(sub/2), cv2.BORDER_CONSTANT, value=[0,0,0]) 

img = cv2.resize(img, (500,500))


cv2.imshow(train_dataset_path, img)
cv2.waitKey(0)
