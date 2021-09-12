import cv2
import numpy as np
import glob

'''
#15555, 15588, 15507
fname = glob.glob('bone_data/test/*.png')
for k, n in enumerate(fname):
    train_dataset_path = n
    img = cv2.imread(train_dataset_path,cv2.IMREAD_GRAYSCALE)
    img=cv2.subtract(img, np.average(img.flatten())+3)
    clahe = cv2.createCLAHE(clipLimit=15)
    img = clahe.apply(img)

    sub = img.shape[0]-img.shape[1]
    if sub < 0:
        img = cv2.copyMakeBorder(img, int(-sub/2), int(-sub/2), 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
    else:
        img = cv2.copyMakeBorder(img, 0, 0, int(sub/2), int(sub/2), cv2.BORDER_CONSTANT, value=[0,0,0]) 

    img = cv2.resize(img, (500,500))
    cv2.imwrite('samples/'+str(k)+'.png', img)
    #cv2.imshow(train_dataset_path, img)
    #cv2.waitKey(0)
'''
train_dataset_path='bone_data/test/4475.png'
img = cv2.imread(train_dataset_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.subtract(gray, np.average(img.flatten())+3)
clahe = cv2.createCLAHE(clipLimit=15)
gray = clahe.apply(gray)

ret, gray = cv2.threshold(gray, 25, 26, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    img = cv2.drawContours(img, [contour], -1, (0,0,255), 2)

#img = cv2.resize(img, (500,500))
cv2.imshow(train_dataset_path, img)
cv2.waitKey(0)