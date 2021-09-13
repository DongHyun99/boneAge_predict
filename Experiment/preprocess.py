import cv2
import numpy as np
import glob


#15555, 15588, 15507
fname = glob.glob('bone_data/test/*.png')
for k, n in enumerate(fname):
    train_dataset_path = n
    img = cv2.imread(train_dataset_path,cv2.IMREAD_GRAYSCALE) 
    img = cv2.subtract(img, np.average(img.flatten()))
    clahe = cv2.createCLAHE(clipLimit=15)
    img = clahe.apply(img)
    img_color = cv2.imread(train_dataset_path)

    sub = img.shape[0]-img.shape[1]
    if sub < 0:
        img = cv2.copyMakeBorder(img, int(-sub/2), int(-sub/2), 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
        img_color = cv2.copyMakeBorder(img_color, int(-sub/2), int(-sub/2), 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
    else:
        img = cv2.copyMakeBorder(img, 0, 0, int(sub/2), int(sub/2), cv2.BORDER_CONSTANT, value=[0,0,0])
        img_color = cv2.copyMakeBorder(img_color, 0, 0, int(sub/2), int(sub/2), cv2.BORDER_CONSTANT, value=[0,0,0]) 

    img_color = cv2.resize(img_color, (500,500))
    ret, img_binary = cv2.threshold(cv2.resize(img, (500,500)), np.average(img.flatten()),255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt)>=40000:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img_color, [box], 0, (255, 0, 0), 3)  # blue
    cv2.imwrite('samples/'+str(k)+'.png', img_color)
'''
train_dataset_path='bone_data/test/4533.png'
img = cv2.imread(train_dataset_path, cv2.IMREAD_GRAYSCALE)
img = cv2.subtract(img, np.average(img.flatten()))
clahe = cv2.createCLAHE(clipLimit=15)
img = clahe.apply(img)

sub = img.shape[0]-img.shape[1]
if sub < 0:
    img = cv2.copyMakeBorder(img, int(-sub/2), int(-sub/2), 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
else:
    img = cv2.copyMakeBorder(img, 0, 0, int(sub/2), int(sub/2), cv2.BORDER_CONSTANT, value=[0,0,0]) 

img = cv2.resize(img, (500,500))
cv2.imshow(train_dataset_path, img)
cv2.waitKey(0)
'''