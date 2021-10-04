import cv2
import numpy as np
import pandas as pd
import imutils
'''
#data load
train_img_path = 'bone_data/train/'
train_csv_path = 'bone_data/training_dataset.csv'

# dataset setting
train_data = pd.read_csv(train_csv_path)
train_data.iloc[:, 1:3] = train_data.iloc[:, 1:3].astype(np.float)

def rotation(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img

for idx, data in train_data.iterrows():
    imlist = []
    angles = [-15, -10, -5, 0, 5, 10, 15]
    id = data['id']
    img = cv2.imread(train_img_path+str(int(id))+'.png', cv2.IMREAD_GRAYSCALE)

    img=cv2.subtract(img, np.average(img.flatten()))
    clahe = cv2.createCLAHE(clipLimit=15)
    img = clahe.apply(img)

    # img 비율을 맞춰주기 위한 pad 추가
    sub = img.shape[0]-img.shape[1]
    if sub < 0:
        img = cv2.copyMakeBorder(img, int(-sub/2), int(-sub/2), 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
    else:
        img = cv2.copyMakeBorder(img, 0, 0, int(sub/2), int(sub/2), cv2.BORDER_CONSTANT, value=[0,0,0])

    for angle in angles:
        aug1 = rotation(img, angle)
        aug1 = cv2.resize(aug1, (500,500))
        aug2 = imutils.translate(aug1, 50, 0)
        aug3 = imutils.translate(aug1, -50, 0)
        cv2.imwrite('D:/train/'+str(int(id))+'-{}-rotate{}.png'.format('original', angle), aug1)
        cv2.imwrite('D:/train/'+str(int(id))+'-{}-rotate{}.png'.format('right50', angle), aug2)
        cv2.imwrite('D:/train/'+str(int(id))+'-{}-rotate{}.png'.format('left50', angle), aug3)

'''
img = cv2.imread('bone_data/test/4461.png', cv2.IMREAD_GRAYSCALE)

img=cv2.subtract(img, np.average(img.flatten()))
clahe = cv2.createCLAHE(clipLimit=15)
img = clahe.apply(img)
# img 비율을 맞춰주기 위한 pad 추가
sub = img.shape[0]-img.shape[1]
if sub < 0:
    img = cv2.copyMakeBorder(img, int(-sub/2), int(-sub/2), 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
else:
    img = cv2.copyMakeBorder(img, 0, 0, int(sub/2), int(sub/2), cv2.BORDER_CONSTANT, value=[0,0,0])
img = cv2.resize(img, (500,500))
cv2.imshow('img',img)
cv2.waitKey(0)

''''''
'''
~57: 779장
~114: 3500장
~171: 7129장
~228: 1203장

7 = rotate(-15 -10 -5 0 5 10 15)
5 = rotate(-10 -5 0 5 10)
3 = translate 10% 왼 오 0

779 * 7 * 3 

female: 5778장
male: 6833장
'''