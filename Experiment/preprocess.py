import cv2
import numpy as np
import pandas as pd

#data load
train_img_path = 'bone_data/train/'
train_csv_path = 'bone_data/training_dataset.csv'
test_img_path = 'bone_data/test/'
test_csv_path = 'bone_data/test_dataset.csv'

# dataset setting
train_data = pd.read_csv(train_csv_path)
train_data.iloc[:, 1:3] = train_data.iloc[:, 1:3].astype(np.float)
test_data = pd.read_csv(test_csv_path)
test_data.iloc[:, 1:3] = test_data.iloc[:, 1:3].astype(np.float)

t1 = train_data[train_data['boneage']<57]
t2 = train_data[(train_data['boneage']>=57) & (train_data['boneage']<=114)]
t3 = train_data[(train_data['boneage']>114) & (train_data['boneage']<=171)]
t4 = train_data[(train_data['boneage']>171) & (train_data['boneage']<=228)]

for idx, data in test_data.iterrows():
    id = data['id']
    img = cv2.imread(test_img_path+str(int(id))+'.png', cv2.IMREAD_GRAYSCALE)

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
    cv2.imwrite('Experiment/test/'+str(int(id))+'.png', img)


'''
~57: 779장
~114: 3500장
~171: 7129장
~228: 1203장

7 = rotate(-15 -10 -5 0 5 10 15)
5 = rotate(-10 -5 0 5 10)
3 = translate 10% 왼 오 0

600*10 =6000
3000*2 = 6000
6000
1200*5 = 6000

female: 5778장
male: 6833장
'''