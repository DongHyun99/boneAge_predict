# -*- coding: utf-8 -*-

from pandas.core.frame import DataFrame
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import cv2
from PIL import Image

# Hyperparameters Setting
batch_size = 4
img_size = 500

train_img_path = 'D:/train/'
train_csv_path = 'D:/new_train.csv'
validation_img_path = 'bone_data/validation/'
validation_csv_path = 'bone_data/validation_dataset.csv'
test_img_path = 'bone_data/test/'
test_csv_path = 'bone_data/test_dataset.csv'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# dataset setting
train_data = pd.read_csv(train_csv_path)
train_data.iloc[:, 2:3] = train_data.iloc[:, 2:3].astype(np.float)

val_data = pd.read_csv(validation_csv_path)
val_data = val_data.reindex(columns=['id', 'boneage', 'male'])
val_data.iloc[:, 1:3] = val_data.iloc[:, 1:3].astype(np.float)

test_data = pd.read_csv(test_csv_path)
test_data.iloc[:, 1:3] = test_data.iloc[:, 1:3].astype(np.float)
 

# Transform Setting
train_composed = transforms.Compose([transforms.ToTensor()])
validation_composed = transforms.Compose([transforms.Resize((img_size,img_size)),transforms.ToTensor()])

#%%
# BoneData Class
class BoneDataSet(Dataset):

    def __init__(self, img, dataframe, transform=None):
        self.img = img
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        img_name = self.img + str(self.dataframe.id[idx]) + '.png'
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

        # convert cv to PIL
        img = Image.fromarray(img.astype(np.float64))
        
        # 입력값을 1차원 이상의 배열로 변환
        gender = np.atleast_1d(self.dataframe.iloc[idx,2])
        bone_age = np.atleast_1d(self.dataframe.iloc[idx,1])
        sample = {'image':img, 'gender':gender, 'bone_age':bone_age}

        if self.transform:
            # age & gender는 미리 tensor로 변환, image는 transform을 적용
            sample['bone_age'] = torch.from_numpy(sample['bone_age']).float()
            sample['gender'] = torch.from_numpy(sample['gender']).float()
            sample['image'] = self.transform(sample['image'])

        return sample

'''
from multiprocessing.spawn import freeze_support
import matplotlib.pyplot as plt

if __name__ == '__main__':
    freeze_support()
    # example image plot
    train_dataset = pd.read_csv(train_csv_path)
    train_dataset.iloc[:, 2:3] = train_dataset.iloc[:, 2:3].astype(np.float)
    t1 = train_dataset[train_dataset['boneage']<57].sample(2000)
    t2 = train_dataset[(train_dataset['boneage']>=57) & (train_dataset['boneage']<=114)].sample(2000)
    t3 = train_dataset[(train_dataset['boneage']>114) & (train_dataset['boneage']<=171)].sample(2000)
    t4 = train_dataset[(train_dataset['boneage']>171) & (train_dataset['boneage']<=228)].sample(2000)
    dataframe = pd.concat([t1, t2, t3, t4])
    dataframe.index = [i for i in range(8000)]
    trainset = BoneDataSet(train_img_path, dataframe, train_composed)
    train_data = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    sample_batch = next(iter(train_data))
    img1 = sample_batch['image'][0]
    img2 = sample_batch['image'][1]
    img3 = sample_batch['image'][2]
    img4 = sample_batch['image'][3]
    
    img1 = img1.permute(1,2,0)
    img2 = img2.permute(1,2,0)
    img3 = img3.permute(1,2,0)
    img4 = img4.permute(1,2,0)

    plt.subplot(221)
    plt.imshow(img1, cmap='gray')

    plt.subplot(222)
    plt.imshow(img2, cmap='gray')

    plt.subplot(223)
    plt.imshow(img3, cmap='gray')

    plt.subplot(224)
    plt.imshow(img4, cmap='gray')

    plt.show()
'''