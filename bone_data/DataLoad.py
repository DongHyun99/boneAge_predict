# -*- coding: utf-8 -*-

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

train_img_path = 'bone_data/train/'
train_csv_path = 'bone_data/training_dataset.csv'
validation_img_path = 'bone_data/validation/'
validation_csv_path = 'bone_data/validation_dataset.csv'
test_img_path = 'bone_data/test/'
test_csv_path = 'bone_data/test_dataset.csv'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# dataset setting
train_data = pd.read_csv(train_csv_path)
train_data.iloc[:, 1:3] = train_data.iloc[:, 1:3].astype(np.float)

val_data = pd.read_csv(validation_csv_path)
val_data = val_data.reindex(columns=['id', 'boneage', 'male'])
val_data.iloc[:, 1:3] = val_data.iloc[:, 1:3].astype(np.float)

test_data = pd.read_csv(test_csv_path)
test_data.iloc[:, 1:3] = test_data.iloc[:, 1:3].astype(np.float)

# Augmentation List
# transforms.RandomAffine(0, translate=(0.1, 0.1)), # tlanslation <= 20
aug_list=[transforms.RandomRotation(20)] # rotate <=20% 
aug_list2 = [transforms.Compose([transforms.Pad(50), transforms.Resize((img_size, img_size))])]

# Transform Setting
train_composed = transforms.Compose([transforms.RandomApply(aug_list),transforms.Resize((img_size,img_size)),transforms.RandomApply(aug_list2),transforms.ToTensor()])
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

        # contrast limited adaptive historgram equalization
        img=cv2.subtract(img, np.average(img.flatten()))
        clahe = cv2.createCLAHE(clipLimit=15)
        img = clahe.apply(img)

        # img 비율을 맞춰주기 위한 pad 추가
        sub = img.shape[0]-img.shape[1]
        if sub < 0:
            img = cv2.copyMakeBorder(img, int(-sub/2), int(-sub/2), 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
        else:
            img = cv2.copyMakeBorder(img, 0, 0, int(sub/2), int(sub/2), cv2.BORDER_CONSTANT, value=[0,0,0])            

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

#%%
trainset = BoneDataSet(train_img_path, train_data, train_composed)
validationset = BoneDataSet(validation_img_path, val_data, validation_composed)
testset = BoneDataSet(test_img_path, test_data, validation_composed)

train_data_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
val_data_loader = DataLoader(validationset, batch_size=batch_size, shuffle=False, num_workers=4)
test_data_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

'''
# example image plot
from multiprocessing.spawn import freeze_support
import matplotlib.pyplot as plt


if __name__ == '__main__':
    freeze_support()
    sample_batch = next(iter(train_data_loader))
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