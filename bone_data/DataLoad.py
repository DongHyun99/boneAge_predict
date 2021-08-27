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

age_max = np.max(train_data.boneage) # 228 month
age_min = np.min(train_data.boneage) # 1 month

# Transform Setting
train_composed = transforms.Compose([transforms.Resize((500,500)),transforms.ToTensor()])

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
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8,8))
        img= clahe.apply(img)

        # convert cv to PIL
        img = Image.fromarray(img.astype(np.float64))
        
        # 입력값을 1차원 이상의 배열로 변환
        gender = np.atleast_1d(self.dataframe.iloc[idx,2])
        bone_age = np.atleast_1d(self.dataframe.iloc[idx,1])
        sample = {'image':img, 'gender':gender, 'bone_age':bone_age}

        if self.transform:
            sample['bone_age'] = torch.from_numpy(sample['bone_age']).float()
            sample['gender'] = torch.from_numpy(sample['gender']).float()
            sample['image'] = self.transform(sample['image'])

        return sample

#%%
trainset = BoneDataSet(train_img_path, train_data, train_composed)
validationset = BoneDataSet(validation_img_path, val_data, eval_composed)
testset = BoneDataSet(test_img_path, test_data, eval_composed)

train_data_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
val_data_loader = DataLoader(validationset, batch_size=batch_size, shuffle=False, num_workers=4)
test_data_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)


# example image plot
from multiprocessing.spawn import freeze_support
import matplotlib.pyplot as plt

'''
if __name__ == '__main__':
    freeze_support()
    sample_batch = next(iter(val_data_loader))
    img = sample_batch['image'][0]
    img = img.permute(1,2,0)
    plt.imshow(img, cmap='gray')
    plt.show()
'''