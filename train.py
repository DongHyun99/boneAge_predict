# -*- coding: utf-8 -*-

from torch.utils.data.dataset import Dataset
from earlyStopping import EarlyStopping
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import cv2

# Hyperparameters Setting 
epochs = 100
batch_size = 4

train_img_path = 'bone_data/train/'
train_csv_path = 'bone_data/training_dataset/csv'
validation_img_path = 'bone_data/validation/'
validation_csv_path = 'bone_data/validation_dataset.csv'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
earlystop = EarlyStopping() # 기본 10 epoch

# dataset setting
train_data = pd.read_csv(train_csv_path)
train_data.iloc[:, 1:3] = train_data.iloc[:, 1:3].astype(np.float)

val_data = pd.read_csv(validation_csv_path)
val_data = val_data.reindex(columns=['id', 'boneage', 'male'])
val_data.iloc[:, 1:3] = val_data.iloc[:, 1:3].astype(np.float)

age_max = np.max(train_data.boneage) # 228 month
age_min = np.min(train_data.boneage) # 1 month

# Transform Setting
composed = transforms.Compose([transforms.CenterCrop(10), transforms.ToTensor()])

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
        img = cv2.read(img_name, 0)
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8,8))
        img= clahe.apply(img)
