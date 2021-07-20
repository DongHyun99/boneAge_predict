# -*- coding: utf-8 -*-

import pandas as pd
import torch
import glob
import random
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler # learning rate 값을 학습과정에서 조정해주는 모듈
import torch.nn as nn
from age_predictor_model import Bottleneck, AgePredictor

# epoch 수 결정
NUM_EPOCHS = 25

# cuda 작동 확인
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Image and CSV file paths
train_dataset_path = 'bone_data/train/'
test_dataset_path = 'bone_data/test/'
val_dataset_path = 'bone_data/validation/'
train_csv_path = 'bone_data/boneage-training-dataset.csv'
val_test_csv_path = 'bone_data/Validation Dataset.csv'
save_path = 'result/model/'

# sample data를 통한 정규화
k=100 # 100장의 임의의 데이터
size=500 # image scale: 500 x 500
train_image_filenames = glob.glob(train_dataset_path+'*.png')
val_image_filenames = glob.glob(val_dataset_path+'*.png')
test_image_filenames = glob.glob(test_dataset_path+'*.png')

# train dataset에서 임의의 100장을 resize하고 평균과 분산을 구한다 (Normalize시 사용)
random_images = random.sample(population = train_image_filenames,k = k)

means=[]
stds=[]

for filename in random_images:
    image = cv2.imread(filename,0)
    image = cv2.resize(image,(size,size))
    mean, std = cv2.meanStdDev(image)
    means.append(mean[0][0]) # 값이 2차원 list안에 있기 때문에 다음과 같은 형태로 표현한다.
    std.append(std[0][0])

avg_mean = np.mean(means) # 평균의 평균
avg_std = np.mean(std) # 분산의 평균

train_dataset_size = len(train_image_filenames)
val_dataset_size = len(val_image_filenames)
test_dataset_size = len(test_image_filenames)


bones_df = pd.read_csv(train_csv_path)
val_bones_df=pd.read_csv(val_test_csv_path)
bones_df.iloc[:,1:3] = bones_df.iloc[:,1:3].astype(np.float)
val_bones_df = val_bones_df.reindex(columns=['id', 'boneage', 'male'])
# validation set의 csv column의 순서가 train set과 다르기 때문에 통일해 줌
val_bones_df.iloc[:,1:3] = val_bones_df.iloc[:,1:3].astype(np.float)
# columns는 [id, boneage, male]로 이루어져있음, float으로 바꿔면서 male은 (1.0, 0.0)으로 바뀌게 됨

train_df = bones_df
val_df = val_bones_df.iloc[:val_dataset_size,:]
test_df = val_bones_df.iloc[val_dataset_size:,:]

# ===============================================================================================================
#%%
# BonesDataset 객체
class BonesDataset(Dataset):

    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return self.dataframe.shape[0] # len 함수를 객체에 사용했을 경우 dataframe의 행 개수를 반환함

    def __getitem__(self, idx):
        # 객체를 슬라이스 했을 때 작동하는 메소드
        # BonesDataset을 슬라이싱하면 이미지를 float으로 반환한 값, 셩별, 뼈나이를 딕셔너리로 변환한 후
        # transform 형태로 변환하여 반환한다.

        img_name = self.image_dir + str(self.dataframe.iloc[idx,0]) + '.img' # 이미지 이름
        image = cv2.imread(img_name)
        image = image.astype(np.float64)
        gender = np.atleast_1d(self.dataframe.iloc[idx,2]) # 입력값(성별)을 1차원 이상의 배열로 변환
        bone_age = np.atleast_1d(self.dataframe.iloc[idx,1])
        sample = {'image':image, 'gender':gender, 'bone_age':bone_age}

        if self.transform: # transform이 있는 경우 형태를 변환함
            sample = self.transform(sample)
        
        return sample

#%%
# Resize and Convert numpy array to tensor
class ToTensor(object):
    
    def __call__(self, sample): # __init__으로 초기화된 인스턴스를 함수로 취급할 때 불러오게 하는 함수
        image, gender, bone_age = sample['image'], sample['gender'], sample['bone_age']

        image = cv2.resize(image,(size, size)) #500 x 500으로 resize
        image = np.expend_dims(image, axis=0) # 행 차원을 하나 추가한다.

        return {'iamge': torch.from_numpy(image).float(),
                'gender': torch.from_numpy(gender).float(),
                'bone_age': torch.from_numpy(bone_age).float(),}
        # 각 값들을 floatTensor로 바꿔줌

#%%
# Normalize image and bone age
class Nolmalize(object):

    def __init__(self, img_mean, img_std, age_min, age_max):
        self.mean = img_mean
        self.std = img_std
        self.age_min = age_min
        self.age_max = age_max

    def __call__(self, sample): 
        image, gender, bone_age = sample['image'], sample['gender'], sample['bone_age']
        
        # img Standardization code
        image -=self.mean
        image /=self.std

        # boneage Normalization code
        bone_age = (bone_age - self.age_min) / (self.age_max - self.age_min)

        return {'image': image,
                'gender': gender,
                'bone_age':bone_age}

# evaluate 할 때 값을 다시 복구시키는 메소드
def denormalize(inputs, age_min, age_max):
    return inputs * (age_max - age_min) + age_min

#%%
# 학습 데이터 저장 메소드
def save_checkpoint(state, filename='checkpoint.pt'):
    torch.save(state, save_path + filename)

#%%
# evaluation model, predict 값을 반환함, 나중에 test.py로 옮겨야 함
def eval_model(model, optimizer, data_loader, age_min, age_max):

    model.eval()

    with torch.no_grad(): 
        # with은 open, close를 하지 않는 것을 방지해준다.
        # no grad는 자동으로 gradient를 트래킹하지 않게해준다.

        result_array = np.array([])

        for batch_no, batch in enumerate(data_loader): # 몇번째 반복문인지 확인할 수 있는 메소드

            optimizer.zero_grad()
            # 학습이 완료되면 (Iteration이 끝나면) gradients를 0으로 초기화

            image = batch['image'].to(device)
            gender = batch['gender'].to(device)
            # tensor의 device를 GPU로 변경

            outputs = model(image, gender)
            preds = outputs.cpu.numpy() # predict value, cpu로 값을 불러와서 tensor를 numpy 배열로 변환한다.

            preds = preds.reshape(preds.shape[0]) # 행의 개수만큼 값을 reshape
            preds = denormalize(preds, age_min, age_max) # 값을 출력하기 위한 denormalize 작업

            result_array = np.concatenate((result_array, preds)) # result array에 값을 계속 추가해준다.

        return result_array

#%%
# training model, 기본 epochs = 25
def train_model(model, data, criterion, optimizer, scheduler, num_epochs=25):

    for epoch in range(num_epochs):
        scheduler.step()
        model.train()
        running_loss = 0.0 # batch loss 값
        val_running_loss = 0.0 # validation loss 값

        for batch_no, batch in enumerate(data): # batch마다 실행
            # Load batch
            image = batch['image'].to(device)
            gender = batch['gender'].to(device)
            age = batch['bone_age'].to(device)

            optimizer.zero_grad() # 기울기 초기화

            with torch.set_grad_enabled(True): # gradient calculation on
                
                # Forward propagation (순전파)
                outputs = model(image, gender)
                loss = criterion(outputs, age)

                #back propagation (역전파)
                loss.backward() # 변화도 계산
                optimizer.step() # lr step
            
            running_loss += loss.item() # * image.size(0)

            if (batch_no + 1) % 30 == 0: print('Epoch {} Batch {}/12611'.format(epoch+1,(batch_no+1)*4)) # 120장마다 얼마나 남았는지 출력

        total_loss = running_loss / train_dataset_size
        print()
