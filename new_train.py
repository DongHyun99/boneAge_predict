# -*- coding: utf-8 -*-

import torch
from earlyStopping import EarlyStopping
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2
from multiprocessing import freeze_support
from torchvision import transforms
from model.BoneAgeNet import BoneAgeNet
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import datetime
import random

# For reproducibility use the seeds below (임의 값 고정)
torch.manual_seed(1498920)
torch.cuda.manual_seed(1498920)
np.random.seed(1498920)
random.seed(1498920)
torch.backends.cudnn.deterministic=True

# cuda 작동 확인
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# epoch 수 결정
NUM_EPOCHS = 100

save_path = 'D:/model/'
es = EarlyStopping()

# loss list (나중에 plot 하기 위해서)
loss_list = []
val_loss_list = []
batch_loss_list = []
batch_val_loss_list = []

train_dataset_path = 'bone_data/train/'
validation_image_path = 'bone_data/validation/'

# Image and CSV file paths
train_csv_path = 'bone_data/training_dataset.csv'
val_csv_path = 'bone_data/training_dataset.csv'
test_csv_path = 'bone_data/training_dataset.csv'

size=500 # image scale: 500 x 500

# img_mean = 46.48850549076203
# img_std = 42.557445370314426

# Split Train Validation Test
# Train - 12611 images
# Val   -  1425 images
# Test  -   200 images

# read dataset
train_df = pd.read_csv(train_csv_path)
train_df.iloc[:,1:3] = train_df.iloc[:,1:3].astype(np.float)

val_df = pd.read_csv(val_csv_path)
val_df = val_df.reindex(columns=['id', 'boneage', 'male'])
val_df.iloc[:,1:3] = val_df.iloc[:,1:3].astype(np.float)

test_df = pd.read_csv(test_csv_path)
test_df.iloc[:,1:3] = test_df.iloc[:,1:3].astype(np.float)

age_max = np.max(train_df['boneage']) # 228 (19 year)
age_min = np.min(train_df['boneage']) # 1

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

        img_name = self.image_dir + str(self.dataframe.iloc[idx,0]) + '.png' # 이미지 이름
        image = cv2.imread(img_name,0)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        image = clahe.apply(image)
        image = cv2.resize(image, (500,500))

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
        
        #image = cv2.resize(image,(size, size)) #500 x 500으로 resize
        image = np.expand_dims(image, axis=0) # 행 차원을 하나 추가한다.

        return {'image': torch.from_numpy(image).float(),
                'gender': torch.from_numpy(gender).float(),
                'bone_age': torch.from_numpy(bone_age).float()}
        # 각 값들을 floatTensor로 바꿔줌

#%%
class Normalize(object):
    
    def __init__(self,age_min, age_max):
        
        self.age_min = age_min
        self.age_max = age_max
        
    
    
    def __call__(self,sample):
        image, gender, bone_age = sample['image'], sample['gender'], sample['bone_age']        
        bone_age = (bone_age - self.age_min)/ (self.age_max - self.age_min)

        return {'image': image,
                'gender': gender,
                'bone_age':bone_age}

def denormalize(inputs, age_min, age_max):
    return inputs * (age_max - age_min) + age_min

#%%
# 학습 데이터 저장 메소드
def save_checkpoint(state, filename='checkpoint.pt'):
    torch.save(state, save_path + filename)

# ===============================================================================================================
#%%
# training model, 기본 epochs = 25
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_model = None
    best_val_loss = None
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        running_loss = 0.0
        val_running_loss = 0.0
        loss = 0

        for batch_no, batch in enumerate(train_data_loader): # batch마다 train set에 대한 학습 진행
            optimizer.zero_grad() # 기울기 초기화
            
            # Load batch
            image = batch['image'].to(device)
            gender = batch['gender'].to(device)
            age = batch['bone_age'].to(device)

            with torch.set_grad_enabled(True): # gradient calculation on
                
                # Forward propagation (순전파)
                outputs = model(image, gender)
                loss = criterion(outputs, age)

                #back propagation (역전파)
                loss.backward() # 변화도 계산
                optimizer.step() # optim step

            running_loss += loss.item()
            if (batch_no + 1) % 25 == 0: print('Epoch {}: {}/12611, batch loss: {}'.format(epoch+1,4*(batch_no+1), loss.item())) # 100장마다 얼마나 남았는지 출력

            batch_loss_list.append(loss.item())

        total_loss = running_loss / 3152.75 # epoch 평균 loss

        print('=================validation evaluate=================')

        model.eval()
        for batch_no, batch in enumerate(val_data_loader): # validation loss 구하기
            # Load batch
            image = batch['image'].to(device)
            gender = batch['gender'].to(device)
            age = batch['bone_age'].to(device)

            optimizer.zero_grad()
            # only forward pass, dont update gradients
            with torch.no_grad():

                outputs = model(image, gender)
                loss = criterion(outputs, age)

            val_running_loss += loss.item()
            if (batch_no + 1) % 25 == 0: print('Epoch {}: {}/1425, batch loss: {}'.format(epoch+1,4*(batch_no+1), loss.item())) # 100장마다 얼마나 남았는지 출력

            batch_val_loss_list.append(loss.item())

        val_loss = val_running_loss / 356.25 # epoch 평균 validation loss
        scheduler.step() # lr step
        print('\ntime: {}\nloss: {}, val_loss: {}\n==============================================='.format(datetime.datetime.now(),total_loss, val_loss))

        states = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss,
                    'val_loss': val_loss
                }
        # 저장하는 states는 epoch, model state, optimizer state, loss, val_loss이다.
        if (epoch+1) % 1 == 0: # 1 epoch마다 저장
            save_checkpoint(states, filename='epoch-{}-loss-{:.4f}-val_loss-{:.4f}.tar'.format(epoch+1, total_loss, val_loss))

        # loss list 저장
        loss_list.append(total_loss)
        val_loss_list.append(val_loss)

        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = states

        if es.step(val_loss):
            break

    save_checkpoint(best_model, filename='BEST_MODEL-epoch-{}-val_loss-{:.4f}.tar'.format(states['epoch']+1,val_loss))
    return model


#%%
# loss visualization
def display_loss(num):
    plt.figure()
    plt.plot([x for x in range(num)], loss_list, label='loss')
    
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.show()
    plt.savefig('result/loss.png', dpi=200)

    plt.plot([x for x in range(num)], val_loss_list, label='validation_loss')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.show()
    plt.savefig('result/val_loss.png', dpi=200)

    l=pd.Series([loss_list])
    vl=pd.Series([val_loss_list])
    bl=pd.Series([batch_loss_list])
    bvl=pd.Series([batch_val_loss_list])

    l.to_csv('result/loss.csv', sep=',',na_rep='NaN')
    vl.to_csv('result/valloss.csv', sep=',',na_rep='NaN')
    bl.to_csv('result/batchloss.csv', sep=',',na_rep='NaN')
    bvl.to_csv('result/batchvalloss.csv', sep=',',na_rep='NaN')

#%%

if __name__ == '__main__':
    # window에서도 구동 되게하는 코드
    freeze_support()

    data_transform = transforms.Compose([Normalize(age_min,age_max),ToTensor()])
    train_dataset = BonesDataset(dataframe = train_df,image_dir=train_dataset_path,transform = data_transform)
    val_dataset = BonesDataset(dataframe = val_df,image_dir = train_dataset_path,transform = data_transform)

    # Sanity Check
    # print(train_dataset[0]['image'].shape) # shape을 보면 [1, 500, 500]임을 알 수 있다.

    train_data_loader = DataLoader(train_dataset,batch_size=4,shuffle=True,num_workers = 4)
    val_data_loader = DataLoader(val_dataset,batch_size=4,shuffle=True,num_workers = 4)

    # Sanity Check 2
    # sample_batch =  next(iter(test_data_loader))
    # print(sample_batch)
    # Initialize the model

    age_predictor = BoneAgeNet(num_classes=1)

    # Set loss as mean squared error (for continuous output)
    # # Initialize Stochastic Gradient Descent optimizer and learning rate scheduler
    
    # age_predictor = nn.DataParallel(age_predictor)
    age_predictor = age_predictor.to(device)
    

    criterion = nn.MSELoss()
    optimizer = optim.SGD(age_predictor.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.5)


    # train model
    print('================= train start =================\n', datetime.datetime.now())
    efficientnet_model = train_model(age_predictor, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS)
    

    # show loss graph

    display_loss(len(loss_list))



    #sample_batch = next(iter(val_data_loader))
    #image = sample_batch['image'][2]
    #img = image.permute(1,2,0)
    #plt.imshow(img, cmap='gray')
    #plt.show()