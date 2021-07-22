# -*- coding: utf-8 -*-

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler # learning rate 값을 학습과정에서 조정해주는 모듈
import torch.nn as nn
from torchvision import transforms
import glob
import random
import cv2
import numpy as np
from age_predictor_model import Bottleneck, AgePredictor
import matplotlib.pyplot as plt
from multiprocessing import freeze_support

# cuda 작동 확인
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# epoch 수 결정
NUM_EPOCHS = 25

# loss list (나중에 plot 하기 위해서)
loss_list = []
val_loss_list = []

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
    stds.append(std[0][0])

avg_mean = np.mean(means) # 평균의 평균
avg_std = np.mean(std) # 분산의 평균

# Split Train Validation Test
# Train - 12611 images
# Val   -  800 images
# Test  -  625 images

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

age_max = np.max(bones_df['boneage'])
age_min = np.min(bones_df['boneage'])

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
        image = np.expand_dims(image, axis=0) # 행 차원을 하나 추가한다.

        return {'image': torch.from_numpy(image).float(),
                'gender': torch.from_numpy(gender).float(),
                'bone_age': torch.from_numpy(bone_age).float()}
        # 각 값들을 floatTensor로 바꿔줌

#%%
# Normalize image and bone age
class Normalize(object):

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
def train_model(model, train_data, val_data, criterion, optimizer, scheduler, num_epochs=25):

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0 # batch loss 값
        val_running_loss = 0.0 # validation loss 값

        for batch_no, batch in enumerate(train_data): # batch마다 train set에 대한 학습 진행
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
                optimizer.step() # optim step
            
            running_loss += loss.item() * image.size(0)

            if (batch_no + 1) % 25 == 0: print('Epoch {} Batch {}/12611'.format(epoch+1,(batch_no+1)*4)) # 100장마다 얼마나 남았는지 출력

        total_loss = running_loss / train_dataset_size # epoch 평균 loss

        print('=================validation evaluate=================')

        model.eval()
        for batch_no, batch in enumerate(val_data): # validation loss 구하기
            # Load batch
            image = batch['image'].to(device)
            gender = batch['gender'].to(device)
            age = batch['bone_age'].to(device)

            optimizer.zero_grad()
            # only forward pass, dont update gradients
            with torch.no_grad():

                outputs = model(image, gender)
                loss = criterion(outputs, age)

            val_running_loss += loss.item() * image.size(0)
            if (batch_no + 1) % 25 == 0: print('Epoch {} Batch {}/800'.format(epoch+1,(batch_no+1)*4)) # 100장마다 얼마나 남았는지 출력

        val_loss = val_running_loss / val_dataset_size # epoch 평균 validation loss

        print('loss: {}, val_loss: {}'.format(total_loss, val_loss))

        states = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss,
                    'val_loss': val_loss
                }
        # 저장하는 states는 epoch, model state, optimizer state, loss, val_loss이다. 
        save_checkpoint(states, filename='epoch-{}-loss-{:.4f}-val_loss-{:.4f}.tar'.format(epoch+1, total_loss, val_loss))

        # loss list 저장
        loss_list.append(total_loss)
        val_loss_list.append(total_loss)

        scheduler.step() # lr step

    return model

#%%
# loss visualization
def display_loss():
    plt.figure()
    plt.plot([x for x in range(NUM_EPOCHS)], loss_list, label='loss')
    plt.plot([x for x in range(NUM_EPOCHS)], val_loss_list, label='valication_loss')
    
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.show()
    plt.savefig('result/loss.png', dpi=200)

#display predict picture
'''
def display_preds(num):
    idx = random.sample(range(0, test_df.shape[0]), num)
    for i in idx:
        image = cv2.imread(test_dataset_path + str(test_df['id'][i]) + '.png')
        image = cv2.resize(image,(500,500))
        cv2.putText(image,'Actual:' + str(test_df['boneage'][i]),(20,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255))
        cv2.putText(image,'Predicted:' + str(test_df['output'][i]),(20,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255))
        cv2.imshow('Bone Age Prediction',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
display_preds(4)
'''

#%%

if __name__ == '__main__':
    # window에서도 구동 되게하는 코드
    freeze_support()

    print('Approx. Mean of Images in Dataset: ',avg_mean)
    print('Approx. Standard Deviation of Images in Dataset: ',avg_std)

    data_transform = transforms.Compose([
        Normalize(avg_mean,avg_std,age_min,age_max),
        ToTensor()])

    train_dataset = BonesDataset(dataframe = train_df,image_dir=train_dataset_path,transform = data_transform)
    val_dataset = BonesDataset(dataframe = val_df,image_dir = val_dataset_path,transform = data_transform)
    test_dataset = BonesDataset(dataframe = test_df,image_dir=test_dataset_path,transform = data_transform)

    # Sanity Check
    print(train_dataset[0]['image'].shape) # shape을 보면 [1, 500, 500]임을 알 수 있다.

    train_data_loader = DataLoader(train_dataset,batch_size=4,shuffle=False,num_workers = 4)
    val_data_loader = DataLoader(val_dataset,batch_size=4,shuffle=False,num_workers = 4)
    test_data_loader = DataLoader(test_dataset,batch_size=4,shuffle=False,num_workers = 4)

    # Sanity Check 2
    sample_batch =  next(iter(test_data_loader))
    print(sample_batch)

    # Initialize the model
    age_predictor = AgePredictor(block = Bottleneck,layers = [3, 4, 23, 3],num_classes =1)

    # Set loss as mean squared error (for continuous output)
    # # Initialize Stochastic Gradient Descent optimizer and learning rate scheduler
    
    age_predictor = age_predictor.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(age_predictor.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.5)

    # train model
    resnet_model = train_model(age_predictor, train_data_loader, val_data_loader, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS)

    # show loss graph
    display_loss()