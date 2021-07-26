# -*- coding: utf-8 -*-

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import glob
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
from age_predictor_model import Bottleneck, AgePredictor
from train import BonesDataset, Normalize, ToTensor

train_dataset_path = 'bone_data/train/'
test_dataset_path = 'bone_data/test/'
val_dataset_path = 'bone_data/validation/'
save_path = 'result/model/'

train_csv_path = 'bone_data/boneage-training-dataset.csv'
val_test_csv_path = 'bone_data/Validation Dataset.csv'

train_image_filenames = glob.glob(train_dataset_path+'*.png')
val_image_filenames = glob.glob(val_dataset_path+'*.png')
test_image_filenames = glob.glob(test_dataset_path+'*.png')

val_dataset_size = len(val_image_filenames)
test_dataset_size = len(test_image_filenames)

bones_df = pd.read_csv(train_csv_path)
val_bones_df=pd.read_csv(val_test_csv_path)
val_bones_df = val_bones_df.reindex(columns=['id', 'boneage', 'male'])
val_bones_df.iloc[:,1:3] = val_bones_df.iloc[:,1:3].astype(np.float)

val_df = val_bones_df.iloc[:val_dataset_size,:]
test_df = val_bones_df.iloc[val_dataset_size:,:]

size=500
k=100

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

age_max = np.max(bones_df['boneage'])
age_min = np.min(bones_df['boneage'])

#%%
def denormalize(inputs, age_min, age_max):
    return inputs * (age_max - age_min) + age_min

#%%
def eval_model(model, test_data, criterion, optimizer):

    test_loss = 0.0
    result_array = np.array([])

    model.eval()
    with torch.no_grad():

        for batch_no, batch in enumerate(test_data):

            # Load batch
            image = batch['image'].to(device)
            gender = batch['gender'].to(device)
            age = batch['bone_age'].to(device)
            optimizer.zero_grad()

            outputs = model(image, gender)
            loss = criterion(outputs, age)

            preds = outputs.cpu().numpy()
            preds = preds.reshape(preds.shape[0])
            preds = denormalize(preds, age_min, age_max)
            result_array = np.concatenate((result_array,preds))

            test_loss += loss.item() * image.size(0)
            if (batch_no + 1) % 25 == 0: print('Batch {}/624'.format((batch_no+1)*4)) # 100장마다 얼마나 남았는지 출력

        print('test loss: ', test_loss/test_dataset_size)
        return result_array
#%%

if __name__ == '__main__':
    freeze_support()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    data_transform = transforms.Compose([
        Normalize(avg_mean,avg_std,age_min,age_max),
        ToTensor()])
    test_dataset = BonesDataset(dataframe = test_df,image_dir=test_dataset_path,transform = data_transform)
    test_data_loader = DataLoader(test_dataset,batch_size=4,shuffle=False,num_workers = 4)
    
    age_predictor = AgePredictor(block = Bottleneck,layers = [3, 4, 23, 3],num_classes =1)
    model = age_predictor.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(age_predictor.parameters(), lr=0.001, momentum=0.9)

    checkpoint = torch.load(save_path+'epoch-7-loss-0.0256-val_loss-0.0336.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    result_array = eval_model(model,test_data_loader ,criterion, optimizer)
    
    predict_df = test_df.copy()

    predict_df['output'] = result_array
    predict_df['output'] = np.round(predict_df['output'], decimals=2) # 2에서 반올림
    predict_df = predict_df.reset_index() # 인덱스 초기화 (재배열)

    predict_df.to_csv('result/predict.csv', sep=',',na_rep='NaN')