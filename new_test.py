# -*- coding: utf-8 -*-

import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import numpy as np
from multiprocessing import freeze_support
from model.BoneAgeNet import BoneAgeNet
from new_train import BonesDataset, Normalize, ToTensor
import random

# For reproducibility use the seeds below (임의 값 고정)
torch.manual_seed(1498920)
torch.cuda.manual_seed(1498920)
np.random.seed(1498920)
random.seed(1498920)
torch.backends.cudnn.deterministic=True

dataset_path = 'bone_data/train/'
save_path = 'D:/model/'

# Image and CSV file paths
train_csv_path = 'bone_data/boneage-training-dataset.csv'

bones_df = pd.read_csv(train_csv_path)
bones_df.iloc[:,1:3] = bones_df.iloc[:,1:3].astype(np.float)
# columns는 [id, boneage, male]로 이루어져있음, float으로 바꿔면서 male은 (1.0, 0.0)으로 바뀌게 됨

train_df = bones_df.sample(n=10000, random_state = 1004)
bones_df = bones_df.drop(train_df.index)
train_df.index = [i for i in range(0,10000,1)]

val_df = bones_df.sample(n=1000, random_state = 1004)
bones_df = bones_df.drop(val_df.index)
val_df.index = [i for i in range(0,1000,1)]

test_df = bones_df[:]
test_df.index = [i for i in range(0,1611,1)]

age_max = 228
age_min = 1

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

            test_loss += loss.item()
            if (batch_no + 1) % 25 == 0: print('Batch {}/1611, batch loss: {}'.format((batch_no+1)*4, loss.item())) # 100장마다 얼마나 남았는지 출력

        print('test loss: ', test_loss/1611*4)
        return result_array
#%%

if __name__ == '__main__':
    freeze_support()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    data_transform = transforms.Compose([Normalize(age_min,age_max), ToTensor()])

    test_dataset = BonesDataset(dataframe = test_df,image_dir=dataset_path,transform = data_transform)
    test_data_loader = DataLoader(test_dataset,batch_size=4,shuffle=False,num_workers = 4)

    age_predictor = BoneAgeNet(num_classes=1)
    # age_predictor = nn.DataParallel(age_predictor) # -> to multi-GPU
    model = age_predictor.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(age_predictor.parameters(), lr=0.001, momentum=0.9)

    checkpoint = torch.load(save_path+'epoch-15-loss-0.0131-val_loss-0.0116.tar')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    result_array = eval_model(model,test_data_loader ,criterion, optimizer)
    
    predict_df = test_df.copy()

    predict_df['output'] = result_array
    predict_df['output'] = np.round(predict_df['output'], decimals=2) # 2에서 반올림
    predict_df = predict_df.reset_index() # 인덱스 초기화 (재배열)

    predict_df.to_csv('predict.csv', sep=',',na_rep='NaN')