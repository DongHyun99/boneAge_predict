# -*- coding: utf-8 -*-

import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import glob
import numpy as np
from multiprocessing import freeze_support
from model.BoneAgeNet import BoneAgeNet
from new_train import BonesDataset, Normalize, ToTensor

test_dataset_path = 'bone_data/test/'
save_path = 'D:/model/'
test_csv_path = 'bone_data/test.csv'

test_image_filenames = glob.glob(test_dataset_path+'*.png')
test_dataset_size = len(test_image_filenames)

test_bones_df=pd.read_csv(test_csv_path)
test_bones_df.iloc[:,1:3] = test_bones_df.iloc[:,1:3].astype(np.float)

test_df = test_bones_df[:]

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

            test_loss += loss.item() * image.size(0)
            if (batch_no + 1) % 25 == 0: print('Batch {}/200'.format((batch_no+1)*4)) # 100장마다 얼마나 남았는지 출력

        print('test loss: ', test_loss/test_dataset_size)
        return result_array
#%%

if __name__ == '__main__':
    freeze_support()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    data_transform = transforms.Compose([Normalize(age_min,age_max), ToTensor()])

    test_dataset = BonesDataset(dataframe = test_df,image_dir=test_dataset_path,transform = data_transform)
    test_data_loader = DataLoader(test_dataset,batch_size=4,shuffle=False,num_workers = 4)

    age_predictor = BoneAgeNet(num_classes=1)
    # age_predictor = nn.DataParallel(age_predictor) # -> to multi-GPU
    model = age_predictor.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(age_predictor.parameters(), lr=0.001, momentum=0.9)

    checkpoint = torch.load(save_path+'epoch-7-loss-0.0289-val_loss-0.0241.tar')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    result_array = eval_model(model,test_data_loader ,criterion, optimizer)
    
    predict_df = test_df.copy()

    predict_df['output'] = result_array
    predict_df['output'] = np.round(predict_df['output'], decimals=2) # 2에서 반올림
    predict_df = predict_df.reset_index() # 인덱스 초기화 (재배열)

    predict_df.to_csv('result/predict.csv', sep=',',na_rep='NaN')