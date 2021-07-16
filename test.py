import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import transforms
import os
import glob
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
from age_predictor_model import Bottleneck, AgePredictor
from sklearn.metrics import mean_squared_error
from multiprocessing import freeze_support
from main import test_dataset

csv_path = 'bone_data/boneage-training-dataset.csv'
bones_df = pd.read_csv(csv_path)
val_dataset_path = 'bone_data/validation/'
val_csv_path = 'bone_data/Validation Dataset.csv'
val_image_filenames = glob.glob(val_dataset_path+'*.png')
val_size = len(val_image_filenames)
val_bones_df=pd.read_csv(val_csv_path)
val_bones_df = val_bones_df.reindex(columns=['id', 'boneage', 'male'])
val_bones_df.iloc[:,1:3] = val_bones_df.iloc[:,1:3].astype(np.float)
test_df = val_bones_df.iloc[val_size:,:]


age_max = np.max(bones_df['boneage'])
age_min = np.min(bones_df['boneage'])

def denormalize(inputs,age_min,age_max):
    return inputs * (age_max - age_min) + age_min

def eval_model(model,data_loader):
    model.eval()

    with torch.no_grad():
        
        result_array = np.array([])
        
        for batch_no,batch in enumerate(data_loader):
            
            optimizer.zero_grad()
            
            image = batch['image'].to(device)
            gender = batch['gender'].to(device)
            
    
            outputs = model(image,gender)
            preds = outputs.cpu().numpy()
    
            preds = preds.reshape(preds.shape[0])
            preds = denormalize(preds,age_min,age_max)
            
            result_array = np.concatenate((result_array,preds))
            
        return result_array
        

if __name__ =='__main__':
    freeze_support()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load('epoch-2-loss-0.0314-val_loss-0.0340.pth.tar')
    start_epoch = checkpoint['epoch']
    age_predictor = AgePredictor(block = Bottleneck,layers = [3, 4, 23, 3],num_classes =1)
    age_predictor = age_predictor.to(device)
    optimizer = optim.SGD(age_predictor.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.5)
    test_data_loader = DataLoader(test_dataset,batch_size=4,shuffle=False,num_workers = 4)

    age_predictor.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    result_array = eval_model(age_predictor,test_data_loader)
    
    test_df['output'] = result_array
    test_df['output'] = np.round(test_df['output'], decimals=2)
    test_df = test_df.reset_index()

    rmse = np.sqrt(mean_squared_error(test_df['boneage'], test_df['output']))
#%%
    print('rmse: ',rmse)
    print(test_df)
    test_df.to_csv('predict.csv', sep=',',na_rep='NaN')
# 25.259