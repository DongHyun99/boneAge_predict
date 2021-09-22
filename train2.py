# -*- coding: utf-8 -*-
# tensorboard --logdir=runs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from utils import EarlyStopping
from bone_data.DataLoad2 import BoneDataSet
from bone_data.DataLoad import val_data_loader
from model.BoneageModel import BoneAgeNet

from multiprocessing.spawn import freeze_support
import datetime
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

# For reproducibility use the seeds below (임의 값 고정)
torch.manual_seed(1498920)
torch.cuda.manual_seed(1498920)
torch.backends.cudnn.deterministic=True

# Hyperparameters Setting 
epochs = 600
batch_size = 4
es = EarlyStopping(patience=30)
save_path = 'D:/model/'
train_img_path = 'D:/train/'
train_csv_path = 'D:/new_train.csv'

# batch loss counter
batch_loss = 0
val_batch_loss = 0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_composed = transforms.Compose([transforms.ToTensor()])

# data load
writer = SummaryWriter()
val_data = val_data_loader
model = BoneAgeNet()
train_dataset = pd.read_csv(train_csv_path)
train_dataset.iloc[:, 2:3] = train_dataset.iloc[:, 2:3].astype(np.float)

if __name__ == '__main__':
    freeze_support()

    # declare model
    model.to(device)
    
    # loss, optimizer, scheduler
    criterion = nn.L1Loss() # L1Loss / MSELoss
    optimizer = optim.Adam(model.parameters())
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=10, verbose=1, eps=0.0001, cooldown=5, min_lr=0.0001)

    # load pre_trained model
    #checkpoint = torch.load(save_path+'epoch-90-loss-6.1438-val_loss-7.3173.pt')
    #model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#%%

def save_checkpoint(state, filename='checkpoint.pt'):
    torch.save(state, save_path + filename)

def train(model, train_data, epoch):
    model.train()
    batch_loss = epoch * (8000//batch_size) + 1
    epoch_loss = 0.0

    for batch_no, batch in enumerate(train_data):
        
        # Load batch
        img = batch['image'].to(device)
        gender = batch['gender'].to(device)
        age = batch['bone_age'].to(device)

        # gradient initialize
        optimizer.zero_grad()

        # Forward propagation (순전파)
        output = model(img, gender)
        loss = criterion(output, age)

        # Backward propagation (역전파)
        loss.backward() # 변화도 계산
        optimizer.step() # optim step

        epoch_loss += loss.item()
        writer.add_scalar('train/batchLoss', loss.item(), batch_loss+batch_no)

        if (batch_no + 1) % 25 == 0: print('\rEpoch {}: {}/8000, batch loss: {}'.format(epoch+1,batch_size*(batch_no+1), loss.item()), end='') # 100장마다 출력
    return epoch_loss / (8000//batch_size)

def eval(model, val_data, epoch):
    model.eval()
    batch_loss = epoch * 357 + 1
    epoch_val_loss = 0.0

    with torch.no_grad():
        for batch_no, batch in enumerate(val_data):

            # Load batch
            img = batch['image'].to(device)
            gender = batch['gender'].to(device)
            age = batch['bone_age'].to(device)

            # gradient initialize
            optimizer.zero_grad()
            
            # Forward propagation (순전파)
            output = model(img, gender)
            loss = criterion(output, age)

            epoch_val_loss += loss.item()
            writer.add_scalar('validation/batchLoss', loss.item(), batch_loss+batch_no)

            if (batch_no + 1) % 25 == 0: print('\rEpoch {}: {}/1425, batch loss: {}'.format(epoch+1,batch_size*(batch_no+1), loss.item()), end='') # 100장마다 출력
    return epoch_val_loss / (1425//batch_size)

def main():
    best_loss = math.inf
    best_model = None
    print('{}\n==============================train start==============================\n'.format(datetime.datetime.now()))
    line = '======================================================================='
    for epoch in range(epochs):
        t1 = train_dataset[train_dataset['boneage']<57].sample(2000)
        t2 = train_dataset[(train_dataset['boneage']>=57) & (train_dataset['boneage']<=114)].sample(2000)
        t3 = train_dataset[(train_dataset['boneage']>114) & (train_dataset['boneage']<=171)].sample(2000)
        t4 = train_dataset[(train_dataset['boneage']>171) & (train_dataset['boneage']<=228)].sample(2000)
        dataframe = pd.concat([t1, t2, t3, t4])
        dataframe.index = [i for i in range(8000)]
        trainset = BoneDataSet(train_img_path, dataframe, train_composed)
        train_data = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        train_loss = train(model,train_data, epoch)
        val_loss = eval(model, val_data, epoch)
        scheduler.step(val_loss)

        writer.add_scalar('train/epochLoss', train_loss, epoch)
        writer.add_scalar('validation/epochLoss', val_loss,epoch)

        print('{}\nepoch:{}, loss:{}, val_loss:{}\n{}'.format(datetime.datetime.now(),epoch+1, train_loss, val_loss, line))
        
        states = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'val_loss': val_loss
        }

        if (epoch+1) % 5 == 0: save_checkpoint(states, filename='epoch-{}-loss-{:.4f}-val_loss-{:.4f}.pt'.format(epoch+1, train_loss, val_loss))

        if best_loss > val_loss:
            best_model = states
            best_loss = val_loss

        if es.step(val_loss):
            break

    save_checkpoint(best_model, filename='BEST_MODEL-epoch-{}-val_loss-{:.4f}.pt'.format(best_model['epoch']+1,best_loss))

#%%
if __name__ == '__main__':
    freeze_support()
    main()