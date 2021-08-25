# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import parameter
import torch.optim as optim
from torch.optim import lr_scheduler

from earlyStopping import EarlyStopping
from bone_data.DataLoad import train_data_loader,val_data_loader
from model.BoneageModel import BoneAgeNet

from multiprocessing.spawn import freeze_support
import datetime
import matplotlib.pyplot as plt
import math

# For reproducibility use the seeds below (임의 값 고정)
torch.manual_seed(1498920)
torch.cuda.manual_seed(1498920)
torch.backends.cudnn.deterministic=True

# Hyperparameters Setting 
epochs = 100
batch_size = 4
es = EarlyStopping()
save_path = 'D:/model/'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
earlystop = EarlyStopping() # 기본 10 epoch

# data load
train_data = train_data_loader
val_data = val_data_loader
model = BoneAgeNet()

if __name__ == '__main__':
    freeze_support()

    # declare model
    model.to(device)
    
# loss, optimizer, scheduler
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters())
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=10, verbose=1, eps=0.0001, cooldown=5, min_lr=0.0001)

#%%

def save_checkpoint(state, filename='checkpoint.pt'):
    torch.save(state, save_path + filename)

def train(model, train_data, epoch):
    model.train()

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

        if (batch_no + 1) % 25 == 0: print('Epoch {}: {}/12611, batch loss: {}'.format(epoch+1,4*(batch_no+1), loss.item())) # 100장마다 출력
    return epoch_loss / 3153

def eval(model, val_data, epoch):
    model.eval()

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

            if (batch_no + 1) % 25 == 0: print('Epoch {}: {}/1425, batch loss: {}'.format(epoch+1,4*(batch_no+1), loss.item())) # 100장마다 출력
    return epoch_val_loss / 357

def main():
    best_loss = math.inf
    best_model = None
    loss_list = []
    val_list = []
    print('{}\n==============================train start==============================\n'.format(datetime.datetime.now()))
    line = '======================================================================='
    for epoch in range(epochs):
        train_loss = train(model,train_data, epoch)
        val_loss = eval(model, val_data, epoch)
        scheduler.step(val_loss)
        print('{}\nepoch:{}, loss:{}, val_loss:{}\n{}'.format(datetime.datetime.now(),epoch+1, train_loss, val_loss, line))
        
        states = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'val_loss': val_loss
        }

        loss_list.append(train_loss)
        val_list.append(val_loss)

        if (epoch+1) % 5 == 0: save_checkpoint(states, filename='epoch-{}-loss-{:.4f}-val_loss-{:.4f}.pt'.format(epoch+1, train_loss, val_loss))
        if best_loss > val_loss:
            best_model = states

        if es.step(val_loss):
            break

    save_checkpoint(best_model, filename='BEST_MODEL-epoch-{}-val_loss-{:.4f}.tar'.format(best_model['epoch']+1,best_loss))
    return loss_list, val_list

# loss visualization
def display_loss(num, loss_list, val_list):
    plt.figure()
    plt.plot([x for x in range(num)], loss_list, label='loss')
    plt.plot([x for x in range(num)], val_list, label='validation_loss')
    
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.show()
    plt.savefig('result/loss.png')

#%%
if __name__ == '__main__':
    freeze_support()

    loss_list, val_list = main()
    display_loss(len(loss_list), loss_list, val_list)