# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from earlyStopping import EarlyStopping
from bone_data.DataLoad import test_data_loader, test_data
from model.BoneageModel import BoneAgeNet

from multiprocessing.spawn import freeze_support
import datetime
import numpy as np

# For reproducibility use the seeds below
torch.manual_seed(1498920)
torch.cuda.manual_seed(1498920)
torch.backends.cudnn.deterministic=True

# Hyperparameters Setting 
save_path = 'D:/model/'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
earlystop = EarlyStopping() # 기본 10 epoch

# data load
model = BoneAgeNet()

if __name__ == '__main__':
    freeze_support()

    # declare model
    model.to(device)
    
# loss, optimizer, scheduler
criterion = nn.L1Loss()

#%%


def eval(model, test_data):
    model.eval()

    result_array = np.array([])
    test_loss = 0.0

    with torch.no_grad():
        for batch_no, batch in enumerate(test_data):

            # Load batch
            img = batch['image'].to(device)
            gender = batch['gender'].to(device)
            age = batch['bone_age'].to(device)
            
            # Forward propagation (순전파)
            output = model(img, gender)
            loss = criterion(output, age)

            preds = output.cpu().numpy()
            preds = preds.reshape(preds.shape[0])
            result_array = np.concatenate((result_array,preds))

            test_loss += loss.item()

            if (batch_no + 1) % 25 == 0: print('{}/200, batch loss: {}'.format(4*(batch_no+1), loss.item())) # 100장마다 출력
    return result_array, test_loss / 50

#%%
if __name__ == '__main__':
    freeze_support()

    checkpoint = torch.load(save_path+'BEST_MODEL-epoch-26-val_loss-144.6790.tar')
    model.load_state_dict(checkpoint['model_state_dict'])

    print('{}\n==============================test start==============================\n'.format(datetime.datetime.now()))
    
    result_array, test_loss = eval(model, test_data_loader)
    predict_df = test_data.copy()
    predict_df['output'] = result_array
    predict_df['output'] = np.round(predict_df['output'], decimals=2)
    predict_df['MAE'] = np.abs(predict_df['output']-predict_df['boneage'])

    print('MAE={}'.format(predict_df['MAE'].mean()))

    predict_df.to_csv('predict.csv', sep=',', na_rep='NaN')