# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from bone_data.DataLoad import test_data_loader, test_data
from model.BoneageModel import BoneAgeNet

import cv2
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

# data load
model = BoneAgeNet()

if __name__ == '__main__':
    freeze_support()
    sample_batch = next(iter(test_data_loader))

    # declare model
    model.eval()
    model.to(device)
    checkpoint = torch.load(save_path+'epoch-50-loss-8.6348-val_loss-8.0583.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    target_layer = model.efficientnet_v2.act2

    

    cam = GradCAM(model = model, target_layer = target_layer)
    target_category = 1
    grayscale_cam = cam(input_tensor=sample_batch, target_category=target_category)
    visualization = show_cam_on_image(cv2.imread('bone_data/test/4360.png'), grayscale_cam)

