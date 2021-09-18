# -*- coding: utf-8 -*-

from timm.models.efficientnet import _gen_efficientnetv2_s

import torch
import torch.nn as nn

class BoneAgeNet(nn.Module):

    def __init__(self, drop_rate=0.0):
        super(BoneAgeNet, self).__init__()
        
        # Backbone
        self.efficientnet_v2 = _gen_efficientnetv2_s('efficientnetv2_s',in_chans=1)
        self.efficientnet_v2.global_pool = nn.AdaptiveAvgPool2d(2)
        self.efficientnet_v2.classifier = nn.Identity()
        self.flatten1 = nn.Flatten()

        # Gender
        self.gender = nn.Linear(1,16)
        self.gen_mish = nn.Mish(inplace=True)
        self.flatten2 = nn.Flatten()

        # FC Layer1
        self.fc_1 = nn.Linear(5120+16, 1000)
        self.mish1 = nn.Mish(inplace=True)
        
        # FC Layer2
        self.fc_2 = nn.Linear(1000,1000)
        self.mish2 = nn.Mish(inplace=True)
        # Final Layer
        self.fc_3 = nn.Linear(1000, 1)

    def forward(self, x, y):
        x = self.efficientnet_v2(x)
        x = self.flatten1(x)

        y = self.gender(y)
        y = self.gen_mish(y)
        y= self.flatten2(y)

        z = self.fc_1(torch.cat([x, y], 1))
        z = self.mish1(z)

        z = self.fc_2(z)
        z = self.mish2(z)

        z = self.fc_3(z)
        return z

'''
#%%
# =============================================================================
#       Model Print
# =============================================================================

from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = BoneAgeNet().to(device)
summary(model,[(1,500,500), (1,1,1)])
'''