# -*- coding: utf-8 -*-

from timm.models.efficientnet import _gen_efficientnetv2_m
import torch
import torch.nn as nn
import torch.nn.functional as F

def efficientnetv2_m(pretrained=False, **kwargs):
    """ EfficientNet-V2 Small. """
    model = _gen_efficientnetv2_m('efficientnetv2_m', pretrained=pretrained, **kwargs)
    return model

class BoneAgeNet(nn.Module):

    def __init__(self, transform_input=False):
        super(BoneAgeNet, self).__init__()
        
        # Backbone
        self.efficientnet_v2 = efficientnetv2_m(in_chans=1, drop_rate=0.2, drop_path_rate=0.2)
        self.efficientnet_v2.global_pool = nn.AdaptiveAvgPool2d(2)
        self.efficientnet_v2.classifier = nn.Identity()
        self.flatten1 = nn.Flatten()

        # Gender
        self.gender = DenseLayer(1, 16)
        self.flatten2 = nn.Flatten()

        # FC Layer
        self.fc_1 = DenseLayer(5136, 1000)
        self.fc_2 = DenseLayer(1000, 1000)

        # Final Layer
        self.fc_3 = nn.Linear(1000, 1)

    def forward(self, x, y):
        x = self.efficientnet_v2(x)
        x = self.flatten1(x)

        y = self.gender(y)
        y= self.flatten2(y)
        
        z = self.fc_1(torch.cat([x, y], 1))
        z = self.fc_2(z)
        z = self.fc_3(z)
        return z

class DenseLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = self.linear(x)
        return F.relu(x, inplace=True)



#%%
# =============================================================================
#       Model Print
# =============================================================================
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = BoneAgeNet().to(device)
summary(model,[(1,500,500), (1,1,1)])