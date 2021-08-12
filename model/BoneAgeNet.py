# encoding:utf-8

import torch
import torch.nn as nn
import math
from timm.models.efficientnet import _gen_efficientnetv2_m



def efficientnetv2_m(pretrained=False, **kwargs):
    """ EfficientNet-V2 Small. """
    model = _gen_efficientnetv2_m('efficientnetv2_m', pretrained=False, num_classes=400, 
    in_chans=1,drop_rate=0.2,drop_path_rate=0.2, **kwargs)
    return model

class BoneAgeNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(BoneAgeNet, self).__init__()

        # EfficientNet-V2 Layer
        self.efficientnet = efficientnetv2_m()
        self.effi_relu = nn.ReLU()

        # Fully Connected Layer for  gender
        self.gen_fc_1 = nn.Linear(1,16)
        self.gen_relu  = nn.ReLU()

        # Feature Fully Connected Layer
        self.cat_fc = nn.Linear(16+400,200)
        self.cat_relu = nn.ReLU()
        
        # Final Fully Connected Layer
        self.final_fc2 = nn.Linear(200, num_classes)
        self.sigmoid = nn.Sigmoid()

        # 초기화  (Weight Initialization)
        for m in self.modules(): # 모듈을 차례대로 불러옴
            # 불러온 모듈이 nn.Conv2d인 경우
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            # nn.BatchNorm2d인 경우
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # Forward Pass. x = Image tensor, y = gender tensor
    def forward(self, x,y):
# =============================================================================
#       EfficientNet-v2 Layers        
# =============================================================================

        x = self.efficientnet(x)
        x = self.effi_relu(x)
        x = x.view(x.size(0), -1)


# =============================================================================
#       Gender Fully Connected Layer
# =============================================================================
        y = self.gen_fc_1(y)
        y = self.gen_relu(y)
        y = y.view(y.size(0), -1)

        
# =============================================================================
#       Feature Concatenation & shuffle Layer
# =============================================================================
      
        z = torch.cat((x,y),dim = 1)

        #idx = torch.randperm(z.shape[0])
        #z = z[idx].view(z.size())
        
        z = self.cat_fc(z)
        z = self.cat_relu(z)

# =============================================================================
#       Final FC Layer
# =============================================================================
        
        z = self.final_fc2(z)
        z = self.sigmoid(z)
        return z



#%%
# =============================================================================
#       Model Print
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = BoneAgeNet(num_classes = 1).to(device)
print(model)
