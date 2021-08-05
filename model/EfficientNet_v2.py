import torch
import torch.nn as nn
import math
import timm

class EffNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super(EffNetV2, self).__init__()
        self.effv2 = timm.create_model('efficientnetv2_s', num_classes=400, in_chans=1)
        self.eff_relu = nn.ReLU()

        # Fully Connected Layer for  gender
        self.gen_fc_1 = nn.Linear(1,16)
        self.gen_relu  = nn.ReLU()

        # Feature Fully Connected Layer
        self.cat_fc = nn.Linear(16+400,200)
        self.cat_relu = nn.ReLU()
        
        # Final Fully Connected Layer
        self.final_fc2 = nn.Linear(200, num_classes)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()

    def forward(self, x, y):
# =============================================================================
#       EfficientNet Layers        
# =============================================================================
        x = self.effv2(x)
        x = self.eff_relu(x)
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

# 'efficientnetv2_l'
# 'efficientnetv2_m'
# 'efficientnetv2_rw_m'
# 'efficientnetv2_rw_s'
# 'efficientnetv2_s'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = EffNetV2(num_classes=1)
print(model)