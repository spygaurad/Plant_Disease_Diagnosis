import timm
import torch
import torch.nn as nn

class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        self.model = timm.create_model('tf_efficientnetv2_b0', num_classes=11)
    
    def forward(self, x):
        output = self.model(x)
        return output

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# input = torch.rand(4, 3, 224, 224).to(device)
# model = EfficientNet().to(device)
# print(model)
# output = model(input)
# print(output.shape)
