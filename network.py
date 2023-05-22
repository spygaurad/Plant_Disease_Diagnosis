import timm
import torch
import torch.nn as nn

class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        self.model = timm.create_model('tf_efficientnetv2_b0', num_classes=64, pretrained=True)
        self.outLayer = nn.Linear(64, 10)
    
    def forward(self, x): return self.outLayer(self.model(x))

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# input = torch.rand(4, 3, 224, 224).to(device)
# model = EfficientNet().to(device)
# output = model(input)
