import torchvision.models as models
import timm
import torch
import torch.nn as nn

class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        self.model = timm.create_model('tf_efficientnetv2_b0', num_classes=64, pretrained=True)
        self.preFinal = nn.Linear(64, 32)
        self.outlayer = nn.Linear(32, 10)
        self.relu = nn.ReLU()
        # self.model = models.efficientnet_b0(pretrained=True)
        # self.model.classifier[1] = nn.Linear(1280, 10)
    
    def forward(self, x): return self.outlayer(self.relu(self.preFinal(self.model(x))))

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# input = torch.rand(4, 3, 224, 224).to(device)
# model = EfficientNet().to(device)
# output = model(input)
