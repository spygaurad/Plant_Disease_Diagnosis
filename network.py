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


class DomainAdaptiveNet(nn.Module):
    def __init__(self):
        super(DomainAdaptiveNet, self).__init__()
        self.effnet = EfficientNet()
        self.effnet.load_state_dict(torch.load('saved_model/TOMATO_LEAF_PLANTVILLAGE_EFFICIENTNET_10CLASSES_V1_6_10.pth', map_location=torch.device(DEVICE)))
        self.shared_layers = self.effnet.model
        self.classificationLayer = nn.Sequential(self.effnet.outlayer(self.effnet.relu(self.effnet.preFinal())))
        self.domainClassificationLayer = nn.Sequential(nn.Linear(64, 32), nn.ReLU, nn.Linear(32, 8), nn.ReLU(8, 2))

    
    def forward(self, x):
        shared_features = self.effnet(x)
        diseaseClass = self.classificationLayer(x)
        domain = self.domainClassificationLayer(x)

        return shared_features, diseaseClass, domain