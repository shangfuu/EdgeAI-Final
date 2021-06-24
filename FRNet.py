import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models


class FRNet(nn.Module):
    def __init__(self, pretrained=False, classify=False, num_classes=None, data="LFW"):
        super().__init__()

        self.classify = classify
        self.num_classses = num_classes

    
        self.model = models.resnet50(pretrained=pretrained)

        # Drop Linear
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        # Add new linear
        self.linear = nn.Linear(2048, 512, bias=False)

        # Add Dropout
        self.bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)            
        
        if data == "LFW":
            # 428 classes in my custom LFW dataset
            self.logits = nn.Linear(412, 428)
        else:
            # WebFace
            self.logits = nn.Linear(512, 10575)
            

        if self.classify and self.num_classses is not None:
            self.logits = nn.Linear(512, self.num_classses)



    def forward(self, x):
        x = self.model(x)

        x = x.view(x.size(0), -1)

        x = self.linear(x)

        x = self.bn(x)

        if self.classify:
            x = self.logits(x)
        else:
            x = F.normalize(x, p=2, dim=1)

        return x
