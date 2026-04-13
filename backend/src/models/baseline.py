import torch
import torch.nn as nn

class FireVisionNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.head(x)