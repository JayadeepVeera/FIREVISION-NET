import torch
import torch.nn as nn
import torch.nn.functional as F

class FireVisionNet(nn.Module):
    def __init__(self, num_classes=2, backbone="dual_path_cnn", pretrained=False):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = 20  # 20x20 grid = 400 anchor points
        self.backbone = backbone
        
        if backbone == "dual_path_cnn":
            self.backbone_layers = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(20)  # Output 20x20 feature map
            )
        else:  # yolo-style
            self.backbone_layers = nn.Sequential(
                nn.Conv2d(3, 64, 6, 2, 2), nn.ReLU(),
                nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(256, 512, 3, 2, 1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(20)
            )
        
        # Detection heads for each grid cell (400 cells)
        self.obj_head = nn.Conv2d(256, 1, 1)      # Objectness
        self.cls_head = nn.Conv2d(256, num_classes, 1)  # Classes
        self.box_head = nn.Conv2d(256, 4, 1)      # Box regression (x,y,w,h)
    
    def forward(self, x):
        features = self.backbone_layers(x)  # [B, 256, 20, 20]
        
        obj = self.obj_head(features)    # [B, 1, 20, 20]
        cls = self.cls_head(features)    # [B, 2, 20, 20] 
        box = self.box_head(features)    # [B, 4, 20, 20]
        
        # Reshape to [B, 400, C] format
        B, _, H, W = obj.shape
        obj = obj.view(B, 1, -1).permute(0, 2, 1)      # [B, 400, 1]
        cls = cls.view(B, self.num_classes, -1).permute(0, 2, 1)  # [B, 400, 2]
        box = box.view(B, 4, -1).permute(0, 2, 1)      # [B, 400, 4]
        
        return {
            "obj": obj,   # Objectness logits [B, 400, 1]
            "cls": cls,   # Class logits [B, 400, 2]  
            "box": box    # Box predictions [B, 400, 4] (x,y,w,h normalized)
        }