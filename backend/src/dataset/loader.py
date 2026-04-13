import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from pathlib import Path
import numpy as np
import cv2

class FireDataset(Dataset):
    def __init__(self, img_dir, labels_dir, img_size=640, split='train', augmentations=False):
        self.img_dir = Path(img_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.split = split
        self.augmentations = augmentations
        
        # Get image-label pairs
        self.pairs = []
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        
        for ext in image_extensions:
            images = list(self.img_dir.glob(ext))
            for img_path in images:
                label_path = self.labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    self.pairs.append((img_path, label_path))
                else:
                    print(f"⚠️ No label for {img_path.name}")
        
        print(f"📊 {split}: {len(self.pairs)} image-label pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def _load_yolo_label(self, label_path):
        """Load YOLO bbox: class x_center y_center width height (normalized)"""
        boxes = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        boxes.append(list(map(float, parts)))
        except:
            pass
        return boxes
    
    def __getitem__(self, idx):
        img_path, label_path = self.pairs[idx]
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        # Load labels and convert to absolute coords
        boxes = self._load_yolo_label(label_path)
        boxes_abs = []
        for cls, x, y, w, h in boxes:
            x1 = (x - w/2) * orig_w
            y1 = (y - h/2) * orig_h
            x2 = (x + w/2) * orig_w  
            y2 = (y + h/2) * orig_h
            boxes_abs.append([x1, y1, x2, y2, cls])
        
        # Resize image
        image_resized = cv2.resize(image, (self.img_size, self.img_size))
        image_tensor = torch.from_numpy(image_resized).permute(2,0,1).float() / 255.0
        
        # Convert boxes to tensor (or dummy if empty)
        if boxes_abs:
            target = torch.tensor(boxes_abs, dtype=torch.float32)
        else:
            target = torch.zeros((1, 5))
            
        return image_tensor, target