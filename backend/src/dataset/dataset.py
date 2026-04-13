import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import albumentations as A

class FireVisionDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.img_files = sorted(list(self.img_dir.glob('*.jpg')) + list(self.img_dir.glob('*.png')))
        print(f"Loaded {len(self.img_files)} images from {img_dir}")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = self.label_dir / f"{img_path.stem}.txt"
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load labels (YOLO format: class x_center y_center width height)
        bboxes = []
        classes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls = float(parts[0])
                        x, y, w, h = map(float, parts[1:])
                        bboxes.append([x, y, w, h])
                        classes.append(int(cls))
        
        bboxes = np.array(bboxes) if bboxes else np.empty((0, 4))
        classes = np.array(classes) if classes else np.empty((0,))
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=classes)
            image = transformed['image']
            bboxes = transformed['bboxes']
            classes = transformed['class_labels']
        
        return {
            'image': image,
            'bboxes': torch.FloatTensor(bboxes),
            'classes': torch.LongTensor(classes)
        }