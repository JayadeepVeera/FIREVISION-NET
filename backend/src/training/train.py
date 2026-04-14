#!/usr/bin/env python3
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import yaml
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset.loader import FireDataset
from src.models.baseline import FireVisionNet


class CIoULoss(nn.Module):
    def forward(self, pred_boxes, target_boxes):
        # pred_boxes/target_boxes: [N, 4] xywh normalized
        pred_xyxy = self.xywh2xyxy(pred_boxes)
        target_xyxy = self.xywh2xyxy(target_boxes)
        return 1.0 - self.compute_ciou(pred_xyxy, target_xyxy)
    
    def xywh2xyxy(self, boxes):
        x, y, w, h = boxes.unbind(-1)
        return torch.stack([x-w/2, y-h/2, x+w/2, y+h/2], dim=-1)
    
    def compute_ciou(self, box1, box2):
        # Simplified CIoU implementation
        iou = self.bbox_iou(box1, box2)
        
        # Distance cost
        rho2 = ((box2[:, 0]+box2[:, 2]-box1[:, 0]-box1[:, 2])**2 +
                (box2[:, 1]+box2[:, 3]-box1[:, 1]-box1[:, 3])**2) / 4
        
        c2 = (torch.max(box1[:, 2], box2[:, 2]) - 
              torch.min(box1[:, 0], box2[:, 0]))**2 + \
             (torch.max(box1[:, 3], box2[:, 3]) - 
              torch.min(box1[:, 1], box2[:, 1]))**2
        
        dist_cost = rho2 / (c2 + 1e-7)
        
        return iou - dist_cost
    
    def bbox_iou(self, box1, box2):
        # Standard IoU
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        
        inter_x1 = torch.max(box1[:, 0], box2[:, 0])
        inter_y1 = torch.max(box1[:, 1], box2[:, 1])
        inter_x2 = torch.min(box1[:, 2], box2[:, 2])
        inter_y2 = torch.min(box1[:, 3], box2[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)
        
        union = area1 + area2 - inter_area + 1e-7
        return inter_area / union


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()


class DetectionLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ciou_loss = CIoULoss()
        self.focal_cls = FocalLoss(cfg["loss"]["focal_alpha"], cfg["loss"]["focal_gamma"])
        self.focal_obj = FocalLoss(cfg["loss"]["focal_alpha"], cfg["loss"]["focal_gamma"])
        
        self.box_w = cfg["loss"]["box_weight"]
        self.cls_w = cfg["loss"]["cls_weight"] 
        self.obj_w = cfg["loss"]["obj_weight"]
        self.num_classes = cfg["dataset"]["nc"]
    
    def forward(self, preds, targets):
        pred_obj = preds["obj"]   # [B, 400, 1]
        pred_cls = preds["cls"]   # [B, 400, 2]
        pred_box = preds["box"]   # [B, 400, 4]
        
        B, N, _ = pred_obj.shape
        
        total_loss = 0
        box_loss = 0
        cls_loss = 0
        obj_loss = 0
        
        for b in range(B):
            # Get targets for this image
            tgt_boxes = targets[b]["boxes"]      # [num_gt, 4]
            tgt_labels = targets[b]["labels"]    # [num_gt]
            
            if len(tgt_boxes) == 0:
                # No objects: objectness loss only (negative examples)
                obj_loss += self.focal_obj(pred_obj[b], torch.zeros_like(pred_obj[b]))
                continue
            
            # Assign each GT to closest prediction (simple greedy assignment)
            best_iou = torch.zeros(len(tgt_boxes))
            assigned_preds = torch.full((N,), -1, dtype=torch.long)
            
            for i, (gt_box, gt_cls) in enumerate(zip(tgt_boxes, tgt_labels)):
                ious = 1 - self.ciou_loss(gt_box.unsqueeze(0), pred_box[b])
                best_idx = ious.argmax()
                if ious[best_idx] > 0.5:  # IoU threshold
                    assigned_preds[best_idx] = i
                    best_iou[i] = ious[best_idx]
            
            # Positive samples
            pos_mask = assigned_preds >= 0
            if pos_mask.sum() > 0:
                pos_obj_tgt = torch.ones_like(pred_obj[b][pos_mask])
                pos_cls_tgt = F.one_hot(assigned_preds[pos_mask], self.num_classes).float()
                pos_box_tgt = torch.stack([tgt_boxes[assigned_preds[pos_mask] == i] 
                                         for i in range(len(tgt_boxes)) if assigned_preds[pos_mask] == i])
                
                box_loss += self.ciou_loss(pred_box[b][pos_mask], pos_box_tgt).mean()
                obj_loss += self.focal_obj(pred_obj[b][pos_mask], pos_obj_tgt).mean()
                cls_loss += self.focal_cls(pred_cls[b][pos_mask], pos_cls_tgt).mean()
            
            # Negative samples
            neg_mask = ~pos_mask
            neg_obj_tgt = torch.zeros_like(pred_obj[b][neg_mask])
            obj_loss += self.focal_obj(pred_obj[b][neg_mask], neg_obj_tgt).mean()
        
        total_loss = (self.box_w * box_loss + self.cls_w * cls_loss + self.obj_w * obj_loss) / B
        
        return total_loss, {
            "box": box_loss.item() / B,
            "cls": cls_loss.item() / B, 
            "obj": obj_loss.item() / B
        }


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    images = torch.stack(images, 0)
    return images, targets


def main():
    cfg = yaml.safe_load(open("configs/train.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data
    train_ds = FireDataset(cfg["dataset"]["train"], cfg["dataset"]["train_labels"])
    val_ds = FireDataset(cfg["dataset"]["val"], cfg["dataset"]["val_labels"], split="val")
    
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], 
                            shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], 
                          shuffle=False, collate_fn=collate_fn, num_workers=2)
    
    # Model
    model = FireVisionNet(num_classes=cfg["dataset"]["nc"]).to(device)
    criterion = DetectionLoss(cfg)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], 
                          weight_decay=cfg["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    
    best_loss = float('inf')
    patience_counter = 0
    
    os.makedirs(cfg["training"]["save_dir"], exist_ok=True)
    save_path = os.path.join(cfg["training"]["save_dir"], cfg["training"]["save_name"])
    
    print(f"🚀 Training {cfg['training']['epochs']} epochs on {device}")
    
    for epoch in range(cfg["training"]["epochs"]):
        # Train
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader)
        
        for imgs, targets in pbar:
            imgs = imgs.to(device)
            targets = [t for t in targets]  # Keep list
            
            optimizer.zero_grad()
            preds = model(imgs)
            
            loss, stats = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        train_loss /= len(train_loader)
        
        # Validate  
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = [t for t in targets]
                preds = model(imgs)
                loss, _ = criterion(preds, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step()
        
        print(f"Epoch {epoch+1:3d}: Train={train_loss:.4f} Val={val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved best model: {save_path}")
        else:
            patience_counter += 1
        
        if patience_counter >= cfg["training"]["patience"]:
            print("🛑 Early stopping")
            break
    
    print("🎉 Training complete!")
    print(f"✅ Best model saved: {save_path}")


if __name__ == "__main__":
    main()