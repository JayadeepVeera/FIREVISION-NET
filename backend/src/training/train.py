#!/usr/bin/env python3
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path: 
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os

from src.dataset.loader import FireDataset
from src.models.baseline import FireVisionNet

def collate_fn(batch):
    """Custom collate for variable YOLO labels"""
    imgs_list, targets_list = zip(*batch)
    imgs_batch = torch.stack(imgs_list, 0)
    
    # Simple classification targets for now
    targets_batch = torch.ones(len(imgs_list), 1)
    return imgs_batch, targets_batch

def main():
    # Load config with BOM handling
    config_path = 'configs/dataset.yaml'
    with open(config_path, 'r', encoding='utf-8-sig') as f:
        config = yaml.safe_load(f)
    
    # Clean BOM keys
    fixed_config = {}
    for key, value in config.items():
        fixed_config[key.lstrip('\ufeff')] = value
    config = fixed_config
    
    print("✅ Config loaded:", list(config.keys()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training FireVision-Net on {device}")
    
    # Model
    model = FireVisionNet(num_classes=1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📐 Model params: {total_params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    # Data
    train_ds = FireDataset(
        img_dir=config['dataset']['train_path'],
        labels_dir=config['dataset']['train_labels'],
        img_size=config['dataset']['img_size']
    )
    val_ds = FireDataset(
        img_dir=config['dataset']['val_path'],
        labels_dir=config['dataset']['val_labels'],
        img_size=config['dataset']['img_size'],
        split='val'
    )
    
    train_loader = DataLoader(train_ds, batch_size=config['dataset']['batch_size'], 
                            shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config['dataset']['batch_size'], 
                          shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    print(f"📊 Train: {len(train_loader)} batches ({len(train_ds)} images)")
    print(f"📊 Val:   {len(val_loader)} batches ({len(val_ds)} images)")
    
    # Training loop - FIXED
    os.makedirs('weights', exist_ok=True)
    best_loss = float('inf')
    patience_counter = 0  # ← FIXED: Initialize here
    patience = config['training'].get('patience', 5)
    
    for epoch in range(config['training']['epochs']):
        # Train
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["training"]["epochs"]}')
        
        for imgs, targets in pbar:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            
            # Classification loss
            if preds.dim() > 2:
                preds = preds.mean(dim=[2,3])
            loss = criterion(preds, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                preds = model(imgs)
                if preds.dim() > 2:
                    preds = preds.mean(dim=[2,3])
                loss = criterion(preds, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"📊 Epoch {epoch+1}: Train={avg_train_loss:.4f} | Val={avg_val_loss:.4f}")
        
        # Save best
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0  # Reset
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': best_loss
            }, 'weights/firevision_best.pt')
            print(f"💾 Best model saved! Val: {best_loss:.4f}")
        else:
            patience_counter += 1  # Now safe
            print(f"⏳ No improvement. Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print("🛑 Early stopping")
            break
    
    print("🎉 Training COMPLETE!")
    print("✅ Best model: weights/firevision_best.pt")

if __name__ == "__main__":
    main()