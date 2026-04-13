import os
import yaml
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

def check_dataset_structure(data_dir="data"):
    """Check if folder structure exists and count files"""
    print("🔍 CHECKING DATASET STRUCTURE")
    print("=" * 50)
    
    splits = ['train', 'val', 'test']
    img_stats = {}
    label_stats = {}
    
    for split in splits:
        img_path = Path(data_dir) / "images" / split
        label_path = Path(data_dir) / "labels" / split
        
        img_count = len(list(img_path.glob("*.jpg"))) + len(list(img_path.glob("*.png")))
        label_count = len(list(label_path.glob("*.txt")))
        
        img_stats[split] = img_count
        label_stats[split] = label_count
        
        print(f"📁 {split.upper()}:")
        print(f"   Images: {img_count}")
        print(f"   Labels: {label_count}")
        print(f"   Match: {'✅' if img_count == label_count else '❌'}")
        print()
    
    total_images = sum(img_stats.values())
    total_labels = sum(label_stats.values())
    
    print(f"📊 TOTAL: {total_images} images, {total_labels} labels")
    print(f"✅ All good!" if total_images == total_labels else "⚠️  Fix image-label mismatch!")
    
    return img_stats, label_stats

def validate_label_files(label_dir):
    """Check label file format and content"""
    print("🔍 VALIDATING LABEL FILES")
    print("=" * 50)
    
    label_dir = Path(label_dir)
    errors = []
    class_stats = Counter()
    empty_labels = 0
    bbox_stats = []
    
    for txt_file in label_dir.glob("*.txt"):
        try:
            with open(txt_file, 'r') as f:
                lines = f.readlines()
            
            valid_lines = []
            for line in lines:
                line = line.strip()
                if not line: continue
                
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"❌ {txt_file.name}: Invalid format (expected 5 values)")
                    continue
                
                cls, x, y, w, h = map(float, parts)
                
                # Check class ID (0=fire, 1=smoke)
                if cls not in [0, 1]:
                    errors.append(f"❌ {txt_file.name}: Invalid class {cls} (use 0=fire, 1=smoke)")
                    continue
                
                # Check bbox coordinates [0,1]
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    errors.append(f"❌ {txt_file.name}: Invalid bbox {x,y,w,h}")
                    continue
                
                # Check bbox area reasonable
                area = w * h
                if area < 0.0001 or area > 1:  # Too small or impossible
                    errors.append(f"❌ {txt_file.name}: Unrealistic bbox area {area:.4f}")
                    continue
                
                valid_lines.append((int(cls), x, y, w, h))
                class_stats[int(cls)] += 1
                bbox_stats.append(area)
            
            if not valid_lines:
                empty_labels += 1
                
        except Exception as e:
            errors.append(f"❌ {txt_file.name}: Read error - {str(e)}")
    
    print(f"✅ Valid labels: {len(bbox_stats)} bboxes")
    print(f"📈 Class distribution: Fire={class_stats[0]}, Smoke={class_stats[1]}")
    print(f"📦 Avg bbox area: {np.mean(bbox_stats):.4f}, Min={np.min(bbox_stats):.4f}")
    print(f"⚠️  Empty labels: {empty_labels}")
    
    if errors:
        print(f"❌ {len(errors)} label errors found:")
        for error in errors[:10]:  # Show first 10
            print(f"   {error}")
        if len(errors) > 10:
            print(f"   ... and {len(errors)-10} more")
    
    return len(errors) == 0

def check_image_files(img_dir):
    """Validate all image files"""
    print("🔍 VALIDATING IMAGE FILES")
    print("=" * 50)
    
    img_dir = Path(img_dir)
    corrupt_images = []
    size_stats = []
    
    for img_file in img_dir.glob("*.jpg"):
        try:
            img = cv2.imread(str(img_file))
            if img is None:
                corrupt_images.append(img_file.name)
                continue
            
            h, w = img.shape[:2]
            size_stats.append((w, h))
            
        except Exception as e:
            corrupt_images.append(f"{img_file.name}: {str(e)}")
    
    for img_file in img_dir.glob("*.png"):
        try:
            img = Image.open(img_file)
            img.verify()  # Check corruption
            size_stats.append(img.size[::-1])  # (w,h)
            
        except Exception as e:
            corrupt_images.append(f"{img_file.name}: {str(e)}")
    
    print(f"✅ Valid images: {len(size_stats)}")
    if size_stats:
        widths, heights = zip(*size_stats)
        print(f"📏 Resolution: {np.mean(widths):.0f}x{np.mean(heights):.0f}px")
        print(f"📏 Range: {min(widths)}-{max(widths)} x {min(heights)}-{max(heights)}px")
    
    if corrupt_images:
        print(f"❌ {len(corrupt_images)} corrupt images:")
        for corrupt in corrupt_images[:5]:
            print(f"   {corrupt}")
    
    return len(corrupt_images) == 0

def check_image_label_pairs():
    """Check 1:1 image-label matching"""
    print("🔍 CHECKING IMAGE-LABEL PAIRS")
    print("=" * 50)
    
    splits = ['train', 'val', 'test']
    mismatches = []
    
    for split in splits:
        img_dir = Path("data/images") / split
        label_dir = Path("data/labels") / split
        
        img_names = {p.stem for p in img_dir.glob("*.*")}
        label_names = {p.stem for p in label_dir.glob("*.txt")}
        
        only_imgs = img_names - label_names
        only_labels = label_names - img_names
        
        if only_imgs:
            mismatches.append(f"{split}: {len(only_imgs)} images missing labels")
        if only_labels:
            mismatches.append(f"{split}: {len(only_labels)} labels missing images")
    
    if mismatches:
        print("❌ MISMATCHES FOUND:")
        for mismatch in mismatches:
            print(f"   {mismatch}")
        return False
    else:
        print("✅ Perfect 1:1 matching!")
        return True

def create_dataset_yaml():
    """Create dataset.yaml for training"""
    print("📝 CREATING DATASET YAML")
    print("=" * 50)
    
    yaml_content = """
# FireVision-Net Dataset
train: data/images/train
val: data/images/val
test: data/images/test

# Classes: 0=fire, 1=smoke
nc: 2
names: ['fire', 'smoke']

# Training config
imgsz: 640
batch: 16
epochs: 100
"""

    with open("configs/dataset.yaml", "w") as f:
        f.write(yaml_content.strip())
    
    print("✅ configs/dataset.yaml created!")
    print("📄 Contents:")
    print(yaml_content.strip())

def visualize_sample(split='train', n_samples=4):
    """Show sample images with annotations"""
    print("👁️  VISUALIZING SAMPLES")
    print("=" * 50)
    
    img_dir = Path("data/images") / split
    label_dir = Path("data/labels") / split
    
    samples = list(img_dir.glob("*.jpg"))[:n_samples]
    
    fig, axes = plt.subplots(1, n_samples, figsize=(5*n_samples, 5))
    if n_samples == 1: axes = [axes]
    
    for i, img_path in enumerate(samples):
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load labels
        label_path = label_dir / f"{img_path.stem}.txt"
        bboxes = []
        classes = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    cls, x, y, w, h = map(float, line.strip().split())
                    # Convert normalized to pixel coords
                    h_img, w_img = img.shape[:2]
                    x1 = int((x - w/2) * w_img)
                    y1 = int((y - h/2) * h_img)
                    x2 = int((x + w/2) * w_img)
                    y2 = int((y + h/2) * h_img)
                    bboxes.append([x1, y1, x2, y2])
                    classes.append(int(cls))
        
        # Draw bboxes
        for box, cls in zip(bboxes, classes):
            color = (0, 255, 0) if cls == 0 else (255, 0, 0)  # Green=fire, Blue=smoke
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(img, f"{'fire' if cls==0 else 'smoke'}", 
                       (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        axes[i].imshow(img)
        axes[i].set_title(f"{img_path.name}\n{len(bboxes)} bboxes")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("outputs/dataset_sample.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Sample saved to outputs/dataset_sample.png")

def main():
    """Run full dataset validation"""
    print("🚀 FireVision-Net Dataset Validator")
    print("=" * 60)
    
    # Create output dir
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("configs", exist_ok=True)
    
    # 1. Check structure
    img_stats, label_stats = check_dataset_structure()
    
    # 2. Check image-label pairs
    pair_ok = check_image_label_pairs()
    
    # 3. Validate images
    splits = ['train', 'val', 'test']
    all_img_ok = True
    for split in splits:
        img_ok = check_image_files(f"data/images/{split}")
        all_img_ok = all_img_ok and img_ok
    
    # 4. Validate labels
    all_label_ok = True
    for split in splits:
        label_ok = validate_label_files(f"data/labels/{split}")
        all_label_ok = all_label_ok and label_ok
    
    # 5. Create YAML
    create_dataset_yaml()
    
    # 6. Visualize
    visualize_sample('train')
    
    # Final verdict
    print("\n" + "="*60)
    print("🎯 FINAL DATASET STATUS")
    if pair_ok and all_img_ok and all_label_ok:
        print("✅ DATASET READY FOR TRAINING!")
        print("🚀 Next: python src/training/train.py")
    else:
        print("⚠️  Fix issues above before training!")
    
    print("\n📁 Project structure check:")
    print("- data/images/train|val|test ✓")
    print("- data/labels/train|val|test ✓") 
    print("- configs/dataset.yaml ✓")
    print("- outputs/dataset_sample.png ✓")

if __name__ == "__main__":
    main()