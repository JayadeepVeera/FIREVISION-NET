import os
from collections import Counter

DATA_DIR = "data"
SPLITS = ["train", "valid", "test"]

image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_files(folder, exts=None):
    if not os.path.exists(folder):
        return []
    files = []
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if os.path.isfile(path):
            if exts is None or os.path.splitext(f.lower())[1] in exts:
                files.append(f)
    return files

def main():
    total_images = 0
    total_labels = 0
    class_counter = Counter()
    bad_lines = []

    for split in SPLITS:
        img_dir = os.path.join(DATA_DIR, split, "images")
        lbl_dir = os.path.join(DATA_DIR, split, "labels")

        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            print(f"[!] Missing split folders for: {split}")
            continue

        images = list_files(img_dir, image_exts)
        labels = list_files(lbl_dir, {".txt"})

        image_stems = {os.path.splitext(f)[0] for f in images}
        label_stems = {os.path.splitext(f)[0] for f in labels}

        missing_labels = sorted(image_stems - label_stems)
        missing_images = sorted(label_stems - image_stems)

        print(f"\n--- {split.upper()} ---")
        print(f"Images: {len(images)}")
        print(f"Labels: {len(labels)}")
        print(f"Missing labels: {len(missing_labels)}")
        print(f"Missing images: {len(missing_images)}")

        total_images += len(images)
        total_labels += len(labels)

        for lbl in labels:
            lbl_path = os.path.join(lbl_dir, lbl)
            with open(lbl_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        bad_lines.append((lbl_path, i, line))
                        continue
                    cls, xc, yc, w, h = parts
                    try:
                        cls = int(cls)
                        xc, yc, w, h = map(float, [xc, yc, w, h])
                        class_counter[cls] += 1
                        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            bad_lines.append((lbl_path, i, line))
                    except:
                        bad_lines.append((lbl_path, i, line))

    print("\n=== SUMMARY ===")
    print(f"Total images: {total_images}")
    print(f"Total labels: {total_labels}")
    print(f"Class counts: {dict(class_counter)}")

    if bad_lines:
        print(f"\nFound {len(bad_lines)} bad annotation lines. Showing first 10:")
        for item in bad_lines[:10]:
            print(item)
    else:
        print("\nNo bad annotation lines found.")

if __name__ == "__main__":
    main()