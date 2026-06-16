import os
import shutil
import random
from pathlib import Path

# ===== CONFIGURATION =====
# Override via CRACK_WORK_IMAGES env var
_WORK = os.environ.get("CRACK_WORK_IMAGES", "D:/work images")
CRACK_IMAGES_DIR = f"{_WORK}/crack_detection/positive_batch"
CRACK_LABELS_DIR = f"{_WORK}/crack_detection/positive_batch"
NON_CRACK_DIR = f"{_WORK}/crack_detection/negative_batch"
OUTPUT_DIR = "dataset"  # Output directory for organized dataset

TRAIN_RATIO = 0.8  # 80% training, 20% validation
SEED = 42  # Random seed for reproducibility
NON_CRACK_PREFIX = "nc_"  # Prefix for non-crack images to avoid filename conflicts


# =========================

def create_dir(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)


def main():
    random.seed(SEED)

    # Create directory structure
    dirs = {
        'images_train': os.path.join(OUTPUT_DIR, 'images', 'train'),
        'images_val': os.path.join(OUTPUT_DIR, 'images', 'val'),
        'labels_train': os.path.join(OUTPUT_DIR, 'labels', 'train'),
        'labels_val': os.path.join(OUTPUT_DIR, 'labels', 'val'),
    }

    for d in dirs.values():
        create_dir(d)

    # Collect and pair image/label paths for crack images
    crack_data = []
    for img_file in os.listdir(CRACK_IMAGES_DIR):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            base_name = os.path.splitext(img_file)[0]
            label_file = f"{base_name}.txt"
            label_path = os.path.join(CRACK_LABELS_DIR, label_file)
            if os.path.exists(label_path):
                crack_data.append((img_file, label_file))

    # Collect non-crack images
    non_crack_images = [f for f in os.listdir(NON_CRACK_DIR)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Found {len(crack_data)} crack images with annotations")
    print(f"Found {len(non_crack_images)} non-crack images")

    # Split crack images into train/val
    random.shuffle(crack_data)
    split_idx = int(len(crack_data) * TRAIN_RATIO)
    crack_train = crack_data[:split_idx]
    crack_val = crack_data[split_idx:]

    # Split non-crack images into train/val
    random.shuffle(non_crack_images)
    split_idx = int(len(non_crack_images) * TRAIN_RATIO)
    non_crack_train = non_crack_images[:split_idx]
    non_crack_val = non_crack_images[split_idx:]

    # Process crack images
    for data, split in [(crack_train, 'train'), (crack_val, 'val')]:
        for img_file, label_file in data:
            # Copy image
            src_img = os.path.join(CRACK_IMAGES_DIR, img_file)
            dst_img = os.path.join(dirs[f'images_{split}'], img_file)
            shutil.copy(src_img, dst_img)

            # Copy label
            src_label = os.path.join(CRACK_LABELS_DIR, label_file)
            dst_label = os.path.join(dirs[f'labels_{split}'], label_file)
            shutil.copy(src_label, dst_label)

    # Process non-crack images - add prefix to avoid filename conflicts
    for img_list, split in [(non_crack_train, 'train'), (non_crack_val, 'val')]:
        for img_file in img_list:
            # Create new filename with prefix
            new_img_name = NON_CRACK_PREFIX + img_file
            new_label_name = NON_CRACK_PREFIX + os.path.splitext(img_file)[0] + '.txt'

            # Copy image with new name
            src_img = os.path.join(NON_CRACK_DIR, img_file)
            dst_img = os.path.join(dirs[f'images_{split}'], new_img_name)
            shutil.copy(src_img, dst_img)

            # Create empty label file with new name
            dst_label = os.path.join(dirs[f'labels_{split}'], new_label_name)
            open(dst_label, 'w').close()

    # Generate train.txt and val.txt
    train_files = []
    val_files = []

    # Add crack images
    for img_file, _ in crack_train:
        train_files.append(os.path.abspath(os.path.join(dirs['images_train'], img_file)))

    for img_file, _ in crack_val:
        val_files.append(os.path.abspath(os.path.join(dirs['images_val'], img_file)))

    # Add non-crack images (with prefix)
    for img_file in non_crack_train:
        new_name = NON_CRACK_PREFIX + img_file
        train_files.append(os.path.abspath(os.path.join(dirs['images_train'], new_name)))

    for img_file in non_crack_val:
        new_name = NON_CRACK_PREFIX + img_file
        val_files.append(os.path.abspath(os.path.join(dirs['images_val'], new_name)))

    # Write file lists
    with open(os.path.join(OUTPUT_DIR, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_files))

    with open(os.path.join(OUTPUT_DIR, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_files))

    # Generate updated data.yaml
    yaml_content = f"""path: {os.path.abspath(OUTPUT_DIR)}
train: {os.path.abspath(os.path.join(OUTPUT_DIR, 'train.txt'))}
val: {os.path.abspath(os.path.join(OUTPUT_DIR, 'val.txt'))}
names:
  0: crack
"""

    with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
        f.write(yaml_content)

    print(f"\nDataset preparation complete!")
    print(f"Total images: {len(crack_data) + len(non_crack_images)}")
    print(f"  Crack images: {len(crack_data)}")
    print(f"  Non-crack images: {len(non_crack_images)}")
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    print(f"Updated data.yaml created at: {os.path.join(OUTPUT_DIR, 'data.yaml')}")
    print(f"Non-crack images prefixed with: '{NON_CRACK_PREFIX}' to avoid filename conflicts")


if __name__ == "__main__":
    main()