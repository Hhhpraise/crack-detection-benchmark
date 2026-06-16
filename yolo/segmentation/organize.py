import os
import shutil
import random
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import yaml

# ===== CONFIGURATION =====
# Override via CRACK_WORK_IMAGES env var
_WORK = os.environ.get("CRACK_WORK_IMAGES", "D:/work images")
CRACK_IMAGES_DIR = f"{_WORK}/crack_seg/yolo-seg/positive_batch"
CRACK_LABELS_DIR = f"{_WORK}/crack_seg/yolo-seg/positive_batch"
NON_CRACK_DIR = f"{_WORK}/crack_seg/yolo-seg/negative_batch"
OUTPUT_DIR = "dataset"

TRAIN_RATIO = 0.8
SEED = 42
NON_CRACK_PREFIX = "nc_"


# =========================

def create_dir(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)


def validate_label_format(label_path):
    """Validate and fix label format for segmentation"""
    if not os.path.exists(label_path):
        return False, "File not found"

    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()

        valid_lines = []
        issues = []

        for i, line in enumerate(lines):
            data = line.strip().split()
            if len(data) < 7:  # Need at least class + 3 points (6 coordinates)
                if len(data) == 5:
                    issues.append(f"Line {i + 1}: Detected bounding box format (5 values), need segmentation format")
                else:
                    issues.append(f"Line {i + 1}: Invalid format ({len(data)} values)")
                continue

            # Check if all coordinates are valid floats between 0 and 1
            try:
                class_id = int(data[0])
                coords = [float(x) for x in data[1:]]

                # Check if coordinates are normalized (0-1)
                if all(0 <= coord <= 1 for coord in coords):
                    # Check if we have pairs of coordinates (even number)
                    if len(coords) % 2 == 0:
                        valid_lines.append(line)
                    else:
                        issues.append(f"Line {i + 1}: Odd number of coordinates (need pairs of x,y)")
                else:
                    issues.append(f"Line {i + 1}: Coordinates not normalized (should be 0-1)")

            except ValueError:
                issues.append(f"Line {i + 1}: Invalid numeric values")

        if issues and not valid_lines:
            return False, "; ".join(issues)
        elif issues:
            # Write back only valid lines
            with open(label_path, 'w') as f:
                f.writelines(valid_lines)
            return True, f"Fixed: removed {len(issues)} invalid lines"
        else:
            return True, "Valid segmentation format"

    except Exception as e:
        return False, f"Error reading file: {e}"


def convert_bbox_to_segmentation(bbox_line, img_width, img_height):
    """Convert bounding box line to segmentation polygon"""
    data = bbox_line.strip().split()
    if len(data) != 5:
        return None

    try:
        class_id = int(data[0])
        x_center, y_center, width, height = map(float, data[1:5])

        # Calculate corners (normalized coordinates)
        x1 = max(0, min(1, x_center - width / 2))
        y1 = max(0, min(1, y_center - height / 2))
        x2 = max(0, min(1, x_center + width / 2))
        y2 = max(0, min(1, y_center + height / 2))

        # Create polygon from rectangle corners
        polygon_coords = [x1, y1, x2, y1, x2, y2, x1, y2]

        return f"{class_id} " + " ".join(f"{coord:.6f}" for coord in polygon_coords) + "\n"

    except ValueError:
        return None


def process_and_validate_labels(src_label_path, dst_label_path):
    """Process labels and ensure they're in correct segmentation format"""
    if not os.path.exists(src_label_path):
        # Create empty label file for non-crack images
        with open(dst_label_path, 'w') as f:
            pass
        return True, "Empty label file created"

    try:
        with open(src_label_path, 'r') as f:
            lines = f.readlines()

        converted_lines = []
        bbox_count = 0
        seg_count = 0

        for line in lines:
            data = line.strip().split()
            if not data:
                continue

            if len(data) == 5:
                # Bounding box format - convert to segmentation
                converted_line = convert_bbox_to_segmentation(line, 640, 640)  # Assume 640x640 for conversion
                if converted_line:
                    converted_lines.append(converted_line)
                    bbox_count += 1
            elif len(data) >= 7:
                # Already segmentation format
                converted_lines.append(line)
                seg_count += 1

        # Write processed labels
        with open(dst_label_path, 'w') as f:
            f.writelines(converted_lines)

        status = f"Processed: {bbox_count} bbox->seg, {seg_count} seg kept"
        return True, status

    except Exception as e:
        return False, f"Processing error: {e}"


def visualize_sample(image_path, label_path, output_path):
    """Visualize sample images with segmentation masks"""
    try:
        # Read image
        img = Image.open(image_path)
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img)

        # Read and plot segmentation labels
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                data = line.strip().split()
                if len(data) < 7:  # Need at least class + 3 points
                    continue

                class_id = int(data[0])
                points = list(map(float, data[1:]))

                # Convert normalized coordinates to image coordinates
                h, w = img.size[1], img.size[0]
                polygon_points = [(points[i] * w, points[i + 1] * h) for i in range(0, len(points), 2)]

                if len(polygon_points) >= 3:  # Need at least 3 points for a polygon
                    polygon = patches.Polygon(polygon_points, linewidth=2,
                                              edgecolor='red', facecolor='red', alpha=0.3)
                    ax.add_patch(polygon)

                    # Add text label
                    if polygon_points:
                        ax.text(polygon_points[0][0], polygon_points[0][1], f'Crack {class_id}',
                                color='white', fontsize=12,
                                bbox=dict(facecolor='red', alpha=0.8, pad=2))

        plt.title(f"Sample: {os.path.basename(image_path)}")
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', dpi=100)
        plt.close()
        return True
    except Exception as e:
        print(f"Error visualizing {image_path}: {e}")
        return False


def main():
    random.seed(SEED)

    print("🚀 Starting Dataset Organization for YOLO Segmentation")
    print("=" * 60)

    # Create directory structure
    dirs = {
        'images_train': os.path.join(OUTPUT_DIR, 'images', 'train'),
        'images_val': os.path.join(OUTPUT_DIR, 'images', 'val'),
        'labels_train': os.path.join(OUTPUT_DIR, 'labels', 'train'),
        'labels_val': os.path.join(OUTPUT_DIR, 'labels', 'val'),
        'samples': os.path.join(OUTPUT_DIR, 'samples'),
    }

    for d in dirs.values():
        create_dir(d)

    # Collect and pair image/label paths for crack images
    print("📁 Collecting crack images and labels...")
    crack_data = []
    for img_file in os.listdir(CRACK_IMAGES_DIR):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            base_name = os.path.splitext(img_file)[0]
            label_file = f"{base_name}.txt"
            label_path = os.path.join(CRACK_LABELS_DIR, label_file)
            crack_data.append((img_file, label_file, os.path.exists(label_path)))

    # Collect non-crack images
    print("📁 Collecting non-crack images...")
    non_crack_images = [f for f in os.listdir(NON_CRACK_DIR)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    crack_with_labels = sum(1 for _, _, has_label in crack_data if has_label)
    print(f"✅ Found {len(crack_data)} crack images ({crack_with_labels} with labels)")
    print(f"✅ Found {len(non_crack_images)} non-crack images")

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

    # Process crack images with label validation
    print("\n🔧 Processing crack images and validating labels...")
    label_stats = {"valid": 0, "converted": 0, "missing": 0, "error": 0}

    for data, split in [(crack_train, 'train'), (crack_val, 'val')]:
        print(f"Processing {split} crack images...")
        for img_file, label_file, has_label in data:
            # Copy image
            src_img = os.path.join(CRACK_IMAGES_DIR, img_file)
            dst_img = os.path.join(dirs[f'images_{split}'], img_file)
            shutil.copy(src_img, dst_img)

            # Process label
            src_label = os.path.join(CRACK_LABELS_DIR, label_file) if has_label else None
            dst_label = os.path.join(dirs[f'labels_{split}'], label_file)

            if src_label and os.path.exists(src_label):
                success, message = process_and_validate_labels(src_label, dst_label)
                if success:
                    if "converted" in message.lower() or "bbox" in message.lower():
                        label_stats["converted"] += 1
                    else:
                        label_stats["valid"] += 1
                else:
                    label_stats["error"] += 1
                    print(f"⚠️  {img_file}: {message}")
            else:
                # Create empty label file
                with open(dst_label, 'w') as f:
                    pass
                label_stats["missing"] += 1

    # Process non-crack images
    print("\n🔧 Processing non-crack images...")
    for img_list, split in [(non_crack_train, 'train'), (non_crack_val, 'val')]:
        for img_file in img_list:
            # Create new filename with prefix
            new_img_name = NON_CRACK_PREFIX + img_file
            new_label_name = NON_CRACK_PREFIX + os.path.splitext(img_file)[0] + '.txt'

            # Copy image with new name
            src_img = os.path.join(NON_CRACK_DIR, img_file)
            dst_img = os.path.join(dirs[f'images_{split}'], new_img_name)
            shutil.copy(src_img, dst_img)

            # Create empty label file
            dst_label = os.path.join(dirs[f'labels_{split}'], new_label_name)
            with open(dst_label, 'w') as f:
                pass

    # Generate file lists
    print("\n📝 Generating file lists...")
    train_files = []
    val_files = []

    # Add crack images to file lists
    for img_file, _, _ in crack_train:
        train_files.append(os.path.abspath(os.path.join(dirs['images_train'], img_file)))

    for img_file, _, _ in crack_val:
        val_files.append(os.path.abspath(os.path.join(dirs['images_val'], img_file)))

    # Add non-crack images to file lists
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

    # Generate data.yaml with proper segmentation task
    yaml_content = {
        'path': os.path.abspath(OUTPUT_DIR),
        'train': os.path.abspath(os.path.join(OUTPUT_DIR, 'train.txt')),
        'val': os.path.abspath(os.path.join(OUTPUT_DIR, 'val.txt')),
        'names': {0: 'crack'},
        'nc': 1,
        'task': 'segment'  # Explicitly specify segmentation task
    }

    with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    # Create sample visualizations
    print("\n🎨 Creating sample visualizations...")
    sample_count = 0
    successful_viz = 0

    # Visualize crack samples
    for img_file, label_file, _ in crack_train[:5]:
        img_path = os.path.join(dirs['images_train'], img_file)
        label_path = os.path.join(dirs['labels_train'], label_file)
        output_path = os.path.join(dirs['samples'], f'sample_{sample_count}_crack.png')

        if visualize_sample(img_path, label_path, output_path):
            successful_viz += 1
        sample_count += 1

    # Visualize non-crack sample
    if non_crack_train:
        img_file = non_crack_train[0]
        new_img_name = NON_CRACK_PREFIX + img_file
        img_path = os.path.join(dirs['images_train'], new_img_name)
        output_path = os.path.join(dirs['samples'], f'sample_{sample_count}_non_crack.png')

        try:
            img = Image.open(img_path)
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.title(f"Non-crack sample: {new_img_name}")
            plt.axis('off')
            plt.savefig(output_path, bbox_inches='tight', dpi=100)
            plt.close()
            successful_viz += 1
        except Exception as e:
            print(f"Error creating non-crack visualization: {e}")

    # Generate comprehensive report
    total_images = len(crack_data) + len(non_crack_images)
    total_train = len(crack_train) + len(non_crack_train)
    total_val = len(crack_val) + len(non_crack_val)

    report = f"""
🎉 DATASET ORGANIZATION COMPLETE!
={'=' * 50}

📊 DATASET SUMMARY:
   Total images: {total_images}
     • Crack images: {len(crack_data)}
     • Non-crack images: {len(non_crack_images)}

   Training samples: {total_train}
     • Crack training: {len(crack_train)}
     • Non-crack training: {len(non_crack_train)}

   Validation samples: {total_val}
     • Crack validation: {len(crack_val)}
     • Non-crack validation: {len(non_crack_val)}

🏷️ LABEL PROCESSING SUMMARY:
   • Valid segmentation labels: {label_stats['valid']}
   • Converted bbox→segmentation: {label_stats['converted']}
   • Missing labels (empty files): {label_stats['missing']}
   • Processing errors: {label_stats['error']}

📁 DATASET STRUCTURE:
{OUTPUT_DIR}/
├── images/
│   ├── train/ ({total_train} images)
│   └── val/ ({total_val} images)
├── labels/
│   ├── train/ ({total_train} label files)
│   └── val/ ({total_val} label files)
├── samples/ ({successful_viz} sample visualizations)
├── train.txt (training image paths)
├── val.txt (validation image paths)
└── data.yaml (YOLO configuration)

💡 READY FOR TRAINING:
   Your dataset is now formatted for YOLO segmentation training!
   Run: python train.py

🔧 CONFIGURATION:
   • Task: Segmentation
   • Classes: 1 (crack)
   • Non-crack prefix: '{NON_CRACK_PREFIX}'
   • Train/Val split: {int(TRAIN_RATIO * 100)}%/{int((1 - TRAIN_RATIO) * 100)}%
"""

    print(report)

    # Save report to file
    with open(os.path.join(OUTPUT_DIR, 'dataset_report.txt'), 'w') as f:
        f.write(report)

    print(f"\n✅ Dataset preparation complete!")
    print(f"📋 Detailed report saved to: {os.path.join(OUTPUT_DIR, 'dataset_report.txt')}")
    print(f"🎨 Sample visualizations: {dirs['samples']}")

    if label_stats['error'] > 0:
        print(f"\n⚠️  Warning: {label_stats['error']} labels had processing errors")
        print("   Check the console output above for details")

    if label_stats['converted'] > 0:
        print(f"\n🔄 Info: Converted {label_stats['converted']} bounding box labels to segmentation format")


if __name__ == "__main__":
    main()