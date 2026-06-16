"""
DUAL-LABEL DATASET UNIFICATION TOOL
Creates separate datasets for detection and segmentation with their respective labels
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import json


class DualLabelDatasetUnifier:
    def __init__(self):
        # Allow overriding base path via environment variable
        _WORK = os.environ.get("CRACK_WORK_IMAGES", "D:/work images")

        # DETECTION DATASET PATHS
        self.detection_root = f"{_WORK}/crack_detection"
        self.detection_positive_batches = f"{_WORK}/crack_detection/positive_batch"
        self.detection_negative_batches = f"{_WORK}/crack_detection/negative_batch"

        # SEGMENTATION DATASET PATHS
        self.segmentation_root = f"{_WORK}/crack_seg/yolo-seg"
        self.segmentation_positive_batches = f"{_WORK}/crack_seg/yolo-seg/positive_batch"
        self.segmentation_negative_batches = f"{_WORK}/crack_seg/yolo-seg/negative_batch"

        # SEGMENTATION LABELS PATH
        self.segmentation_labels_folder = f"{_WORK}/crack_seg/yolo-seg/labels/train"

        # MASK PATHS
        self.segmentation_masks_folder = f"{_WORK}/swinn/SegmentationClass"

        # OUTPUT PATHS - relative to cwd
        self.output_detection_dataset = "unified_crack_dataset"
        self.output_segmentation_dataset = "unified_crack_dataset_seg"

        # CONFIGURATION
        self.test_split_ratio = 0.2
        self.include_negative_images = True
        self.random_seed = 42

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    def find_all_images_in_batches(self, base_path, batch_type='positive'):
        """Find all images in flat batch folders"""
        images = {}
        base_path = Path(base_path)

        if not base_path.exists():
            print(f"⚠️ Warning: Path does not exist: {base_path}")
            return images

        print(f"   Scanning {batch_type} batches in: {base_path}")

        for img_file in base_path.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in self.image_extensions:
                unique_key = f"{batch_type}_{img_file.name}"
                images[unique_key] = {
                    'filename': img_file.name,
                    'path': str(img_file),
                    'batch_folder': str(base_path),
                    'type': batch_type
                }

        print(f"      Found {len(images)} images")
        return images

    def find_detection_label_file(self, filename, batch_folder):
        """Find detection label in the SAME folder as the image"""
        img_stem = Path(filename).stem
        batch_path = Path(batch_folder)
        label_path = batch_path / f"{img_stem}.txt"
        return str(label_path) if label_path.exists() else None

    def find_segmentation_label_file(self, filename, labels_base_path):
        """Find segmentation label in the labels folder"""
        img_stem = Path(filename).stem
        labels_path = Path(labels_base_path)

        # Try different possible structures
        possible_paths = [
            labels_path / f"{img_stem}.txt",  # Flat structure
            labels_path / 'positive_batch' / f"{img_stem}.txt",  # Subfolder
            labels_path / 'train' / f"{img_stem}.txt",  # Train/val split
        ]

        for label_path in possible_paths:
            if label_path.exists():
                return str(label_path)

        return None

    def find_mask_file(self, filename, masks_base_path):
        """Find corresponding segmentation mask"""
        img_stem = Path(filename).stem
        masks_path = Path(masks_base_path)
        mask_path = masks_path / f"{img_stem}.png"
        return str(mask_path) if mask_path.exists() else None

    def validate_label_file(self, label_path):
        """Check if label file has content"""
        if not label_path or not os.path.exists(label_path):
            return False
        with open(label_path, 'r') as f:
            content = f.read().strip()
            return len(content) > 0

    def is_segmentation_label(self, label_path):
        """
        Determine if a label file is segmentation format (polygons) or detection format (bbox)
        Segmentation: class_id x1 y1 x2 y2 x3 y3 x4 y4 ... (more than 5 values)
        Detection: class_id x_center y_center width height (exactly 5 values)
        """
        if not label_path or not os.path.exists(label_path):
            return False

        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 5:  # More than bbox format = segmentation
                        return True
            return False
        except:
            return False

    def validate_mask_file(self, mask_path):
        """Check if mask has any foreground pixels"""
        if not mask_path or not os.path.exists(mask_path):
            return False
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return False
        return np.any(mask > 127)

    def create_empty_label_file(self, output_path):
        """Create empty label file for negative images"""
        with open(output_path, 'w') as f:
            pass

    def create_empty_mask(self, output_path, height, width):
        """Create empty mask for negative images"""
        empty_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.imwrite(output_path, empty_mask)

    def create_dual_datasets(self):
        """Main function to create BOTH detection and segmentation datasets"""
        print("=" * 100)
        print("DUAL-LABEL DATASET UNIFICATION TOOL")
        print("Creates separate datasets for detection and segmentation")
        print("=" * 100)

        # Step 1: Find all images
        print("\n🔍 Step 1: Scanning for images in both datasets...")

        print("\n📂 Detection Dataset:")
        det_positive_imgs = self.find_all_images_in_batches(self.detection_positive_batches, 'positive')
        det_negative_imgs = self.find_all_images_in_batches(self.detection_negative_batches, 'negative')

        print("\n📂 Segmentation Dataset:")
        seg_positive_imgs = self.find_all_images_in_batches(self.segmentation_positive_batches, 'positive')
        seg_negative_imgs = self.find_all_images_in_batches(self.segmentation_negative_batches, 'negative')

        print(f"\n📊 Summary:")
        print(f"   Detection: {len(det_positive_imgs)} positive, {len(det_negative_imgs)} negative")
        print(f"   Segmentation: {len(seg_positive_imgs)} positive, {len(seg_negative_imgs)} negative")

        # Step 2: Match images and find ALL annotations
        print("\n🔗 Step 2: Matching images and finding annotations...")
        matched_data = []

        stats = {
            'positive_common': 0,
            'positive_with_det_label': 0,
            'positive_with_seg_label': 0,
            'positive_with_mask': 0,
            'positive_with_all': 0,
            'negative_common': 0,
        }

        # Process POSITIVE images
        print("\n   Processing POSITIVE images...")
        det_positive_filenames = {v['filename']: k for k, v in det_positive_imgs.items()}
        seg_positive_filenames = {v['filename']: k for k, v in seg_positive_imgs.items()}

        common_positive = set(det_positive_filenames.keys()) & set(seg_positive_filenames.keys())
        stats['positive_common'] = len(common_positive)

        print(f"      Found {len(common_positive)} common positive images")
        print(f"      Searching for labels in: {self.segmentation_labels_folder}")

        for filename in tqdm(common_positive, desc="      Matching positive"):
            det_key = det_positive_filenames[filename]
            seg_key = seg_positive_filenames[filename]

            det_info = det_positive_imgs[det_key]
            seg_info = seg_positive_imgs[seg_key]

            # Find detection label (bbox format)
            det_label_path = self.find_detection_label_file(filename, det_info['batch_folder'])

            # Find segmentation label (polygon format)
            seg_label_path = self.find_segmentation_label_file(filename, self.segmentation_labels_folder)

            # Find mask
            mask_path = self.find_mask_file(filename, self.segmentation_masks_folder)

            has_det_label = self.validate_label_file(det_label_path)
            has_seg_label = self.validate_label_file(seg_label_path)
            has_mask = self.validate_mask_file(mask_path)

            if has_det_label:
                stats['positive_with_det_label'] += 1
            if has_seg_label:
                stats['positive_with_seg_label'] += 1
            if has_mask:
                stats['positive_with_mask'] += 1

            # Only include if we have at least detection label OR segmentation label
            if has_det_label or has_seg_label:
                if has_det_label and has_seg_label and has_mask:
                    stats['positive_with_all'] += 1

                matched_data.append({
                    'filename': filename,
                    'det_image_path': det_info['path'],
                    'seg_image_path': seg_info['path'],
                    'det_label_path': det_label_path if has_det_label else None,
                    'seg_label_path': seg_label_path if has_seg_label else None,
                    'mask_path': mask_path if has_mask else None,
                    'type': 'positive',
                    'has_crack': True,
                    'has_det_label': has_det_label,
                    'has_seg_label': has_seg_label,
                    'has_mask': has_mask
                })

        # Process NEGATIVE images
        if self.include_negative_images:
            print("\n   Processing NEGATIVE images...")
            det_negative_filenames = {v['filename']: k for k, v in det_negative_imgs.items()}
            seg_negative_filenames = {v['filename']: k for k, v in seg_negative_imgs.items()}

            common_negative = set(det_negative_filenames.keys()) & set(seg_negative_filenames.keys())
            stats['negative_common'] = len(common_negative)

            print(f"      Found {len(common_negative)} common negative images")

            for filename in tqdm(common_negative, desc="      Matching negative"):
                det_key = det_negative_filenames[filename]
                seg_key = seg_negative_filenames[filename]

                det_info = det_negative_imgs[det_key]
                seg_info = seg_negative_imgs[seg_key]

                matched_data.append({
                    'filename': filename,
                    'det_image_path': det_info['path'],
                    'seg_image_path': seg_info['path'],
                    'det_label_path': None,
                    'seg_label_path': None,
                    'mask_path': None,
                    'type': 'negative',
                    'has_crack': False,
                    'has_det_label': False,
                    'has_seg_label': False,
                    'has_mask': False
                })

        # Print statistics
        print(f"\n📊 Annotation Statistics:")
        print(f"   POSITIVE images:")
        print(f"      Common in both datasets: {stats['positive_common']}")
        print(f"      With detection labels (bbox): {stats['positive_with_det_label']}")
        print(f"      With segmentation labels (polygon): {stats['positive_with_seg_label']}")
        print(f"      With masks: {stats['positive_with_mask']}")
        print(f"      With ALL annotations: {stats['positive_with_all']}")
        print(f"   NEGATIVE images:")
        print(f"      Common in both datasets: {stats['negative_common']}")

        positive_count = sum(1 for x in matched_data if x['has_crack'])
        negative_count = len(matched_data) - positive_count

        print(f"\n✅ Matched {len(matched_data)} total images:")
        print(f"   Positive (with cracks): {positive_count}")
        print(f"   Negative (no cracks): {negative_count}")

        if len(matched_data) == 0:
            print("\n❌ ERROR: No matching images found!")
            return False

        # Step 3: Split into train/test (use same split for both datasets)
        print(f"\n📊 Step 3: Splitting dataset (test ratio: {self.test_split_ratio})...")

        random.shuffle(matched_data)

        test_size = int(len(matched_data) * self.test_split_ratio)
        test_data = matched_data[:test_size]
        train_data = matched_data[test_size:]

        test_positive = sum(1 for x in test_data if x['has_crack'])
        train_positive = sum(1 for x in train_data if x['has_crack'])

        print(f"   Train: {len(train_data)} images ({train_positive} positive, {len(train_data) - train_positive} negative)")
        print(f"   Test: {len(test_data)} images ({test_positive} positive, {len(test_data) - test_positive} negative)")

        # Step 4: Create DETECTION dataset
        print("\n🔍 Step 4: Creating DETECTION dataset...")
        self._create_dataset_structure(
            self.output_detection_dataset,
            train_data,
            test_data,
            label_type='detection'
        )

        # Step 5: Create SEGMENTATION dataset
        print("\n🎨 Step 5: Creating SEGMENTATION dataset...")
        self._create_dataset_structure(
            self.output_segmentation_dataset,
            train_data,
            test_data,
            label_type='segmentation'
        )

        # Success message
        print("\n" + "=" * 100)
        print("✅ DUAL DATASETS CREATED SUCCESSFULLY!")
        print("=" * 100)

        print(f"\n📁 Output Locations:")
        print(f"   Detection dataset: {Path(self.output_detection_dataset).absolute()}")
        print(f"   Segmentation dataset: {Path(self.output_segmentation_dataset).absolute()}")

        print(f"\n📊 Dataset Statistics:")
        print(f"   Total images: {len(matched_data)}")
        print(f"   ├── Positive (with cracks): {positive_count}")
        print(f"   └── Negative (no cracks): {negative_count}")
        print(f"\n   Training set: {len(train_data)} images")
        print(f"   Testing set: {len(test_data)} images")

        print("\n💡 Next Steps:")
        print("   1. Use unified_crack_dataset/data.yaml for DETECTION training")
        print("   2. Use unified_crack_dataset_seg/data.yaml for SEGMENTATION training")
        print("   3. Run fixed_yolo_trainer.py to train both models")

        print("\n" + "=" * 100)

        return True

    def _create_dataset_structure(self, output_path, train_data, test_data, label_type='detection'):
        """
        Create dataset structure for either detection or segmentation

        Args:
            output_path: Output directory
            train_data: List of training data
            test_data: List of testing data
            label_type: 'detection' or 'segmentation'
        """
        output_path = Path(output_path)

        # Create structure
        for split in ['train', 'test']:
            (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
            if label_type == 'segmentation':
                (output_path / 'masks' / split).mkdir(parents=True, exist_ok=True)

        (output_path / 'annotations').mkdir(parents=True, exist_ok=True)

        # Copy files
        for split, data_list in [('train', train_data), ('test', test_data)]:
            print(f"   Copying {split} files...")

            for item in tqdm(data_list, desc=f"   {split}"):
                filename = item['filename']
                stem = Path(filename).stem

                # Copy image
                src_img = item['det_image_path']
                dst_img = output_path / 'images' / split / filename
                shutil.copy2(src_img, dst_img)

                # Copy appropriate label
                dst_label = output_path / 'labels' / split / f"{stem}.txt"

                if label_type == 'detection':
                    # Use detection label (bbox)
                    if item['det_label_path']:
                        shutil.copy2(item['det_label_path'], dst_label)
                    else:
                        self.create_empty_label_file(dst_label)

                elif label_type == 'segmentation':
                    # Use segmentation label (polygon)
                    if item['seg_label_path']:
                        shutil.copy2(item['seg_label_path'], dst_label)
                    else:
                        self.create_empty_label_file(dst_label)

                    # Copy mask
                    dst_mask = output_path / 'masks' / split / f"{stem}.png"
                    if item['mask_path']:
                        shutil.copy2(item['mask_path'], dst_mask)
                    else:
                        img = cv2.imread(src_img)
                        if img is not None:
                            h, w = img.shape[:2]
                            self.create_empty_mask(dst_mask, h, w)

        # Save dataset info
        positive_count = sum(1 for x in train_data + test_data if x['has_crack'])
        train_positive = sum(1 for x in train_data if x['has_crack'])
        test_positive = sum(1 for x in test_data if x['has_crack'])

        info = {
            'dataset_type': label_type,
            'total_images': len(train_data) + len(test_data),
            'positive_images': positive_count,
            'negative_images': len(train_data) + len(test_data) - positive_count,
            'train_images': len(train_data),
            'test_images': len(test_data),
            'test_split_ratio': self.test_split_ratio,
            'train_positive': train_positive,
            'train_negative': len(train_data) - train_positive,
            'test_positive': test_positive,
            'test_negative': len(test_data) - test_positive,
        }

        with open(output_path / 'annotations' / 'dataset_info.json', 'w') as f:
            json.dump(info, f, indent=4)

        # Create data.yaml
        yaml_content = f"""# Crack {'Detection' if label_type == 'detection' else 'Segmentation'} Dataset
path: {output_path.absolute()}
train: images/train
val: images/test
test: images/test

# Classes
names:
  0: crack

# Dataset info
nc: 1
task: {'detect' if label_type == 'detection' else 'segment'}
"""

        with open(output_path / 'data.yaml', 'w') as f:
            f.write(yaml_content)

        print(f"   ✅ {label_type.capitalize()} dataset created at: {output_path}")


def main():
    print("\n" + "=" * 100)
    print("STARTING DUAL-LABEL DATASET UNIFICATION")
    print("=" * 100)

    print("\n📝 IMPORTANT: Update the segmentation labels path!")
    print("   Edit line 25 in this script:")
    print("   self.segmentation_labels_folder = 'YOUR_PATH_HERE'")

    print("\n   Common locations to check:")
    print("   - D:/work images/crack_seg/yolo-seg/labels")
    print("   - D:/work images/crack_seg/labels")
    print("   - D:/work images/crack_seg/yolo-seg/annotations")

    unifier = DualLabelDatasetUnifier()

    # Verify segmentation labels folder exists
    if not Path(unifier.segmentation_labels_folder).exists():
        print(f"\n⚠️ WARNING: Segmentation labels folder not found:")
        print(f"   {unifier.segmentation_labels_folder}")
        print("\n   Please update the path and run again.")
        return

    success = unifier.create_dual_datasets()

    if success:
        print("\n🎉 Success! Both datasets are ready for training.")
    else:
        print("\n❌ Failed to create datasets. Check errors above.")


if __name__ == "__main__":
    main()