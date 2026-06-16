import os
import json
import shutil
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2
import glob


class CrackDetectionDataOrganizer:
    def __init__(self, source_dir, output_dir):
        """
        Initialize the detection data organizer

        Args:
            source_dir: Directory containing your current data structure
            output_dir: Directory where organized data will be saved
        """
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.image_size = 224

        # Create output directory structure
        self.create_directory_structure()

    def create_directory_structure(self):
        """Create the required directory structure for detection training"""
        directories = [
            'images/train',
            'images/val',
            'images/test',
            'labels/train',
            'labels/val',
            'labels/test',
            'annotations'
        ]

        for dir_path in directories:
            full_path = os.path.join(self.output_dir, dir_path)
            os.makedirs(full_path, exist_ok=True)

    def parse_coco_annotations(self, json_path):
        """
        Parse COCO format annotations from merged JSON file

        Args:
            json_path: Path to merged COCO JSON file

        Returns:
            Dictionary mapping image_id to annotations and image info
        """
        print(f"\n{'=' * 60}")
        print(f"PARSING ANNOTATIONS FROM: {json_path}")
        print(f"{'=' * 60}")

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return {}, {}
        except FileNotFoundError:
            print(f"File not found: {json_path}")
            return {}, {}

        # Create mappings
        image_id_to_info = {}
        for img in data['images']:
            image_id = img['id']
            image_id_to_info[image_id] = img

        print(f"Found {len(image_id_to_info)} images in JSON")
        print(f"  - Image ID range: {min(image_id_to_info.keys())} to {max(image_id_to_info.keys())}")

        category_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
        print(f"Categories: {list(category_id_to_name.values())}")

        # Group annotations by image_id
        image_annotations = {}
        valid_annotations = 0
        invalid_annotations = 0
        orphan_annotations = 0

        for ann in data['annotations']:
            try:
                image_id = ann['image_id']

                # Check if image exists
                if image_id not in image_id_to_info:
                    orphan_annotations += 1
                    if orphan_annotations <= 5:
                        print(f"Warning: Annotation {ann['id']} references non-existent image_id {image_id}")
                    continue

                if image_id not in image_annotations:
                    image_annotations[image_id] = []

                image_info = image_id_to_info[image_id]
                img_width = image_info['width']
                img_height = image_info['height']

                # Validate bbox format
                if 'bbox' not in ann or len(ann['bbox']) != 4:
                    invalid_annotations += 1
                    continue

                bbox = ann['bbox']  # [x, y, width, height]

                # Validate bbox values
                if (bbox[2] <= 0 or bbox[3] <= 0 or bbox[0] < 0 or bbox[1] < 0):
                    invalid_annotations += 1
                    continue

                # Convert COCO bbox format to YOLO format (normalized)
                x_center = (bbox[0] + bbox[2] / 2) / img_width
                y_center = (bbox[1] + bbox[3] / 2) / img_height
                width = bbox[2] / img_width
                height = bbox[3] / img_height

                # Validate normalized coordinates
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                        0 < width <= 1 and 0 < height <= 1):
                    invalid_annotations += 1
                    continue

                # Class ID (crack = 0)
                class_id = 0

                image_annotations[image_id].append({
                    'class_id': class_id,
                    'bbox': [x_center, y_center, width, height],
                    'original_bbox': bbox,
                    'area': ann.get('area', bbox[2] * bbox[3])
                })
                valid_annotations += 1

            except Exception as e:
                invalid_annotations += 1
                if invalid_annotations <= 5:
                    print(f"Error processing annotation: {e}")
                continue

        # Statistics
        images_with_annotations = len([img_id for img_id in image_annotations if len(image_annotations[img_id]) > 0])
        images_without_annotations = len(image_id_to_info) - images_with_annotations

        print(f"\n{'=' * 60}")
        print("ANNOTATION STATISTICS")
        print(f"{'=' * 60}")
        print(f"Valid annotations: {valid_annotations}")
        print(f"Invalid annotations: {invalid_annotations}")
        if orphan_annotations > 0:
            print(f"Orphan annotations (no matching image): {orphan_annotations}")
        print(f"\nImages with annotations: {images_with_annotations}")
        print(f"Images without annotations: {images_without_annotations}")

        # Show annotation distribution
        if image_annotations:
            ann_counts = [len(anns) for anns in image_annotations.values() if len(anns) > 0]
            print(f"\nAnnotations per image:")
            print(f"  - Min: {min(ann_counts)}")
            print(f"  - Max: {max(ann_counts)}")
            print(f"  - Average: {sum(ann_counts) / len(ann_counts):.2f}")

        return image_annotations, image_id_to_info

    def find_image_file(self, image_filename, search_dirs):
        """Search for image file in multiple directories"""
        for search_dir in search_dirs:
            # Try exact match
            image_path = os.path.join(search_dir, image_filename)
            if os.path.exists(image_path):
                return image_path

            # Try with different extensions
            base_name = os.path.splitext(image_filename)[0]
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                image_path = os.path.join(search_dir, base_name + ext)
                if os.path.exists(image_path):
                    return image_path

            # Try case-insensitive search
            if os.path.exists(search_dir):
                for file in os.listdir(search_dir):
                    if file.lower() == image_filename.lower():
                        return os.path.join(search_dir, file)

        return None

    def process_images(self, image_path, output_path):
        """Process and resize images for detection"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                return False

            image = cv2.resize(image, (self.image_size, self.image_size))
            cv2.imwrite(output_path, image)
            return True

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return False

    def create_yolo_labels(self, annotations, output_label_path):
        """Create YOLO format label file"""
        try:
            with open(output_label_path, 'w') as f:
                for ann in annotations:
                    class_id = ann['class_id']
                    bbox = ann['bbox']
                    line = f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
                    f.write(line)
            return True
        except Exception as e:
            print(f"Error creating label file {output_label_path}: {e}")
            return False

    def organize_detection_data(self, images_dirs, json_path, split_ratios=(0.7, 0.2, 0.1)):
        """Organize detection data into train/val/test splits"""
        print("\n" + "=" * 60)
        print("STARTING DETECTION DATA ORGANIZATION")
        print("=" * 60)

        # Parse annotations
        image_annotations, image_id_to_info = self.parse_coco_annotations(json_path)

        if not image_annotations:
            print("\nNo valid annotations found. Please check your JSON file.")
            return {}

        # Search for image files
        print(f"\n{'=' * 60}")
        print("SEARCHING FOR IMAGE FILES")
        print(f"{'=' * 60}")
        print(f"Search directories:")
        for d in images_dirs:
            print(f"  - {d}")

        annotated_images = []
        missing_images = []
        found_by_dir = {d: 0 for d in images_dirs}

        for image_id, img_info in image_id_to_info.items():
            if image_id in image_annotations and len(image_annotations[image_id]) > 0:
                img_filename = img_info['file_name']
                img_path = self.find_image_file(img_filename, images_dirs)

                if img_path and os.path.exists(img_path):
                    for d in images_dirs:
                        if img_path.startswith(d):
                            found_by_dir[d] += 1
                            break

                    annotated_images.append({
                        'image_id': image_id,
                        'image_path': img_path,
                        'filename': img_filename,
                        'annotations': image_annotations[image_id]
                    })
                else:
                    missing_images.append((image_id, img_filename))

        print(f"\n{'=' * 60}")
        print("IMAGE SEARCH RESULTS")
        print(f"{'=' * 60}")
        print(f"Found images: {len(annotated_images)}")
        for d, count in found_by_dir.items():
            if count > 0:
                print(f"  - {os.path.basename(d)}: {count} images")

        if missing_images:
            print(f"\nMissing images: {len(missing_images)}")
            print("First 10 missing files:")
            for img_id, img_filename in missing_images[:10]:
                print(f"  - ID {img_id}: {img_filename}")
            if len(missing_images) > 10:
                print(f"  ... and {len(missing_images) - 10} more")

        if len(annotated_images) == 0:
            print("\nNo annotated images found! Please check your JSON file and image paths.")
            return {}

        annotated_images.sort(key=lambda x: x['image_id'])

        # Split data
        print(f"\n{'=' * 60}")
        print(f"SPLITTING DATA (train: {split_ratios[0]:.0%}, val: {split_ratios[1]:.0%}, test: {split_ratios[2]:.0%})")
        print(f"{'=' * 60}")

        train_ratio, val_ratio, test_ratio = split_ratios

        image_paths = [img['image_path'] for img in annotated_images]
        annotations_list = [img['annotations'] for img in annotated_images]
        filenames = [img['filename'] for img in annotated_images]
        image_ids = [img['image_id'] for img in annotated_images]

        # First split: train+val vs test
        train_val_paths, test_paths, train_val_anns, test_anns, train_val_names, test_names, train_val_ids, test_ids = train_test_split(
            image_paths, annotations_list, filenames, image_ids,
            test_size=test_ratio, random_state=42, shuffle=True
        )

        # Second split: train vs val
        val_size = val_ratio / (train_ratio + val_ratio)
        train_paths, val_paths, train_anns, val_anns, train_names, val_names, train_ids, val_ids = train_test_split(
            train_val_paths, train_val_anns, train_val_names, train_val_ids,
            test_size=val_size, random_state=42, shuffle=True
        )

        splits = {
            'train': (train_paths, train_anns, train_names, train_ids),
            'val': (val_paths, val_anns, val_names, val_ids),
            'test': (test_paths, test_anns, test_names, test_ids)
        }

        data_info = {}

        for split_name, (img_paths, annotations, filenames, image_ids) in splits.items():
            print(f"\n{'=' * 60}")
            print(f"PROCESSING {split_name.upper()} SPLIT: {len(img_paths)} images")
            print(f"{'=' * 60}")

            processed_count = 0
            failed_count = 0

            for i, (img_path, anns, filename, img_id) in enumerate(zip(img_paths, annotations, filenames, image_ids)):
                base_name = f"{split_name}_{processed_count:05d}"

                img_output = os.path.join(self.output_dir, f'images/{split_name}', f'{base_name}.jpg')
                if self.process_images(img_path, img_output):
                    label_output = os.path.join(self.output_dir, f'labels/{split_name}', f'{base_name}.txt')
                    if self.create_yolo_labels(anns, label_output):
                        processed_count += 1
                        if processed_count % 100 == 0:
                            print(f"  Processed {processed_count}/{len(img_paths)}...")
                    else:
                        if os.path.exists(img_output):
                            os.remove(img_output)
                        failed_count += 1
                else:
                    failed_count += 1

            data_info[split_name] = processed_count
            print(f"Successfully processed: {processed_count}/{len(img_paths)}")
            if failed_count > 0:
                print(f"Failed: {failed_count}")

        self.save_dataset_info(data_info)
        return data_info

    def process_negative_samples(self, negative_images_dirs, split_ratios=(0.7, 0.2, 0.1)):
        """
        Process negative samples (images without cracks)
        Create empty label files for these images
        """
        print(f"\n{'=' * 60}")
        print("PROCESSING NEGATIVE SAMPLES")
        print(f"{'=' * 60}")

        negative_files = []
        for negative_dir in negative_images_dirs:
            if os.path.exists(negative_dir):
                neg_imgs = [f for f in os.listdir(negative_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                print(f"Found {len(neg_imgs)} images in {negative_dir}")
                for img_file in neg_imgs:
                    negative_files.append(os.path.join(negative_dir, img_file))

        print(f"\nTotal negative samples found: {len(negative_files)}")

        if len(negative_files) == 0:
            print("No negative samples found!")
            return {'train': 0, 'val': 0, 'test': 0}

        # Split negative samples
        train_ratio, val_ratio, test_ratio = split_ratios

        train_val_negs, test_negs = train_test_split(
            negative_files, test_size=test_ratio, random_state=42, shuffle=True
        )

        val_size = val_ratio / (train_ratio + val_ratio)
        train_negs, val_negs = train_test_split(
            train_val_negs, test_size=val_size, random_state=42, shuffle=True
        )

        splits = {
            'train': train_negs,
            'val': val_negs,
            'test': test_negs
        }

        negative_counts = {}

        for split_name, neg_imgs in splits.items():
            print(f"\nProcessing {len(neg_imgs)} negative samples for {split_name.upper()}")

            # Get current count to continue numbering
            existing_files = os.listdir(os.path.join(self.output_dir, f'images/{split_name}'))
            current_count = len([f for f in existing_files if f.startswith(f'{split_name}_')])

            processed = 0
            failed = 0

            for i, img_path in enumerate(neg_imgs):
                base_name = f"{split_name}_{current_count + i:05d}"

                img_output = os.path.join(self.output_dir, f'images/{split_name}', f'{base_name}.jpg')
                if self.process_images(img_path, img_output):
                    # Create empty label file (no objects)
                    label_output = os.path.join(self.output_dir, f'labels/{split_name}', f'{base_name}.txt')
                    open(label_output, 'w').close()
                    processed += 1

                    if processed % 100 == 0:
                        print(f"  Processed {processed}/{len(neg_imgs)}...")
                else:
                    failed += 1

            negative_counts[split_name] = processed
            print(f"Successfully processed: {processed}/{len(neg_imgs)}")
            if failed > 0:
                print(f"Failed: {failed}")

        return negative_counts

    def save_dataset_info(self, data_info):
        """Save dataset information for detection"""
        dataset_info = {
            'image_size': self.image_size,
            'num_classes': 1,
            'class_names': ['crack'],
            'splits': data_info,
            'total_samples': sum(data_info.values()),
            'task': 'object_detection',
            'format': 'yolo'
        }

        info_path = os.path.join(self.output_dir, 'annotations', 'dataset_info.json')
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)

        yaml_content = f"""# Crack Detection Dataset
path: {os.path.abspath(self.output_dir)}
train: images/train
val: images/val
test: images/test

# Classes
nc: 1  # number of classes
names: ['crack']  # class names
"""

        yaml_path = os.path.join(self.output_dir, 'data.yaml')
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

        print(f"\nDataset info saved to {info_path}")
        print(f"YAML config saved to {yaml_path}")


def main():
    """Main function to organize your crack detection data"""
    print("\n" + "=" * 60)
    print("CRACK DETECTION DATASET ORGANIZER")
    print("=" * 60)

    SOURCE_DIR = os.environ.get("CRACK_WORK_IMAGES", "D:/work images") + "/swinn/"
    OUTPUT_DIR = "crack_detection_dataset"

    POSITIVE_IMAGES_DIRS = [
        os.path.join(SOURCE_DIR, "positive_batch"),
    ]

    ANNOTATIONS_JSON = "instances_merged.json"

    # Check JSON file
    if not os.path.exists(ANNOTATIONS_JSON):
        print(f"\nError: JSON file '{ANNOTATIONS_JSON}' not found!")
        print("Please run the merge script first to create instances_merged.json")
        return

    # Check image directories
    existing_image_dirs = [d for d in POSITIVE_IMAGES_DIRS if os.path.exists(d)]
    if not existing_image_dirs:
        print("\nError: No image directories found!")
        for d in POSITIVE_IMAGES_DIRS:
            print(f"  - {d}")
        return

    print(f"\nFound {len(existing_image_dirs)} image directory(s)")
    for d in existing_image_dirs:
        img_count = len([f for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"  - {d} ({img_count} images)")

    # Initialize organizer
    organizer = CrackDetectionDataOrganizer(SOURCE_DIR, OUTPUT_DIR)

    # Organize positive samples
    data_info = organizer.organize_detection_data(existing_image_dirs, ANNOTATIONS_JSON)

    if not data_info:
        print("\nDataset organization failed. Please check the errors above.")
        return

    # Process negative samples
    NEGATIVE_IMAGES_DIRS = [
        os.path.join(SOURCE_DIR, "negative_batch"),
    ]

    existing_negative_dirs = [d for d in NEGATIVE_IMAGES_DIRS if os.path.exists(d)]

    if existing_negative_dirs:
        negative_info = organizer.process_negative_samples(existing_negative_dirs)

        # Combine counts
        for split in data_info:
            if split in negative_info:
                data_info[split] += negative_info[split]
    else:
        print("\nWarning: No negative images directories found.")
        print("Expected:", NEGATIVE_IMAGES_DIRS[0])

    # Final summary
    print("\n" + "=" * 60)
    print("DATASET ORGANIZATION COMPLETE!")
    print("=" * 60)
    print(f"Organized dataset saved to: {OUTPUT_DIR}")
    print("\nDataset summary:")
    total_samples = 0
    for split, count in data_info.items():
        percentage = (count / sum(data_info.values())) * 100
        print(f"  {split:5s}: {count:4d} samples ({percentage:.1f}%)")
        total_samples += count
    print(f"  {'Total':5s}: {total_samples:4d} samples")
    print(f"\nFormat: YOLO (normalized bounding boxes)")
    print(f"Classes: 1 (crack)")
    print(f"Image size: 224x224")


if __name__ == "__main__":
    main()