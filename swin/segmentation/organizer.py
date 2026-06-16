import os
import shutil
import numpy as np
from PIL import Image
import json
from sklearn.model_selection import train_test_split
import cv2


class CrackDataOrganizer:
    def __init__(self, source_dir, output_dir):
        """
        Initialize the data organizer

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
        """Create the required directory structure for training"""
        directories = [
            'images/train',
            'images/val',
            'images/test',
            'masks/train',
            'masks/val',
            'masks/test',
            'annotations'
        ]

        for dir_path in directories:
            full_path = os.path.join(self.output_dir, dir_path)
            os.makedirs(full_path, exist_ok=True)

    def process_segmentation_masks(self, mask_path, output_path):
        """
        Process segmentation masks to ensure correct format
        Convert to binary masks where crack=1, background=0
        """
        try:
            # Read mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Could not read mask {mask_path}")
                return False

            # Resize to target size
            mask = cv2.resize(mask, (self.image_size, self.image_size))

            # Convert to binary (crack=1, background=0)
            # Assuming non-zero pixels are cracks
            binary_mask = (mask > 0).astype(np.uint8)

            # Save processed mask
            cv2.imwrite(output_path, binary_mask * 255)  # Save as 0-255 for visualization
            return True

        except Exception as e:
            print(f"Error processing mask {mask_path}: {e}")
            return False

    def process_images(self, image_path, output_path):
        """
        Process and resize images
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                return False

            # Resize to target size
            image = cv2.resize(image, (self.image_size, self.image_size))

            # Save processed image
            cv2.imwrite(output_path, image)
            return True

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return False

    def organize_data(self, images_dir, masks_dir, split_ratios=(0.7, 0.2, 0.1)):
        """
        Organize data into train/val/test splits

        Args:
            images_dir: Directory containing original images
            masks_dir: Directory containing segmentation masks
            split_ratios: (train, val, test) ratios
        """
        print("Starting data organization...")

        # Get all image files
        image_files = []
        mask_files = []

        # Assuming images and masks have corresponding names
        for img_file in os.listdir(images_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(images_dir, img_file)

                # Find corresponding mask
                mask_name = os.path.splitext(img_file)[0] + '.png'  # Adjust extension as needed
                mask_path = os.path.join(masks_dir, mask_name)

                if os.path.exists(mask_path):
                    image_files.append(img_path)
                    mask_files.append(mask_path)
                else:
                    print(f"Warning: No mask found for {img_file}")

        print(f"Found {len(image_files)} image-mask pairs")

        # Split data
        train_ratio, val_ratio, test_ratio = split_ratios

        # First split: train+val vs test
        train_val_imgs, test_imgs, train_val_masks, test_masks = train_test_split(
            image_files, mask_files, test_size=test_ratio, random_state=42
        )

        # Second split: train vs val
        val_size = val_ratio / (train_ratio + val_ratio)
        train_imgs, val_imgs, train_masks, val_masks = train_test_split(
            train_val_imgs, train_val_masks, test_size=val_size, random_state=42
        )

        # Process and copy files
        splits = {
            'train': (train_imgs, train_masks),
            'val': (val_imgs, val_masks),
            'test': (test_imgs, test_masks)
        }

        data_info = {}

        for split_name, (imgs, masks) in splits.items():
            print(f"\nProcessing {split_name} split: {len(imgs)} samples")

            processed_count = 0
            for img_path, mask_path in zip(imgs, masks):
                # Generate new filename
                base_name = f"{split_name}_{processed_count:04d}"

                # Process and save image
                img_output = os.path.join(self.output_dir, f'images/{split_name}', f'{base_name}.jpg')
                if self.process_images(img_path, img_output):

                    # Process and save mask
                    mask_output = os.path.join(self.output_dir, f'masks/{split_name}', f'{base_name}.png')
                    if self.process_segmentation_masks(mask_path, mask_output):
                        processed_count += 1
                    else:
                        # Remove image if mask processing failed
                        if os.path.exists(img_output):
                            os.remove(img_output)

            data_info[split_name] = processed_count
            print(f"Successfully processed {processed_count} samples for {split_name}")

        # Save dataset info
        self.save_dataset_info(data_info)
        print(f"\nData organization complete! Dataset info saved.")
        return data_info

    def create_negative_samples(self, negative_images_dir, split_ratios=(0.7, 0.2, 0.1)):
        """
        Process negative samples (images without cracks)
        Create empty masks for these images
        """
        print("Processing negative samples...")

        negative_files = []
        for img_file in os.listdir(negative_images_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                negative_files.append(os.path.join(negative_images_dir, img_file))

        print(f"Found {len(negative_files)} negative samples")

        # Split negative samples
        train_ratio, val_ratio, test_ratio = split_ratios

        train_val_negs, test_negs = train_test_split(
            negative_files, test_size=test_ratio, random_state=42
        )

        val_size = val_ratio / (train_ratio + val_ratio)
        train_negs, val_negs = train_test_split(
            train_val_negs, test_size=val_size, random_state=42
        )

        splits = {
            'train': train_negs,
            'val': val_negs,
            'test': test_negs
        }

        for split_name, neg_imgs in splits.items():
            print(f"Processing {len(neg_imgs)} negative samples for {split_name}")

            for i, img_path in enumerate(neg_imgs):
                base_name = f"{split_name}_neg_{i:04d}"

                # Process and save image
                img_output = os.path.join(self.output_dir, f'images/{split_name}', f'{base_name}.jpg')
                if self.process_images(img_path, img_output):
                    # Create empty mask (all zeros)
                    mask_output = os.path.join(self.output_dir, f'masks/{split_name}', f'{base_name}.png')
                    empty_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
                    cv2.imwrite(mask_output, empty_mask)

    def save_dataset_info(self, data_info):
        """Save dataset information"""
        dataset_info = {
            'image_size': self.image_size,
            'num_classes': 2,  # background + crack
            'class_names': ['background', 'crack'],
            'class_colors': [(0, 0, 0), (165, 236, 223)],  # From your labelmap
            'splits': data_info,
            'total_samples': sum(data_info.values())
        }

        with open(os.path.join(self.output_dir, 'annotations', 'dataset_info.json'), 'w') as f:
            json.dump(dataset_info, f, indent=2)


def main():
    """
    Main function to organize your crack segmentation data

    Update these paths according to your data structure:
    """

    # Update these paths according to your data structure
    SOURCE_DIR = os.environ.get("CRACK_WORK_IMAGES", "D:/work images") + "/swinn/"
    OUTPUT_DIR = "crack_segmentation_dataset"  # Output organized dataset

    # Subdirectories in your source data (update as needed)
    POSITIVE_IMAGES_DIR = os.path.join(SOURCE_DIR, "positive_batch")  # Images with cracks
    NEGATIVE_IMAGES_DIR = os.path.join(SOURCE_DIR, "negative_batch")  # Images without cracks
    SEGMENTATION_MASKS_DIR = os.path.join(SOURCE_DIR, "SegmentationClass")  # Your mask directory

    # Initialize organizer
    organizer = CrackDataOrganizer(SOURCE_DIR, OUTPUT_DIR)

    # Organize positive samples (with cracks and masks)
    if os.path.exists(POSITIVE_IMAGES_DIR) and os.path.exists(SEGMENTATION_MASKS_DIR):
        data_info = organizer.organize_data(POSITIVE_IMAGES_DIR, SEGMENTATION_MASKS_DIR)
    else:
        print("Error: Positive images or segmentation masks directory not found!")
        print(f"Please check: {POSITIVE_IMAGES_DIR}")
        print(f"Please check: {SEGMENTATION_MASKS_DIR}")
        return

    # Process negative samples if available
    if os.path.exists(NEGATIVE_IMAGES_DIR):
        organizer.create_negative_samples(NEGATIVE_IMAGES_DIR)
        print("Negative samples processed successfully!")
    else:
        print("Warning: Negative images directory not found. Skipping negative samples.")

    print("\n" + "=" * 50)
    print("DATA ORGANIZATION COMPLETE!")
    print("=" * 50)
    print(f"Organized dataset saved to: {OUTPUT_DIR}")
    print("\nDirectory structure created:")
    print("├── images/")
    print("│   ├── train/")
    print("│   ├── val/")
    print("│   └── test/")
    print("├── masks/")
    print("│   ├── train/")
    print("│   ├── val/")
    print("│   └── test/")
    print("└── annotations/")
    print("    └── dataset_info.json")

    print(f"\nDataset summary:")
    for split, count in data_info.items():
        print(f"  {split}: {count} samples")


if __name__ == "__main__":
    main()