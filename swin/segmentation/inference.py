import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import os
import glob
from pathlib import Path

# Import your model architecture
from common.swin_seg import SwinUNet


class CrackSegmenter:
    def __init__(self, model_path, device='cuda'):
        """
        Initialize the crack segmentation model

        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load model configuration from checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model_config = checkpoint['model_config']

        # Initialize model with the same configuration as training
        self.model = SwinUNet(
            img_size=model_config['img_size'],
            num_classes=model_config['num_classes'],
            embed_dim=model_config['embed_dim'],
            depths=model_config['depths'],
            num_heads=model_config['num_heads'],
            window_size=model_config['window_size']
        )

        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Image transformation (same as validation)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((model_config['img_size'], model_config['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.image_size = model_config['img_size']
        self.num_classes = model_config['num_classes']

        print(f"Model loaded from {model_path}")
        print(f"Input size: {self.image_size}x{self.image_size}")
        print(f"Number of classes: {self.num_classes}")

    def predict(self, image_path, output_path=None, confidence_threshold=0.5):
        """
        Perform segmentation on a single image

        Args:
            image_path: Path to input image
            output_path: Path to save output visualization (optional)
            confidence_threshold: Threshold for binary classification

        Returns:
            segmentation_mask: Binary segmentation mask
            overlay: Original image with segmentation overlay
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = original_image.shape[:2]

        # Preprocess for model
        input_tensor = self.transform(original_image).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predictions = torch.argmax(probabilities, dim=1).squeeze(0).cpu().numpy()

            # Get confidence scores
            confidence = torch.max(probabilities, dim=1)[0].squeeze(0).cpu().numpy()

        # Resize to original dimensions
        segmentation_mask = cv2.resize(
            predictions.astype(np.uint8),
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # Resize confidence map
        confidence_map = cv2.resize(
            confidence,
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # Apply confidence threshold
        if confidence_threshold > 0:
            segmentation_mask[confidence_map < confidence_threshold] = 0

        # Create overlay visualization
        overlay = self.create_overlay(original_image, segmentation_mask)

        # Save results if requested
        if output_path:
            self.save_results(original_image, segmentation_mask, overlay, output_path, image_path)

        return segmentation_mask, overlay

    def create_overlay(self, image, mask):
        """
        Create an overlay of the segmentation on the original image

        Args:
            image: Original RGB image
            mask: Segmentation mask

        Returns:
            overlay: Image with segmentation overlay
        """
        # Create a color mask (red for cracks)
        color_mask = np.zeros_like(image)
        color_mask[mask == 1] = [255, 0, 0]  # Red for cracks

        # Blend with original image
        alpha = 0.5
        overlay = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)

        return overlay

    def save_results(self, original_image, segmentation_mask, overlay, output_path, image_path):
        """
        Save the segmentation results

        Args:
            original_image: Original input image
            segmentation_mask: Binary segmentation mask
            overlay: Overlay visualization
            output_path: Directory to save results
            image_path: Path to original image (for naming)
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        # Get base filename
        filename = os.path.splitext(os.path.basename(image_path))[0]

        # Save original image
        cv2.imwrite(os.path.join(output_path, f"{filename}_original.png"),
                    cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))

        # Save segmentation mask
        cv2.imwrite(os.path.join(output_path, f"{filename}_mask.png"),
                    segmentation_mask * 255)  # Scale to 0-255

        # Save overlay
        cv2.imwrite(os.path.join(output_path, f"{filename}_overlay.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # Save side-by-side comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(segmentation_mask, cmap='gray')
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title('Overlay (Cracks in Red)')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"{filename}_comparison.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Results saved to {output_path}/{filename}_*.png")

    def process_folder(self, input_folder, output_folder, confidence_threshold=0.5):
        """
        Process all images in a folder

        Args:
            input_folder: Folder containing input images
            output_folder: Folder to save results
            confidence_threshold: Threshold for binary classification
        """
        # Supported image extensions
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

        # Get all image files
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(input_folder, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(input_folder, f"*{ext.upper()}")))

        print(f"Found {len(image_files)} images in {input_folder}")

        # Process each image
        for i, image_file in enumerate(image_files):
            print(f"Processing {i + 1}/{len(image_files)}: {os.path.basename(image_file)}")
            try:
                self.predict(image_file, output_folder, confidence_threshold)
            except Exception as e:
                print(f"Error processing {image_file}: {e}")


def main():
    # HARDCODED PARAMETERS - MODIFY THESE VALUES
    MODEL_PATH = "best_swin_crack_segmentation.pth"  # Path to your model
    INPUT_PATH = "test/0209.png"  # Path to single image, folder, or pattern
    OUTPUT_PATH = "./results"  # Output directory
    CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold (0-1)
    DEVICE = "cuda"  # Device: 'cuda' or 'cpu'
    DISPLAY_RESULTS = True  # Set to False to not display results (just save them)

    # Initialize segmenter
    segmenter = CrackSegmenter(MODEL_PATH, DEVICE)

    # Check if input is a file, folder, or pattern
    if os.path.isfile(INPUT_PATH):
        # Process single image
        segmentation_mask, overlay = segmenter.predict(
            INPUT_PATH, OUTPUT_PATH, CONFIDENCE_THRESHOLD
        )

        # Display results unless disabled
        if DISPLAY_RESULTS:
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(cv2.imread(INPUT_PATH), cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(segmentation_mask, cmap='gray')
            plt.title('Segmentation Mask')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title('Overlay (Cracks in Red)')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

    elif os.path.isdir(INPUT_PATH):
        # Process all images in folder
        segmenter.process_folder(INPUT_PATH, OUTPUT_PATH, CONFIDENCE_THRESHOLD)
    else:
        # Handle patterns like "images/*.jpg"
        if '*' in INPUT_PATH:
            # Extract folder and pattern
            folder = os.path.dirname(INPUT_PATH) or '.'
            pattern = os.path.basename(INPUT_PATH)

            # Get all matching files
            image_files = glob.glob(INPUT_PATH)

            if not image_files:
                print(f"No files found matching pattern: {INPUT_PATH}")
                return

            # Create output subfolder for pattern
            pattern_name = pattern.replace('*', 'all').replace('.', '_')
            output_folder = os.path.join(OUTPUT_PATH, pattern_name)

            # Process each file
            for i, image_file in enumerate(image_files):
                print(f"Processing {i + 1}/{len(image_files)}: {os.path.basename(image_file)}")
                try:
                    segmenter.predict(image_file, output_folder, CONFIDENCE_THRESHOLD)
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
        else:
            raise ValueError(f"Input path {INPUT_PATH} does not exist")


if __name__ == "__main__":
    main()