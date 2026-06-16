"""
VISUAL COMPARISON GENERATOR FOR RESEARCH PAPER
Generates publication-quality comparison images
"""

import torch
import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from tqdm import tqdm
import torchvision.transforms as transforms

try:
    from transformers import SwinModel
    from ultralytics import YOLO
except ImportError:
    print("Install required packages")

from common.swin_detection import SwinDetectionModel

# Import your model classes
import torch.nn as nn


from common.swin_detection import SwinDetectionModel


class VisualComparator:
    def __init__(self, detection_data_dir, output_dir='paper_figures'):
        self.detection_data_dir = detection_data_dir
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        os.makedirs(output_dir, exist_ok=True)

        # Color scheme for publication
        self.colors = {
            'swin_bbox': '#FF6B6B',      # Red
            'yolo_bbox': '#4ECDC4',      # Teal
            'gt_bbox': '#95E1D3',        # Light teal
            'swin_mask': (255, 107, 107),  # Red in BGR
            'yolo_mask': (196, 205, 78),   # Teal in BGR
            'gt_mask': (211, 225, 149)     # Light teal in BGR
        }

        self.swin_detector = None
        self.swin_segmentor = None
        self.yolo_detector = None
        self.yolo_segmentor = None

    def load_models(self, swin_det_path, swin_seg_path, yolo_det_path, yolo_seg_path):
        """Load all models"""
        print("📦 Loading models...")

        # Load Swin Detection
        checkpoint = torch.load(swin_det_path, map_location=self.device)
        self.swin_detector = SwinDetectionModel(num_classes=1, hidden_dim=256, max_detections=10)
        if 'model_state_dict' in checkpoint:
            self.swin_detector.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.swin_detector.load_state_dict(checkpoint)
        self.swin_detector = self.swin_detector.to(self.device).eval()

        # Load Swin Segmentation
        from common.swin_seg import SwinUNet
        checkpoint = torch.load(swin_seg_path, map_location=self.device)
        self.swin_segmentor = SwinUNet(img_size=224, num_classes=2, embed_dim=96,
                                       depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7)
        if 'model_state_dict' in checkpoint:
            self.swin_segmentor.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.swin_segmentor.load_state_dict(checkpoint)
        self.swin_segmentor = self.swin_segmentor.to(self.device).eval()

        # Load YOLO models
        self.yolo_detector = YOLO(yolo_det_path)
        self.yolo_segmentor = YOLO(yolo_seg_path)

        print("✅ All models loaded successfully")

    def preprocess_swin(self, image_path):
        """Preprocess for Swin"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return transform(image).unsqueeze(0), image, (orig_h, orig_w)

    def get_swin_predictions(self, image_path, det_conf=0.3):
        """Get Swin predictions"""
        img_tensor, orig_img, orig_size = self.preprocess_swin(image_path)
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            # Detection
            bbox_pred, objectness_probs, _ = self.swin_detector(img_tensor)
            detections = []
            for j in range(self.swin_detector.max_detections):
                confidence = objectness_probs[0, j].item()
                if confidence > det_conf:
                    bbox = bbox_pred[0, j].cpu().numpy()
                    x1, y1, x2, y2 = np.clip(bbox, 0, 1)
                    if (x2 - x1) > 0.05 and (y2 - y1) > 0.05:
                        detections.append({'bbox': [x1, y1, x2, y2], 'confidence': confidence})

            # Segmentation
            seg_output = self.swin_segmentor(img_tensor)
            seg_pred = torch.softmax(seg_output, dim=1)
            seg_mask = torch.argmax(seg_pred, dim=1)[0].cpu().numpy()

        return detections, seg_mask

    def get_yolo_predictions(self, image_path, det_conf=0.25):
        """Get YOLO predictions"""
        # Detection
        det_results = self.yolo_detector.predict(source=image_path, conf=det_conf,
                                                  save=False, imgsz=192, verbose=False)
        detections = []
        if len(det_results) > 0 and det_results[0].boxes is not None:
            boxes = det_results[0].boxes
            orig_img = cv2.imread(image_path)
            img_h, img_w = orig_img.shape[:2]

            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = xyxy / np.array([img_w, img_h, img_w, img_h])
                detections.append({'bbox': [x1, y1, x2, y2], 'confidence': boxes.conf[i].item()})

        # Segmentation
        seg_results = self.yolo_segmentor.predict(source=image_path, conf=det_conf,
                                                   save=False, imgsz=192, verbose=False)
        seg_mask = None
        if len(seg_results) > 0 and hasattr(seg_results[0], 'masks') and seg_results[0].masks is not None:
            masks_data = seg_results[0].masks
            seg_mask = masks_data.data[0].cpu().numpy() if len(masks_data) > 0 else None

        return detections, seg_mask

    def load_ground_truth(self, image_path):
        """Load ground truth boxes and mask"""
        # Load GT boxes - handle both forward and backslashes
        label_path = image_path.replace('\\images\\', '\\labels\\').replace('/images/', '/labels/')
        label_path = label_path.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt')

        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, x_c, y_c, w, h = map(float, parts[:5])
                        x1, y1 = x_c - w/2, y_c - h/2
                        x2, y2 = x_c + w/2, y_c + h/2
                        gt_boxes.append([x1, y1, x2, y2])

        # Load GT mask - handle both forward and backslashes
        mask_path = image_path.replace('\\images\\', '\\masks\\').replace('/images/', '/masks/')
        mask_path = mask_path.replace('.jpg', '.png').replace('.jpeg', '.png')

        gt_mask = None
        if os.path.exists(mask_path):
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                gt_mask = (gt_mask > 127).astype(np.uint8)
                if len(gt_mask.shape) == 3:
                    gt_mask = gt_mask[:, :, 0]

        return gt_boxes, gt_mask

    def create_side_by_side_comparison(self, image_path, save_name):
        """Create side-by-side comparison: GT | Swin | YOLO"""
        # Load original image
        orig_img = cv2.imread(image_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        h, w = orig_img.shape[:2]

        # Get predictions
        gt_boxes, gt_mask = self.load_ground_truth(image_path)
        swin_dets, swin_mask = self.get_swin_predictions(image_path)
        yolo_dets, yolo_mask = self.get_yolo_predictions(image_path)

        # Create figure
        fig = plt.figure(figsize=(18, 6))
        gs = GridSpec(2, 3, hspace=0.3, wspace=0.2)

        # Row 1: Detection with bounding boxes
        for idx, (title, boxes, color) in enumerate([
            ('Ground Truth', gt_boxes, self.colors['gt_bbox']),
            ('Swin Transformer', [(d['bbox'], d['confidence']) for d in swin_dets], self.colors['swin_bbox']),
            ('YOLO', [(d['bbox'], d['confidence']) for d in yolo_dets], self.colors['yolo_bbox'])
        ]):
            ax = fig.add_subplot(gs[0, idx])
            ax.imshow(orig_img)
            ax.set_title(f'{title}\nDetection', fontsize=12, fontweight='bold')
            ax.axis('off')

            # Draw boxes
            if idx == 0:  # Ground truth
                for box in boxes:
                    x1, y1, x2, y2 = [coord * dim for coord, dim in zip(box, [w, h, w, h])]
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2,
                                            edgecolor=color, facecolor='none')
                    ax.add_patch(rect)
            else:  # Predictions
                for box, conf in boxes:
                    x1, y1, x2, y2 = [coord * dim for coord, dim in zip(box, [w, h, w, h])]
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2,
                                            edgecolor=color, facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x1, y1-5, f'{conf:.2f}', color=color, fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        # Row 2: Segmentation masks
        for idx, (title, mask, color_name) in enumerate([
            ('Ground Truth Mask', gt_mask, 'Greens'),
            ('Swin Segmentation', swin_mask, 'Reds'),
            ('YOLO Segmentation', yolo_mask, 'Blues')
        ]):
            ax = fig.add_subplot(gs[1, idx])

            if mask is not None:
                # Resize mask to match original image
                if len(mask.shape) == 3:
                    mask = mask[0] if mask.shape[0] == 1 else mask.max(axis=0)
                mask_resized = cv2.resize(mask.astype(np.float32), (w, h),
                                        interpolation=cv2.INTER_NEAREST)

                # Create overlay
                overlay = orig_img.copy()
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                overlay[mask_binary > 0] = overlay[mask_binary > 0] * 0.4 + \
                                           np.array([255, 0, 0] if 'Swin' in title else
                                                   [0, 0, 255] if 'YOLO' in title else
                                                   [0, 255, 0]) * 0.6

                ax.imshow(overlay.astype(np.uint8))
            else:
                ax.imshow(orig_img)
                ax.text(0.5, 0.5, 'No Mask', transform=ax.transAxes,
                       ha='center', va='center', fontsize=14, color='red')

            ax.set_title(f'{title}\nSegmentation', fontsize=12, fontweight='bold')
            ax.axis('off')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {save_path}")

    def create_performance_charts(self, results_json_path):
        """Create performance comparison charts"""
        with open(results_json_path, 'r') as f:
            results = json.load(f)

        # Extract metrics
        swin = results['swin']['joint']
        yolo = results['yolo']['joint']

        # 1. Detection Performance Bar Chart
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Detection metrics
        det_metrics = ['Precision', 'Recall', 'F1-Score', 'Avg IoU']
        swin_det = [swin['det_precision'], swin['det_recall'], swin['det_f1'], swin['det_avg_iou']]
        yolo_det = [yolo['det_precision'], yolo['det_recall'], yolo['det_f1'], yolo['det_avg_iou']]

        x = np.arange(len(det_metrics))
        width = 0.35

        axes[0, 0].bar(x - width/2, swin_det, width, label='Swin', color='#FF6B6B', alpha=0.8)
        axes[0, 0].bar(x + width/2, yolo_det, width, label='YOLO', color='#4ECDC4', alpha=0.8)
        axes[0, 0].set_xlabel('Metrics', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Score', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Detection Performance', fontsize=13, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(det_metrics)
        axes[0, 0].legend()
        axes[0, 0].set_ylim([0, 1.0])
        axes[0, 0].grid(axis='y', alpha=0.3)

        # Segmentation metrics
        seg_metrics = ['IoU', 'Dice', 'Pixel Acc.']
        swin_seg = [swin['seg_avg_iou'], swin['seg_avg_dice'], swin['seg_avg_pixel_acc']]
        yolo_seg = [yolo['seg_avg_iou'], yolo['seg_avg_dice'], yolo['seg_avg_pixel_acc']]

        x = np.arange(len(seg_metrics))
        axes[0, 1].bar(x - width/2, swin_seg, width, label='Swin', color='#FF6B6B', alpha=0.8)
        axes[0, 1].bar(x + width/2, yolo_seg, width, label='YOLO', color='#4ECDC4', alpha=0.8)
        axes[0, 1].set_xlabel('Metrics', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Score', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Segmentation Performance', fontsize=13, fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(seg_metrics)
        axes[0, 1].legend()
        axes[0, 1].set_ylim([0, 1.0])
        axes[0, 1].grid(axis='y', alpha=0.3)

        # Speed comparison
        speed_metrics = ['Detection\n(ms)', 'Segmentation\n(ms)', 'Total\n(ms)']
        swin_speed = [swin['avg_det_time_ms'], swin['avg_seg_time_ms'], swin['avg_total_time_ms']]
        yolo_speed = [yolo['avg_det_time_ms'], yolo['avg_seg_time_ms'], yolo['avg_total_time_ms']]

        x = np.arange(len(speed_metrics))
        axes[1, 0].bar(x - width/2, swin_speed, width, label='Swin', color='#FF6B6B', alpha=0.8)
        axes[1, 0].bar(x + width/2, yolo_speed, width, label='YOLO', color='#4ECDC4', alpha=0.8)
        axes[1, 0].set_xlabel('Pipeline Stage', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Time (ms)', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Inference Speed (Lower is Better)', fontsize=13, fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(speed_metrics)
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)

        # Overall comparison (radar chart would be better but bar for simplicity)
        overall_metrics = ['Detection\nF1', 'Seg.\nIoU', 'FPS/10', 'Success\nRate']
        swin_overall = [swin['det_f1'], swin['seg_avg_iou'], swin['fps']/10, swin['pipeline_success_rate']]
        yolo_overall = [yolo['det_f1'], yolo['seg_avg_iou'], yolo['fps']/10, yolo['pipeline_success_rate']]

        x = np.arange(len(overall_metrics))
        axes[1, 1].bar(x - width/2, swin_overall, width, label='Swin', color='#FF6B6B', alpha=0.8)
        axes[1, 1].bar(x + width/2, yolo_overall, width, label='YOLO', color='#4ECDC4', alpha=0.8)
        axes[1, 1].set_xlabel('Metrics', fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel('Score', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Overall Performance', fontsize=13, fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(overall_metrics)
        axes[1, 1].legend()
        axes[1, 1].set_ylim([0, 1.0])
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'performance_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {save_path}")

    def generate_multiple_examples(self, num_examples=5):
        """Generate multiple comparison examples"""
        test_images_dir = os.path.join(self.detection_data_dir, 'images', 'test')

        print(f"\n📁 Looking for test images in: {test_images_dir}")
        print(f"   Directory exists: {os.path.exists(test_images_dir)}")

        if not os.path.exists(test_images_dir):
            print(f"❌ Test directory not found!")
            print(f"   Current working directory: {os.getcwd()}")
            print(f"   Detection data dir: {self.detection_data_dir}")
            return

        image_files = [f for f in os.listdir(test_images_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        print(f"   Found {len(image_files)} image files")
        if len(image_files) == 0:
            print("❌ No image files found in test directory!")
            return

        print(f"\n🔍 Searching through {len(image_files)} test images...")

        # Select diverse examples (with ground truth)
        selected = []
        checked = 0
        for img_file in image_files:
            img_path = os.path.join(test_images_dir, img_file)
            gt_boxes, gt_mask = self.load_ground_truth(img_path)

            checked += 1
            if checked % 100 == 0:
                print(f"   Checked {checked} images, found {len(selected)} with GT...")

            # More relaxed criteria: just need GT boxes OR mask
            if len(gt_boxes) > 0:
                selected.append(img_path)
                if len(selected) >= num_examples:
                    break

        if len(selected) == 0:
            print("❌ No images found with ground truth!")
            print(f"   Test images dir: {test_images_dir}")
            print(f"   Sample paths checked:")
            if len(image_files) > 0:
                sample_path = os.path.join(test_images_dir, image_files[0])
                print(f"   - Image: {sample_path}")
                print(f"   - Exists: {os.path.exists(sample_path)}")

                label_path = sample_path.replace('\\images\\', '\\labels\\').replace('/images/', '/labels/')
                label_path = label_path.replace('.png', '.txt').replace('.jpg', '.txt')
                print(f"   - Label: {label_path}")
                print(f"   - Exists: {os.path.exists(label_path)}")

                mask_path = sample_path.replace('\\images\\', '\\masks\\').replace('/images/', '/masks/')
                mask_path = mask_path.replace('.jpg', '.png').replace('.jpeg', '.png')
                print(f"   - Mask: {mask_path}")
                print(f"   - Exists: {os.path.exists(mask_path)}")
            return

        print(f"\n🎨 Generating {len(selected)} comparison examples...")
        for i, img_path in enumerate(tqdm(selected, desc="Creating comparisons")):
            try:
                self.create_side_by_side_comparison(
                    img_path,
                    f'comparison_example_{i+1}.png'
                )
            except Exception as e:
                print(f"\n❌ Error processing {img_path}: {e}")
                import traceback
                traceback.print_exc()


def main():
    print("=" * 100)
    print("VISUAL COMPARISON GENERATOR FOR RESEARCH PAPER")
    print("=" * 100)

    # Configuration
    DETECTION_DATA_DIR = "unified_crack_dataset"
    SWIN_DETECTION = "best_swin_crack_detection.pth"
    SWIN_SEGMENTATION = "best_swin_crack_segmentation.pth"
    YOLO_DETECTION = "yolo12s_cbam_ca_crack.pt"
    YOLO_SEGMENTATION = "yolo12s_seg_cbam_ca_crack.pt"
    RESULTS_JSON = "research_results/comparison_results.json"

    # Initialize comparator
    comparator = VisualComparator(DETECTION_DATA_DIR)

    # Load models
    comparator.load_models(SWIN_DETECTION, SWIN_SEGMENTATION,
                          YOLO_DETECTION, YOLO_SEGMENTATION)

    # Generate visualizations
    print("\n" + "=" * 100)
    print("GENERATING VISUALIZATIONS")
    print("=" * 100)

    # 1. Performance charts
    print("\n📊 Creating performance comparison charts...")
    comparator.create_performance_charts(RESULTS_JSON)

    # 2. Side-by-side examples
    comparator.generate_multiple_examples(num_examples=20)

    print("\n" + "=" * 100)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("=" * 100)
    print(f"\n📁 Results saved to: {comparator.output_dir}/")
    print("=" * 100)


if __name__ == "__main__":
    main()