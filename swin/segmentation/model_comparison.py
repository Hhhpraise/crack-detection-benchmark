"""
JOINT DETECTION + SEGMENTATION COMPARISON
Two-stage pipeline: Detection → Segmentation
Compares Swin Transformer vs Custom YOLO
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import torchvision.transforms as transforms
from sklearn.metrics import precision_recall_curve, auc
import seaborn as sns

try:
    from transformers import SwinConfig, SwinModel
except ImportError:
    print("Install transformers: pip install transformers")
    exit()

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    print("Install ultralytics: pip install ultralytics")
    YOLO_AVAILABLE = False

from common.swin_detection import SwinDetectionModel
from common.swin_seg import SwinUNet


class JointModelComparator:
    def __init__(self, data_dir, image_size=224):
        self.data_dir = data_dir
        self.image_size = image_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        with open(os.path.join(data_dir, 'annotations', 'dataset_info.json'), 'r') as f:
            self.dataset_info = json.load(f)

        self.swin_detector = None
        self.swin_segmentor = None
        self.yolo_detector = None
        self.yolo_segmentor = None

        self.results = {
            'swin': {'detection': {}, 'segmentation': {}, 'joint': {}},
            'yolo': {'detection': {}, 'segmentation': {}, 'joint': {}}
        }

    # ==================== MODEL LOADING ====================
    def load_swin_models(self, detection_path, segmentation_path):
        """Load both Swin models"""
        print("\n📦 Loading Swin Transformer Models...")

        # Load detection model
        try:
            checkpoint = torch.load(detection_path, map_location=self.device)
            self.swin_detector = SwinDetectionModel(num_classes=1, hidden_dim=256, max_detections=10)

            if 'model_state_dict' in checkpoint:
                self.swin_detector.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.swin_detector.load_state_dict(checkpoint)

            self.swin_detector = self.swin_detector.to(self.device).eval()
            print("✅ Swin Detection model loaded")
        except Exception as e:
            print(f"❌ Failed to load Swin detection: {e}")
            return False

        # Load segmentation model
        try:
            # Import the full SwinUNet from common
            from common.swin_seg import SwinUNet as FullSwinUNet

            checkpoint = torch.load(segmentation_path, map_location=self.device)
            self.swin_segmentor = FullSwinUNet(
                img_size=self.image_size,
                num_classes=2,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7
            )

            if 'model_state_dict' in checkpoint:
                self.swin_segmentor.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.swin_segmentor.load_state_dict(checkpoint)

            self.swin_segmentor = self.swin_segmentor.to(self.device).eval()
            print("✅ Swin Segmentation model loaded")
        except Exception as e:
            print(f"❌ Failed to load Swin segmentation: {e}")
            return False

        return True

    def load_yolo_models(self, detection_path, segmentation_path):
        """Load both YOLO models"""
        if not YOLO_AVAILABLE:
            return False

        print("\n📦 Loading YOLO Models...")

        try:
            self.yolo_detector = YOLO(detection_path)
            print("✅ YOLO Detection model loaded")
        except Exception as e:
            print(f"❌ Failed to load YOLO detection: {e}")
            return False

        try:
            self.yolo_segmentor = YOLO(segmentation_path)
            print("✅ YOLO Segmentation model loaded")
        except Exception as e:
            print(f"❌ Failed to load YOLO segmentation: {e}")
            return False

        return True

    # ==================== INFERENCE FUNCTIONS ====================
    def preprocess_swin(self, image_path):
        """Preprocess for Swin models"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return transform(image).unsqueeze(0), image, (orig_h, orig_w)

    def swin_joint_inference(self, image_path, det_conf=0.3):
        """Two-stage Swin: Detection → Segmentation"""
        try:
            img_tensor, orig_img, orig_size = self.preprocess_swin(image_path)
            img_tensor = img_tensor.to(self.device)

            # Stage 1: Detection
            start_time = time.time()
            with torch.no_grad():
                bbox_pred, objectness_probs, _ = self.swin_detector(img_tensor)
            det_time = time.time() - start_time

            # Get detections
            detections = []
            for j in range(self.swin_detector.max_detections):
                confidence = objectness_probs[0, j].item()
                if confidence > det_conf:
                    bbox = bbox_pred[0, j].cpu().numpy()
                    x1, y1, x2, y2 = np.clip(bbox, 0, 1)

                    if (x2 - x1) > 0.05 and (y2 - y1) > 0.05:
                        detections.append({
                            'bbox': np.array([x1, y1, x2, y2]),
                            'confidence': confidence
                        })

            # Stage 2: Segmentation on detected regions
            start_time = time.time()
            all_masks = []

            if len(detections) > 0:
                with torch.no_grad():
                    seg_output = self.swin_segmentor(img_tensor)
                    seg_pred = torch.softmax(seg_output, dim=1)
                    seg_mask = torch.argmax(seg_pred, dim=1)[0].cpu().numpy()

                # Extract masks for each detection
                for det in detections:
                    bbox = det['bbox']
                    x1 = int(bbox[0] * self.image_size)
                    y1 = int(bbox[1] * self.image_size)
                    x2 = int(bbox[2] * self.image_size)
                    y2 = int(bbox[3] * self.image_size)

                    roi_mask = seg_mask[y1:y2, x1:x2]
                    all_masks.append({
                        'bbox': bbox,
                        'mask': roi_mask,
                        'confidence': det['confidence']
                    })

            seg_time = time.time() - start_time

            return {
                'detections': detections,
                'masks': all_masks,
                'det_time': det_time,
                'seg_time': seg_time,
                'total_time': det_time + seg_time
            }

        except Exception as e:
            print(f"Error in Swin joint inference: {e}")
            return {
                'detections': [],
                'masks': [],
                'det_time': 0,
                'seg_time': 0,
                'total_time': 0
            }

    def yolo_joint_inference(self, image_path, det_conf=0.25):
        """Two-stage YOLO: Detection → Segmentation"""
        try:
            # Stage 1: Detection
            start_time = time.time()
            det_results = self.yolo_detector.predict(
                source=image_path,
                conf=det_conf,
                save=False,
                imgsz=192,
                verbose=False
            )
            det_time = time.time() - start_time

            detections = []
            if len(det_results) > 0 and det_results[0].boxes is not None:
                boxes = det_results[0].boxes
                orig_img = cv2.imread(image_path)
                img_h, img_w = orig_img.shape[:2]

                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = xyxy / np.array([img_w, img_h, img_w, img_h])

                    detections.append({
                        'bbox': np.array([x1, y1, x2, y2]),
                        'confidence': boxes.conf[i].item()
                    })

            # Stage 2: Segmentation
            start_time = time.time()
            seg_results = self.yolo_segmentor.predict(
                source=image_path,
                conf=det_conf,
                save=False,
                imgsz=192,
                verbose=False
            )
            seg_time = time.time() - start_time

            all_masks = []
            if len(seg_results) > 0 and hasattr(seg_results[0], 'masks') and seg_results[0].masks is not None:
                masks_data = seg_results[0].masks
                boxes = seg_results[0].boxes

                for i in range(len(masks_data)):
                    mask = masks_data.data[i].cpu().numpy()
                    bbox = boxes.xyxy[i].cpu().numpy()

                    orig_img = cv2.imread(image_path)
                    img_h, img_w = orig_img.shape[:2]
                    x1, y1, x2, y2 = bbox / np.array([img_w, img_h, img_w, img_h])

                    all_masks.append({
                        'bbox': np.array([x1, y1, x2, y2]),
                        'mask': mask,
                        'confidence': boxes.conf[i].item()
                    })

            return {
                'detections': detections,
                'masks': all_masks,
                'det_time': det_time,
                'seg_time': seg_time,
                'total_time': det_time + seg_time
            }

        except Exception as e:
            print(f"Error in YOLO joint inference: {e}")
            return {
                'detections': [],
                'masks': [],
                'det_time': 0,
                'seg_time': 0,
                'total_time': 0
            }

    # ==================== EVALUATION METRICS ====================
    def calculate_iou_bbox(self, box1, box2):
        """Calculate IoU for bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def calculate_iou_mask(self, mask1, mask2):
        """Calculate IoU for segmentation masks"""
        if mask1.size == 0 or mask2.size == 0:
            return 0.0

        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        return intersection / union if union > 0 else 0.0

    def calculate_dice_score(self, mask1, mask2):
        """Calculate Dice score for segmentation"""
        if mask1.size == 0 or mask2.size == 0:
            return 0.0

        intersection = np.logical_and(mask1, mask2).sum()
        return (2.0 * intersection) / (mask1.sum() + mask2.sum()) if (mask1.sum() + mask2.sum()) > 0 else 0.0

    def calculate_pixel_accuracy(self, pred_mask, gt_mask):
        """Calculate pixel-wise accuracy"""
        if pred_mask.size == 0 or gt_mask.size == 0:
            return 0.0

        correct = (pred_mask == gt_mask).sum()
        total = gt_mask.size
        return correct / total

    # ==================== COMPREHENSIVE EVALUATION ====================
    def evaluate_joint_pipeline(self, model_type='swin', det_conf=0.3, iou_threshold=0.5):
        """Evaluate complete detection + segmentation pipeline"""
        test_images_dir = os.path.join(self.data_dir, 'images/test')
        test_labels_dir = os.path.join(self.data_dir, 'labels/test')
        test_masks_dir = os.path.join(self.data_dir, 'masks/test')

        image_files = [f for f in os.listdir(test_images_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        metrics = {
            # Detection metrics
            'det_tp': 0, 'det_fp': 0, 'det_fn': 0,
            'det_ious': [],
            'det_times': [],

            # Segmentation metrics
            'seg_ious': [],
            'dice_scores': [],
            'pixel_accuracies': [],
            'seg_times': [],

            # Joint metrics
            'total_times': [],
            'successful_pipelines': 0,
            'total_images': len(image_files)
        }

        print(f"\n🔍 Evaluating {model_type.upper()} joint pipeline...")

        for img_file in tqdm(image_files, desc=f"Testing {model_type}"):
            img_path = os.path.join(test_images_dir, img_file)
            label_path = os.path.join(test_labels_dir, os.path.splitext(img_file)[0] + '.txt')
            mask_path = os.path.join(test_masks_dir, os.path.splitext(img_file)[0] + '.png')

            # Load ground truth boxes
            gt_boxes = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls, x_c, y_c, w, h = map(float, parts[:5])
                            x1 = x_c - w / 2
                            y1 = y_c - h / 2
                            x2 = x_c + w / 2
                            y2 = y_c + h / 2
                            gt_boxes.append([x1, y1, x2, y2])

            # Load ground truth mask
            gt_mask = None
            if os.path.exists(mask_path):
                gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                gt_mask = (gt_mask > 127).astype(np.uint8)

            # Run inference
            if model_type == 'swin':
                results = self.swin_joint_inference(img_path, det_conf)
            else:
                results = self.yolo_joint_inference(img_path, det_conf)

            # Evaluate detection
            metrics['det_times'].append(results['det_time'])
            metrics['seg_times'].append(results['seg_time'])
            metrics['total_times'].append(results['total_time'])

            pred_boxes = [det['bbox'] for det in results['detections']]

            # Match detections to ground truth
            matched_gt = set()
            for pred_box in pred_boxes:
                best_iou = 0
                best_gt_idx = -1

                for j, gt_box in enumerate(gt_boxes):
                    if j not in matched_gt:
                        iou = self.calculate_iou_bbox(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = j

                if best_iou >= iou_threshold:
                    metrics['det_tp'] += 1
                    metrics['det_ious'].append(best_iou)
                    matched_gt.add(best_gt_idx)
                else:
                    metrics['det_fp'] += 1

            metrics['det_fn'] += len(gt_boxes) - len(matched_gt)

            # Evaluate segmentation
            if gt_mask is not None and len(results['masks']) > 0:
                # Combine all predicted masks
                combined_pred_mask = np.zeros_like(gt_mask)
                for mask_info in results['masks']:
                    mask = mask_info['mask']
                    if mask.size > 0:
                        # Resize mask to match ground truth size
                        mask_resized = cv2.resize(mask.astype(np.uint8),
                                                  (gt_mask.shape[1], gt_mask.shape[0]),
                                                  interpolation=cv2.INTER_NEAREST)
                        combined_pred_mask = np.maximum(combined_pred_mask, mask_resized)

                # Calculate segmentation metrics
                seg_iou = self.calculate_iou_mask(combined_pred_mask, gt_mask)
                dice = self.calculate_dice_score(combined_pred_mask, gt_mask)
                pixel_acc = self.calculate_pixel_accuracy(combined_pred_mask, gt_mask)

                metrics['seg_ious'].append(seg_iou)
                metrics['dice_scores'].append(dice)
                metrics['pixel_accuracies'].append(pixel_acc)

                if len(results['detections']) > 0 and seg_iou > 0:
                    metrics['successful_pipelines'] += 1

        # Calculate final metrics
        tp, fp, fn = metrics['det_tp'], metrics['det_fp'], metrics['det_fn']

        final_metrics = {
            # Detection Performance
            'det_precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'det_recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'det_f1': 0,
            'det_avg_iou': np.mean(metrics['det_ious']) if metrics['det_ious'] else 0,

            # Segmentation Performance
            'seg_avg_iou': np.mean(metrics['seg_ious']) if metrics['seg_ious'] else 0,
            'seg_avg_dice': np.mean(metrics['dice_scores']) if metrics['dice_scores'] else 0,
            'seg_avg_pixel_acc': np.mean(metrics['pixel_accuracies']) if metrics['pixel_accuracies'] else 0,

            # Speed Performance
            'avg_det_time_ms': np.mean(metrics['det_times']) * 1000,
            'avg_seg_time_ms': np.mean(metrics['seg_times']) * 1000,
            'avg_total_time_ms': np.mean(metrics['total_times']) * 1000,
            'fps': 1.0 / np.mean(metrics['total_times']) if np.mean(metrics['total_times']) > 0 else 0,

            # Pipeline Success
            'pipeline_success_rate': metrics['successful_pipelines'] / metrics['total_images'],
            'total_images': metrics['total_images']
        }

        # Calculate F1 score
        if (final_metrics['det_precision'] + final_metrics['det_recall']) > 0:
            final_metrics['det_f1'] = (2 * final_metrics['det_precision'] * final_metrics['det_recall']) / \
                                      (final_metrics['det_precision'] + final_metrics['det_recall'])

        return final_metrics

    # ==================== VISUALIZATION & REPORTING ====================
    def print_comparison_table(self):
        """Print comprehensive comparison table"""
        print("\n" + "=" * 120)
        print("JOINT DETECTION + SEGMENTATION COMPARISON")
        print("=" * 120)

        print(f"\n{'Metric':<30} {'Swin Transformer':<25} {'Custom YOLO':<25} {'Winner':<20}")
        print("-" * 120)

        # Detection metrics
        print("\n📍 DETECTION PERFORMANCE:")
        det_metrics = ['det_precision', 'det_recall', 'det_f1', 'det_avg_iou']
        for metric in det_metrics:
            swin_val = self.results['swin']['joint'].get(metric, 0)
            yolo_val = self.results['yolo']['joint'].get(metric, 0)
            winner = "YOLO" if yolo_val > swin_val else "Swin" if swin_val > yolo_val else "Tie"
            print(f"  {metric:<28} {swin_val:<25.4f} {yolo_val:<25.4f} {winner:<20}")

        # Segmentation metrics
        print("\n🎨 SEGMENTATION PERFORMANCE:")
        seg_metrics = ['seg_avg_iou', 'seg_avg_dice', 'seg_avg_pixel_acc']
        for metric in seg_metrics:
            swin_val = self.results['swin']['joint'].get(metric, 0)
            yolo_val = self.results['yolo']['joint'].get(metric, 0)
            winner = "YOLO" if yolo_val > swin_val else "Swin" if swin_val > yolo_val else "Tie"
            print(f"  {metric:<28} {swin_val:<25.4f} {yolo_val:<25.4f} {winner:<20}")

        # Speed metrics
        print("\n⚡ SPEED PERFORMANCE:")
        speed_metrics = ['avg_det_time_ms', 'avg_seg_time_ms', 'avg_total_time_ms', 'fps']
        for metric in speed_metrics:
            swin_val = self.results['swin']['joint'].get(metric, 0)
            yolo_val = self.results['yolo']['joint'].get(metric, 0)
            # For speed, lower is better (except FPS)
            if metric == 'fps':
                winner = "YOLO" if yolo_val > swin_val else "Swin" if swin_val > yolo_val else "Tie"
            else:
                winner = "YOLO" if yolo_val < swin_val else "Swin" if swin_val < yolo_val else "Tie"
            print(f"  {metric:<28} {swin_val:<25.2f} {yolo_val:<25.2f} {winner:<20}")

        # Pipeline success
        print("\n✅ PIPELINE RELIABILITY:")
        swin_success = self.results['swin']['joint'].get('pipeline_success_rate', 0)
        yolo_success = self.results['yolo']['joint'].get('pipeline_success_rate', 0)
        winner = "YOLO" if yolo_success > swin_success else "Swin" if swin_success > yolo_success else "Tie"
        print(f"  {'Pipeline Success Rate':<28} {swin_success:<25.4f} {yolo_success:<25.4f} {winner:<20}")

        print("=" * 120)

    def generate_research_plots(self, save_dir='research_results'):
        """Generate publication-quality plots for research paper"""
        os.makedirs(save_dir, exist_ok=True)

        # Set publication style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")

        # 1. Performance Comparison Radar Chart
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

        categories = ['Detection\nPrecision', 'Detection\nRecall', 'Detection\nF1',
                      'Segmentation\nIoU', 'Segmentation\nDice', 'Speed\n(FPS)']

        swin_values = [
            self.results['swin']['joint']['det_precision'],
            self.results['swin']['joint']['det_recall'],
            self.results['swin']['joint']['det_f1'],
            self.results['swin']['joint']['seg_avg_iou'],
            self.results['swin']['joint']['seg_avg_dice'],
            self.results['swin']['joint']['fps'] / 30  # Normalize FPS
        ]

        yolo_values = [
            self.results['yolo']['joint']['det_precision'],
            self.results['yolo']['joint']['det_recall'],
            self.results['yolo']['joint']['det_f1'],
            self.results['yolo']['joint']['seg_avg_iou'],
            self.results['yolo']['joint']['seg_avg_dice'],
            self.results['yolo']['joint']['fps'] / 30
        ]

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        swin_values += swin_values[:1]
        yolo_values += yolo_values[:1]
        angles += angles[:1]

        ax.plot(angles, swin_values, 'o-', linewidth=2, label='Swin Transformer', color='#2E86AB')
        ax.fill(angles, swin_values, alpha=0.25, color='#2E86AB')
        ax.plot(angles, yolo_values, 's-', linewidth=2, label='Custom YOLO', color='#A23B72')
        ax.fill(angles, yolo_values, alpha=0.25, color='#A23B72')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
        ax.set_title('Overall Performance Comparison', size=14, fontweight='bold', pad=20)
        ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/performance_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {save_dir}/performance_radar.png")

        # 2. Bar Chart Comparison - Detection vs Segmentation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Detection metrics
        det_metrics = ['Precision', 'Recall', 'F1-Score', 'IoU']
        swin_det = [self.results['swin']['joint']['det_precision'],
                    self.results['swin']['joint']['det_recall'],
                    self.results['swin']['joint']['det_f1'],
                    self.results['swin']['joint']['det_avg_iou']]
        yolo_det = [self.results['yolo']['joint']['det_precision'],
                    self.results['yolo']['joint']['det_recall'],
                    self.results['yolo']['joint']['det_f1'],
                    self.results['yolo']['joint']['det_avg_iou']]

        x = np.arange(len(det_metrics))
        width = 0.35

        bars1 = ax1.bar(x - width / 2, swin_det, width, label='Swin', color='#2E86AB', alpha=0.8)
        bars2 = ax1.bar(x + width / 2, yolo_det, width, label='YOLO', color='#A23B72', alpha=0.8)

        ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax1.set_title('Detection Performance', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(det_metrics, fontsize=10)
        ax1.legend(fontsize=11)
        ax1.set_ylim(0, 1.1)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        # Segmentation metrics
        seg_metrics = ['IoU', 'Dice', 'Pixel Acc']
        swin_seg = [self.results['swin']['joint']['seg_avg_iou'],
                    self.results['swin']['joint']['seg_avg_dice'],
                    self.results['swin']['joint']['seg_avg_pixel_acc']]
        yolo_seg = [self.results['yolo']['joint']['seg_avg_iou'],
                    self.results['yolo']['joint']['seg_avg_dice'],
                    self.results['yolo']['joint']['seg_avg_pixel_acc']]

        x2 = np.arange(len(seg_metrics))

        bars3 = ax2.bar(x2 - width / 2, swin_seg, width, label='Swin', color='#2E86AB', alpha=0.8)
        bars4 = ax2.bar(x2 + width / 2, yolo_seg, width, label='YOLO', color='#A23B72', alpha=0.8)

        ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax2.set_title('Segmentation Performance', fontsize=14, fontweight='bold')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(seg_metrics, fontsize=10)
        ax2.legend(fontsize=11)
        ax2.set_ylim(0, 1.1)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        for bars in [bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/detection_segmentation_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {save_dir}/detection_segmentation_comparison.png")

        # 3. Speed Comparison
        fig, ax = plt.subplots(figsize=(10, 6))

        stages = ['Detection', 'Segmentation', 'Total Pipeline']
        swin_times = [self.results['swin']['joint']['avg_det_time_ms'],
                      self.results['swin']['joint']['avg_seg_time_ms'],
                      self.results['swin']['joint']['avg_total_time_ms']]
        yolo_times = [self.results['yolo']['joint']['avg_det_time_ms'],
                      self.results['yolo']['joint']['avg_seg_time_ms'],
                      self.results['yolo']['joint']['avg_total_time_ms']]

        x = np.arange(len(stages))
        width = 0.35

        bars1 = ax.bar(x - width / 2, swin_times, width, label='Swin Transformer',
                       color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width / 2, yolo_times, width, label='Custom YOLO',
                       color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1.5)

        ax.set_ylabel('Inference Time (ms)', fontsize=12, fontweight='bold')
        ax.set_title('Speed Performance Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(stages, fontsize=11)
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.1f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Add FPS annotation
        swin_fps = self.results['swin']['joint']['fps']
        yolo_fps = self.results['yolo']['joint']['fps']
        ax.text(0.02, 0.98, f"Swin FPS: {swin_fps:.2f}\nYOLO FPS: {yolo_fps:.2f}",
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(f'{save_dir}/speed_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {save_dir}/speed_comparison.png")

        # 4. Efficiency Score (Combined metric)
        fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate efficiency scores (accuracy per second)
        swin_efficiency = (self.results['swin']['joint']['det_f1'] +
                           self.results['swin']['joint']['seg_avg_dice']) / 2 * \
                          self.results['swin']['joint']['fps']
        yolo_efficiency = (self.results['yolo']['joint']['det_f1'] +
                           self.results['yolo']['joint']['seg_avg_dice']) / 2 * \
                          self.results['yolo']['joint']['fps']

        models = ['Swin Transformer', 'Custom YOLO']
        efficiency = [swin_efficiency, yolo_efficiency]
        colors = ['#2E86AB', '#A23B72']

        bars = ax.barh(models, efficiency, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

        ax.set_xlabel('Efficiency Score (Accuracy × FPS)', fontsize=12, fontweight='bold')
        ax.set_title('Overall Efficiency Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        for i, (bar, val) in enumerate(zip(bars, efficiency)):
            ax.text(val, bar.get_y() + bar.get_height() / 2, f'{val:.3f}',
                    ha='left', va='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(f'{save_dir}/efficiency_score.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {save_dir}/efficiency_score.png")

        print(f"\n📊 All research plots saved to: {save_dir}/")

    def visualize_sample_results(self, num_samples=3, save_dir='research_results'):
        """Visualize side-by-side predictions from both models"""
        os.makedirs(save_dir, exist_ok=True)

        test_images_dir = os.path.join(self.data_dir, 'images/test')
        test_masks_dir = os.path.join(self.data_dir, 'masks/test')

        image_files = sorted([f for f in os.listdir(test_images_dir)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[:num_samples]

        for idx, img_file in enumerate(image_files):
            img_path = os.path.join(test_images_dir, img_file)
            mask_path = os.path.join(test_masks_dir, os.path.splitext(img_file)[0] + '.png')

            # Load original image
            orig_img = cv2.imread(img_path)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            h, w = orig_img.shape[:2]

            # Load ground truth mask
            gt_mask = None
            if os.path.exists(mask_path):
                gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Get predictions from both models
            swin_results = self.swin_joint_inference(img_path, 0.3)
            yolo_results = self.yolo_joint_inference(img_path, 0.25)

            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Row 1: Swin Transformer
            # Original with detections
            swin_det_img = orig_img.copy()
            for det in swin_results['detections']:
                bbox = det['bbox']
                x1, y1 = int(bbox[0] * w), int(bbox[1] * h)
                x2, y2 = int(bbox[2] * w), int(bbox[3] * h)
                cv2.rectangle(swin_det_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(swin_det_img, f"{det['confidence']:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            axes[0, 0].imshow(swin_det_img)
            axes[0, 0].set_title(f'Swin Detection\n{len(swin_results["detections"])} cracks',
                                 fontsize=11, fontweight='bold')
            axes[0, 0].axis('off')

            # Segmentation mask
            swin_mask = np.zeros((h, w), dtype=np.uint8)
            for mask_info in swin_results['masks']:
                if mask_info['mask'].size > 0:
                    mask_resized = cv2.resize(mask_info['mask'].astype(np.uint8), (w, h),
                                              interpolation=cv2.INTER_NEAREST)
                    swin_mask = np.maximum(swin_mask, mask_resized)

            axes[0, 1].imshow(swin_mask, cmap='jet')
            axes[0, 1].set_title('Swin Segmentation', fontsize=11, fontweight='bold')
            axes[0, 1].axis('off')

            # Overlay
            swin_overlay = orig_img.copy()
            swin_mask_colored = np.zeros_like(orig_img)
            swin_mask_colored[:, :, 0] = swin_mask * 255
            swin_overlay = cv2.addWeighted(swin_overlay, 0.7, swin_mask_colored, 0.3, 0)

            axes[0, 2].imshow(swin_overlay)
            axes[0, 2].set_title(f'Swin Overlay\n{swin_results["total_time"] * 1000:.1f}ms',
                                 fontsize=11, fontweight='bold')
            axes[0, 2].axis('off')

            # Row 2: Custom YOLO
            # Original with detections
            yolo_det_img = orig_img.copy()
            for det in yolo_results['detections']:
                bbox = det['bbox']
                x1, y1 = int(bbox[0] * w), int(bbox[1] * h)
                x2, y2 = int(bbox[2] * w), int(bbox[3] * h)
                cv2.rectangle(yolo_det_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(yolo_det_img, f"{det['confidence']:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            axes[1, 0].imshow(yolo_det_img)
            axes[1, 0].set_title(f'YOLO Detection\n{len(yolo_results["detections"])} cracks',
                                 fontsize=11, fontweight='bold')
            axes[1, 0].axis('off')

            # Segmentation mask
            yolo_mask = np.zeros((h, w), dtype=np.uint8)
            for mask_info in yolo_results['masks']:
                if mask_info['mask'].size > 0:
                    mask_resized = cv2.resize(mask_info['mask'].astype(np.uint8), (w, h),
                                              interpolation=cv2.INTER_NEAREST)
                    yolo_mask = np.maximum(yolo_mask, mask_resized)

            axes[1, 1].imshow(yolo_mask, cmap='jet')
            axes[1, 1].set_title('YOLO Segmentation', fontsize=11, fontweight='bold')
            axes[1, 1].axis('off')

            # Overlay
            yolo_overlay = orig_img.copy()
            yolo_mask_colored = np.zeros_like(orig_img)
            yolo_mask_colored[:, :, 2] = yolo_mask * 255
            yolo_overlay = cv2.addWeighted(yolo_overlay, 0.7, yolo_mask_colored, 0.3, 0)

            axes[1, 2].imshow(yolo_overlay)
            axes[1, 2].set_title(f'YOLO Overlay\n{yolo_results["total_time"] * 1000:.1f}ms',
                                 fontsize=11, fontweight='bold')
            axes[1, 2].axis('off')

            plt.suptitle(f'Sample {idx + 1}: {img_file}', fontsize=14, fontweight='bold', y=0.98)
            plt.tight_layout()
            plt.savefig(f'{save_dir}/sample_comparison_{idx + 1}.png', dpi=300, bbox_inches='tight')
            plt.close()

        print(f"✅ Saved {num_samples} sample visualizations to: {save_dir}/")

    def generate_research_report(self, save_path='research_results/research_report.txt'):
        """Generate detailed text report for research paper"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("COMPREHENSIVE JOINT DETECTION + SEGMENTATION COMPARISON REPORT\n")
            f.write("Crack Detection and Segmentation in Infrastructure Images\n")
            f.write("=" * 100 + "\n\n")

            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 100 + "\n")

            swin_avg = (self.results['swin']['joint']['det_f1'] +
                        self.results['swin']['joint']['seg_avg_dice']) / 2
            yolo_avg = (self.results['yolo']['joint']['det_f1'] +
                        self.results['yolo']['joint']['seg_avg_dice']) / 2

            winner = "Custom YOLO" if yolo_avg > swin_avg else "Swin Transformer"

            f.write(f"Overall Winner: {winner}\n")
            f.write(f"Swin Combined Score: {swin_avg:.4f}\n")
            f.write(f"YOLO Combined Score: {yolo_avg:.4f}\n")
            f.write(f"Performance Gap: {abs(yolo_avg - swin_avg):.4f} ({abs(yolo_avg - swin_avg) * 100:.2f}%)\n\n")

            # Detailed Metrics
            f.write("\nDETAILED PERFORMANCE METRICS\n")
            f.write("-" * 100 + "\n\n")

            f.write("1. DETECTION PERFORMANCE\n")
            f.write(f"   Metric                  Swin Transformer    Custom YOLO         Winner\n")
            f.write(f"   {'=' * 85}\n")
            det_metrics = [
                ('Precision', 'det_precision'),
                ('Recall', 'det_recall'),
                ('F1-Score', 'det_f1'),
                ('Average IoU', 'det_avg_iou')
            ]
            for name, key in det_metrics:
                swin_val = self.results['swin']['joint'][key]
                yolo_val = self.results['yolo']['joint'][key]
                winner = "YOLO" if yolo_val > swin_val else "Swin"
                f.write(f"   {name:<23} {swin_val:<19.4f} {yolo_val:<19.4f} {winner}\n")

            f.write(f"\n2. SEGMENTATION PERFORMANCE\n")
            f.write(f"   Metric                  Swin Transformer    Custom YOLO         Winner\n")
            f.write(f"   {'=' * 85}\n")
            seg_metrics = [
                ('IoU Score', 'seg_avg_iou'),
                ('Dice Score', 'seg_avg_dice'),
                ('Pixel Accuracy', 'seg_avg_pixel_acc')
            ]
            for name, key in seg_metrics:
                swin_val = self.results['swin']['joint'][key]
                yolo_val = self.results['yolo']['joint'][key]
                winner = "YOLO" if yolo_val > swin_val else "Swin"
                f.write(f"   {name:<23} {swin_val:<19.4f} {yolo_val:<19.4f} {winner}\n")

            f.write(f"\n3. COMPUTATIONAL EFFICIENCY\n")
            f.write(f"   Metric                  Swin Transformer    Custom YOLO         Winner\n")
            f.write(f"   {'=' * 85}\n")
            speed_metrics = [
                ('Detection Time (ms)', 'avg_det_time_ms'),
                ('Segmentation Time (ms)', 'avg_seg_time_ms'),
                ('Total Time (ms)', 'avg_total_time_ms'),
                ('Frames Per Second', 'fps')
            ]
            for name, key in speed_metrics:
                swin_val = self.results['swin']['joint'][key]
                yolo_val = self.results['yolo']['joint'][key]
                if key == 'fps':
                    winner = "YOLO" if yolo_val > swin_val else "Swin"
                else:
                    winner = "YOLO" if yolo_val < swin_val else "Swin"
                f.write(f"   {name:<23} {swin_val:<19.2f} {yolo_val:<19.2f} {winner}\n")

            # Statistical Analysis
            f.write(f"\n\nSTATISTICAL ANALYSIS\n")
            f.write("-" * 100 + "\n")

            speedup = self.results['swin']['joint']['avg_total_time_ms'] / \
                      self.results['yolo']['joint']['avg_total_time_ms']
            f.write(f"Speed Improvement: {speedup:.2f}x {'(YOLO faster)' if speedup > 1 else '(Swin faster)'}\n")

            acc_diff = yolo_avg - swin_avg
            f.write(f"Accuracy Difference: {acc_diff:+.4f} ({'YOLO better' if acc_diff > 0 else 'Swin better'})\n")

            # Key Findings
            f.write(f"\n\nKEY FINDINGS FOR RESEARCH PAPER\n")
            f.write("-" * 100 + "\n")
            f.write("1. ")
            if yolo_avg > swin_avg:
                f.write(
                    f"Custom YOLO outperforms Swin Transformer by {(yolo_avg - swin_avg) * 100:.2f}% in combined accuracy.\n")
            else:
                f.write(
                    f"Swin Transformer outperforms Custom YOLO by {(swin_avg - yolo_avg) * 100:.2f}% in combined accuracy.\n")

            f.write("2. ")
            if self.results['yolo']['joint']['fps'] > self.results['swin']['joint']['fps']:
                f.write(
                    f"Custom YOLO achieves {speedup:.2f}x faster inference speed ({self.results['yolo']['joint']['fps']:.2f} FPS vs {self.results['swin']['joint']['fps']:.2f} FPS).\n")
            else:
                f.write(f"Swin Transformer achieves faster inference speed.\n")

            f.write("3. ")
            if self.results['yolo']['joint']['det_f1'] > self.results['swin']['joint']['det_f1']:
                f.write(
                    f"Custom YOLO shows superior detection performance (F1: {self.results['yolo']['joint']['det_f1']:.4f} vs {self.results['swin']['joint']['det_f1']:.4f}).\n")
            else:
                f.write(f"Swin Transformer shows superior detection performance.\n")

            f.write("4. ")
            if self.results['yolo']['joint']['seg_avg_dice'] > self.results['swin']['joint']['seg_avg_dice']:
                f.write(
                    f"Custom YOLO achieves better segmentation accuracy (Dice: {self.results['yolo']['joint']['seg_avg_dice']:.4f} vs {self.results['swin']['joint']['seg_avg_dice']:.4f}).\n")
            else:
                f.write(f"Swin Transformer achieves better segmentation accuracy.\n")

            f.write("\n" + "=" * 100 + "\n")

        print(f"✅ Research report saved to: {save_path}")

    def save_results_json(self, save_path='research_results/comparison_results.json'):
        """Save all results to JSON for further analysis"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=4)

        print(f"✅ Results JSON saved to: {save_path}")


# ==================== MAIN EXECUTION ====================
def main():
    print("=" * 120)
    print("JOINT DETECTION + SEGMENTATION COMPARISON FOR CRACK ANALYSIS")
    print("=" * 120)

    # Configuration
    DATA_DIR = "crack_segmentation_dataset"  # Your detection dataset

    # Model paths
    SWIN_DETECTION = "best_swin_crack_detection.pth"
    SWIN_SEGMENTATION = "best_swin_crack_segmentation.pth"
    YOLO_DETECTION = "yolo12s_cbam_ca_crack.pt"
    YOLO_SEGMENTATION = "yolo12s_seg_cbam_ca_crack.pt"  # Update this path

    print("\n📁 Dataset:", DATA_DIR)
    print("📦 Model Configuration:")
    print(f"   Swin Detection: {SWIN_DETECTION}")
    print(f"   Swin Segmentation: {SWIN_SEGMENTATION}")
    print(f"   YOLO Detection: {YOLO_DETECTION}")
    print(f"   YOLO Segmentation: {YOLO_SEGMENTATION}")

    # Initialize comparator
    comparator = JointModelComparator(DATA_DIR, image_size=224)

    # Load models
    print("\n" + "=" * 120)
    swin_loaded = comparator.load_swin_models(SWIN_DETECTION, SWIN_SEGMENTATION)
    yolo_loaded = comparator.load_yolo_models(YOLO_DETECTION, YOLO_SEGMENTATION)

    if not (swin_loaded and yolo_loaded):
        print("\n❌ Failed to load all models. Please check the paths.")
        return

    # Run evaluations
    print("\n" + "=" * 120)
    print("STARTING COMPREHENSIVE EVALUATION")
    print("=" * 120)

    # Evaluate Swin pipeline
    comparator.results['swin']['joint'] = comparator.evaluate_joint_pipeline(
        model_type='swin',
        det_conf=0.3,
        iou_threshold=0.5
    )

    # Evaluate YOLO pipeline
    comparator.results['yolo']['joint'] = comparator.evaluate_joint_pipeline(
        model_type='yolo',
        det_conf=0.25,
        iou_threshold=0.5
    )

    # Generate all outputs
    print("\n" + "=" * 120)
    print("GENERATING RESEARCH OUTPUTS")
    print("=" * 120)

    comparator.print_comparison_table()
    comparator.generate_research_plots()
    comparator.visualize_sample_results(num_samples=5)
    comparator.generate_research_report()
    comparator.save_results_json()

    print("\n" + "=" * 120)
    print("✅ COMPARISON COMPLETE!")
    print("=" * 120)
    print("\n📊 Generated Research Materials:")
    print("   1. research_results/performance_radar.png - Overall performance comparison")
    print("   2. research_results/detection_segmentation_comparison.png - Detailed metrics")
    print("   3. research_results/speed_comparison.png - Inference speed analysis")
    print("   4. research_results/efficiency_score.png - Combined efficiency metric")
    print("   5. research_results/sample_comparison_*.png - Visual results")
    print("   6. research_results/research_report.txt - Detailed text report")
    print("   7. research_results/comparison_results.json - Raw data for tables")
    print("\n💡 Use these materials for your research paper!")
    print("=" * 120)


if __name__ == "__main__":
    main()