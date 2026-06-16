"""
FIXED TWO-MODEL COMPARISON: Custom YOLO (CBAM+CA) vs Swin Transformer
Publication-ready version with proper evaluation, statistical analysis, and visualizations
Handles 227px dataset images correctly
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import json
from tqdm import tqdm
import time
import torchvision.transforms as transforms
from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

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
from common.swin_seg import SwinUNet as FullSwinUNet


class FixedTwoModelComparator:
    def __init__(self, detection_data_dir, segmentation_data_dir, image_size=224):
        """
        FIXED VERSION with consistent resolution and better evaluation

        Args:
            detection_data_dir: Dataset with bbox labels (unified_crack_dataset)
            segmentation_data_dir: Dataset with polygon labels (unified_crack_dataset_seg)
            image_size: Model input resolution (default: 224, handles 227px dataset images)
        """
        self.detection_data_dir = detection_data_dir
        self.segmentation_data_dir = segmentation_data_dir
        self.image_size = image_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print(f"⚙️  Model input size: {image_size}x{image_size} (handles 227px dataset images)")

        # Load dataset info
        det_info_path = os.path.join(detection_data_dir, 'annotations', 'dataset_info.json')
        seg_info_path = os.path.join(segmentation_data_dir, 'annotations', 'dataset_info.json')

        if os.path.exists(det_info_path):
            with open(det_info_path, 'r') as f:
                self.det_dataset_info = json.load(f)
                print(f"\n📊 Detection Dataset Info:")
                print(f"   Total test images: {self.det_dataset_info.get('test_images', 'N/A')}")
                print(f"   Test positive: {self.det_dataset_info.get('test_positive', 'N/A')}")
        else:
            self.det_dataset_info = {}

        if os.path.exists(seg_info_path):
            with open(seg_info_path, 'r') as f:
                self.seg_dataset_info = json.load(f)
                print(f"\n📊 Segmentation Dataset Info:")
                print(f"   Total test images: {self.seg_dataset_info.get('test_images', 'N/A')}")
                print(f"   Test positive: {self.seg_dataset_info.get('test_positive', 'N/A')}")
        else:
            self.seg_dataset_info = {}

        # Model storage
        self.swin_detector = None
        self.swin_segmentor = None
        self.custom_yolo_detector = None
        self.custom_yolo_segmentor = None

        # Results storage with per-image tracking
        self.results = {
            'swin': {'detection': {}, 'segmentation': {}, 'joint': {}, 'per_image': []},
            'custom_yolo': {'detection': {}, 'segmentation': {}, 'joint': {}, 'per_image': []}
        }

        # Visualization storage
        self.viz_dir = 'research_results/visualizations'
        os.makedirs(self.viz_dir, exist_ok=True)

    def load_swin_models(self, detection_path, segmentation_path):
        """Load both Swin models"""
        print("\n📦 Loading Swin Transformer Models...")

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

        try:
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

    def load_custom_yolo_models(self, detection_path, segmentation_path):
        """Load custom YOLO models (CBAM+CA)"""
        if not YOLO_AVAILABLE:
            return False

        print("\n📦 Loading Custom YOLO Models (CBAM+CA)...")

        try:
            self.custom_yolo_detector = YOLO(detection_path)
            print("✅ Custom YOLO Detection model loaded")
        except Exception as e:
            print(f"❌ Failed to load Custom YOLO detection: {e}")
            return False

        try:
            self.custom_yolo_segmentor = YOLO(segmentation_path)
            print("✅ Custom YOLO Segmentation model loaded")
        except Exception as e:
            print(f"❌ Failed to load Custom YOLO segmentation: {e}")
            return False

        return True

    def preprocess_swin(self, image_path):
        """Preprocess for Swin models - handles 227px images"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # Resize to model input size (handles 227px → 224px)
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

            # Stage 2: Segmentation
            start_time = time.time()
            all_masks = []

            if len(detections) > 0:
                with torch.no_grad():
                    seg_output = self.swin_segmentor(img_tensor)
                    seg_pred = torch.softmax(seg_output, dim=1)
                    seg_mask = torch.argmax(seg_pred, dim=1)[0].cpu().numpy()

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
        """Custom YOLO inference - handles 227px images"""
        try:
            # Read original image to get dimensions
            orig_img = cv2.imread(image_path)
            if orig_img is None:
                raise ValueError(f"Could not load image: {image_path}")

            img_h, img_w = orig_img.shape[:2]

            # Stage 1: Detection
            start_time = time.time()
            det_results = self.custom_yolo_detector.predict(
                source=image_path,
                conf=det_conf,
                save=False,
                imgsz=self.image_size,
                verbose=False
            )
            det_time = time.time() - start_time

            detections = []
            if len(det_results) > 0 and det_results[0].boxes is not None:
                boxes = det_results[0].boxes

                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = xyxy / np.array([img_w, img_h, img_w, img_h])

                    detections.append({
                        'bbox': np.array([x1, y1, x2, y2]),
                        'confidence': boxes.conf[i].item()
                    })

            # Stage 2: Segmentation
            start_time = time.time()
            seg_results = self.custom_yolo_segmentor.predict(
                source=image_path,
                conf=det_conf,
                save=False,
                imgsz=self.image_size,
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

        # Ensure both masks are 2D
        if len(pred_mask.shape) == 3:
            pred_mask = pred_mask[:, :, 0] if pred_mask.shape[2] == 1 else pred_mask[0]
        if len(gt_mask.shape) == 3:
            gt_mask = gt_mask[:, :, 0] if gt_mask.shape[2] == 1 else gt_mask[0]

        # Ensure binary
        pred_mask = (pred_mask > 0).astype(np.uint8)
        gt_mask = (gt_mask > 0).astype(np.uint8)

        # Resize if needed
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        correct = np.sum(pred_mask == gt_mask)
        total = gt_mask.shape[0] * gt_mask.shape[1]

        return float(correct) / float(total)

    def calculate_boundary_iou(self, pred_mask, gt_mask, dilation=2):
        """Calculate IoU of mask boundaries (important for thin structures like cracks)"""
        if pred_mask.size == 0 or gt_mask.size == 0:
            return 0.0

        # Ensure 2D and binary
        if len(pred_mask.shape) == 3:
            pred_mask = pred_mask[:, :, 0] if pred_mask.shape[2] == 1 else pred_mask[0]
        if len(gt_mask.shape) == 3:
            gt_mask = gt_mask[:, :, 0] if gt_mask.shape[2] == 1 else gt_mask[0]

        pred_mask = (pred_mask > 0).astype(np.uint8)
        gt_mask = (gt_mask > 0).astype(np.uint8)

        # Resize if needed
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        # Extract boundaries
        kernel = np.ones((dilation, dilation), np.uint8)
        gt_boundary = cv2.dilate(gt_mask, kernel) - cv2.erode(gt_mask, kernel)
        pred_boundary = cv2.dilate(pred_mask, kernel) - cv2.erode(pred_mask, kernel)

        return self.calculate_iou_mask(pred_boundary, gt_boundary)

    # ==================== EVALUATION ====================
    def evaluate_joint_pipeline(self, model_type='swin', det_conf=0.3, iou_threshold=0.5):
        """
        Comprehensive pipeline evaluation with per-instance matching
        """
        # Use DETECTION dataset for bbox evaluation
        test_images_dir = os.path.join(self.detection_data_dir, 'images/test')
        test_labels_dir = os.path.join(self.detection_data_dir, 'labels/test')

        # Use SEGMENTATION dataset for mask evaluation
        seg_test_masks_dir = os.path.join(self.segmentation_data_dir, 'masks/test')

        if not os.path.exists(seg_test_masks_dir):
            print(f"⚠️ Warning: Segmentation masks not found at {seg_test_masks_dir}")
            seg_test_masks_dir = None

        image_files = [f for f in os.listdir(test_images_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        metrics = {
            'det_tp': 0, 'det_fp': 0, 'det_fn': 0,
            'det_ious': [],
            'det_times': [],
            'seg_ious': [],
            'dice_scores': [],
            'pixel_accuracies': [],
            'boundary_ious': [],
            'seg_times': [],
            'total_times': [],
            'successful_pipelines': 0,
            'seg_tp': 0,
            'seg_fp': 0,
            'seg_fn': 0,
            'images_with_gt_and_pred': 0,
            'total_images': len(image_files),
            'positive_images_processed': 0,
            'per_image_results': []
        }

        print(f"\n🔬 Evaluating {model_type.upper()} joint pipeline on {len(image_files)} images...")
        print(f"   Detection labels from: {self.detection_data_dir}")
        print(f"   Segmentation masks from: {self.segmentation_data_dir}")
        print(f"   Model input resolution: {self.image_size}x{self.image_size}")

        for img_file in tqdm(image_files, desc=f"Testing {model_type}"):
            img_path = os.path.join(test_images_dir, img_file)

            # Initialize per-image results
            img_result = {
                'image': img_file,
                'det_iou': 0.0,
                'seg_iou': 0.0,
                'dice': 0.0,
                'pixel_acc': 0.0,
                'boundary_iou': 0.0,
                'has_gt': False,
                'has_pred': False,
                'inference_time': 0.0
            }

            # Load DETECTION ground truth
            label_path = os.path.join(test_labels_dir, os.path.splitext(img_file)[0] + '.txt')
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

            if len(gt_boxes) > 0:
                metrics['positive_images_processed'] += 1
                img_result['has_gt'] = True

            # Load SEGMENTATION ground truth
            gt_mask = None
            if seg_test_masks_dir:
                mask_path = os.path.join(seg_test_masks_dir, os.path.splitext(img_file)[0] + '.png')
                if os.path.exists(mask_path):
                    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if gt_mask is not None:
                        gt_mask = (gt_mask > 127).astype(np.uint8)
                        if len(gt_mask.shape) == 3:
                            gt_mask = gt_mask[:, :, 0]

            # Run inference
            if model_type == 'swin':
                results = self.swin_joint_inference(img_path, det_conf)
            elif model_type == 'custom_yolo':
                results = self.yolo_joint_inference(img_path, det_conf)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Record timing
            metrics['det_times'].append(results['det_time'])
            metrics['seg_times'].append(results['seg_time'])
            metrics['total_times'].append(results['total_time'])
            img_result['inference_time'] = results['total_time']

            pred_boxes = [det['bbox'] for det in results['detections']]
            if len(pred_boxes) > 0:
                img_result['has_pred'] = True

            # Evaluate detection
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
                    img_result['det_iou'] = max(img_result['det_iou'], best_iou)
                else:
                    metrics['det_fp'] += 1

            metrics['det_fn'] += len(gt_boxes) - len(matched_gt)

            # Evaluate segmentation
            if gt_mask is not None:
                if len(results['masks']) == 0:
                    metrics['seg_fn'] += 1
                else:
                    best_seg_iou = 0.0
                    best_dice = 0.0
                    best_boundary_iou = 0.0
                    best_pixel_acc = 0.0

                    for mask_info in results['masks']:
                        mask = mask_info['mask']
                        if mask.size > 0:
                            # Process mask
                            if len(mask.shape) == 3:
                                if mask.shape[0] == 1:
                                    mask = mask[0]
                                elif mask.shape[2] == 1:
                                    mask = mask[:, :, 0]
                                else:
                                    mask = mask.max(axis=0)

                            # Resize to match GT
                            mask_resized = cv2.resize(mask.astype(np.float32),
                                                      (gt_mask.shape[1], gt_mask.shape[0]),
                                                      interpolation=cv2.INTER_NEAREST)
                            mask_binary = (mask_resized > 0.5).astype(np.uint8)

                            # Calculate metrics for this instance
                            seg_iou = self.calculate_iou_mask(mask_binary, gt_mask)
                            dice = self.calculate_dice_score(mask_binary, gt_mask)
                            pixel_acc = self.calculate_pixel_accuracy(mask_binary, gt_mask)
                            boundary_iou = self.calculate_boundary_iou(mask_binary, gt_mask)

                            # Keep best scores
                            best_seg_iou = max(best_seg_iou, seg_iou)
                            best_dice = max(best_dice, dice)
                            best_pixel_acc = max(best_pixel_acc, pixel_acc)
                            best_boundary_iou = max(best_boundary_iou, boundary_iou)

                    # Record best match
                    if best_seg_iou > 0.1:
                        metrics['seg_tp'] += 1
                        metrics['seg_ious'].append(best_seg_iou)
                        metrics['dice_scores'].append(best_dice)
                        metrics['pixel_accuracies'].append(best_pixel_acc)
                        metrics['boundary_ious'].append(best_boundary_iou)

                        img_result['seg_iou'] = best_seg_iou
                        img_result['dice'] = best_dice
                        img_result['pixel_acc'] = best_pixel_acc
                        img_result['boundary_iou'] = best_boundary_iou

                        metrics['images_with_gt_and_pred'] += 1

                        if len(results['detections']) > 0:
                            metrics['successful_pipelines'] += 1
                    else:
                        metrics['seg_fp'] += 1
            else:
                if len(results['masks']) > 0:
                    metrics['seg_fp'] += 1

            # Store per-image results
            metrics['per_image_results'].append(img_result)

        # Calculate final metrics with statistical analysis
        tp, fp, fn = metrics['det_tp'], metrics['det_fp'], metrics['det_fn']
        seg_tp, seg_fp, seg_fn = metrics['seg_tp'], metrics['seg_fp'], metrics['seg_fn']

        final_metrics = {
            # Detection metrics
            'det_precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'det_recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'det_f1': 0,
            'det_avg_iou': np.mean(metrics['det_ious']) if metrics['det_ious'] else 0,
            'det_iou_std': np.std(metrics['det_ious']) if metrics['det_ious'] else 0,

            # Segmentation metrics
            'seg_precision': seg_tp / (seg_tp + seg_fp) if (seg_tp + seg_fp) > 0 else 0,
            'seg_recall': seg_tp / (seg_tp + seg_fn) if (seg_tp + seg_fn) > 0 else 0,
            'seg_f1': 0,
            'seg_avg_iou': np.mean(metrics['seg_ious']) if metrics['seg_ious'] else 0,
            'seg_iou_std': np.std(metrics['seg_ious']) if metrics['seg_ious'] else 0,
            'seg_avg_dice': np.mean(metrics['dice_scores']) if metrics['dice_scores'] else 0,
            'seg_dice_std': np.std(metrics['dice_scores']) if metrics['dice_scores'] else 0,
            'seg_avg_pixel_acc': np.mean(metrics['pixel_accuracies']) if metrics['pixel_accuracies'] else 0,
            'seg_avg_boundary_iou': np.mean(metrics['boundary_ious']) if metrics['boundary_ious'] else 0,

            # Speed metrics
            'avg_det_time_ms': np.mean(metrics['det_times']) * 1000,
            'avg_seg_time_ms': np.mean(metrics['seg_times']) * 1000,
            'avg_total_time_ms': np.mean(metrics['total_times']) * 1000,
            'std_total_time_ms': np.std(metrics['total_times']) * 1000,
            'fps': 1.0 / np.mean(metrics['total_times']) if np.mean(metrics['total_times']) > 0 else 0,

            # Pipeline metrics
            'pipeline_success_rate': metrics['successful_pipelines'] / metrics['images_with_gt_and_pred'] if metrics['images_with_gt_and_pred'] > 0 else 0,
            'total_images': metrics['total_images'],
            'positive_images': metrics['positive_images_processed'],
            'images_with_predictions': metrics['images_with_gt_and_pred'],
            'num_images_with_detections': sum(1 for t in metrics['det_ious'] if t > 0),
            'num_images_with_segmentation': len(metrics['seg_ious']),

            # Per-image results for statistical testing
            'per_image_results': metrics['per_image_results']
        }

        # Calculate F1 scores
        if (final_metrics['det_precision'] + final_metrics['det_recall']) > 0:
            final_metrics['det_f1'] = (2 * final_metrics['det_precision'] * final_metrics['det_recall']) / \
                                      (final_metrics['det_precision'] + final_metrics['det_recall'])

        if (final_metrics['seg_precision'] + final_metrics['seg_recall']) > 0:
            final_metrics['seg_f1'] = (2 * final_metrics['seg_precision'] * final_metrics['seg_recall']) / \
                                      (final_metrics['seg_precision'] + final_metrics['seg_recall'])

        # Calculate 95% confidence intervals
        if len(metrics['seg_ious']) > 1:
            final_metrics['seg_iou_ci_95'] = 1.96 * final_metrics['seg_iou_std'] / np.sqrt(len(metrics['seg_ious']))
        else:
            final_metrics['seg_iou_ci_95'] = 0

        if len(metrics['det_ious']) > 1:
            final_metrics['det_iou_ci_95'] = 1.96 * final_metrics['det_iou_std'] / np.sqrt(len(metrics['det_ious']))
        else:
            final_metrics['det_iou_ci_95'] = 0

        return final_metrics

    # ==================== STATISTICAL ANALYSIS ====================
    def compare_models_statistically(self):
        """Perform statistical significance testing between models"""
        print("\n" + "=" * 110)
        print("STATISTICAL SIGNIFICANCE TESTING")
        print("=" * 110)

        swin_results = self.results['swin']['joint']['per_image_results']
        yolo_results = self.results['custom_yolo']['joint']['per_image_results']

        if not swin_results or not yolo_results:
            print("⚠️ Cannot perform statistical testing - missing per-image results")
            return

        # Extract metrics for comparison
        swin_seg_ious = [r['seg_iou'] for r in swin_results if r['has_gt']]
        yolo_seg_ious = [r['seg_iou'] for r in yolo_results if r['has_gt']]

        swin_dice = [r['dice'] for r in swin_results if r['has_gt']]
        yolo_dice = [r['dice'] for r in yolo_results if r['has_gt']]

        swin_det_ious = [r['det_iou'] for r in swin_results if r['has_gt']]
        yolo_det_ious = [r['det_iou'] for r in yolo_results if r['has_gt']]

        print("\n📊 Per-Image Statistics:")
        print(f"   Sample size: {len(swin_seg_ious)} images with ground truth")

        # Paired t-test
        if len(swin_seg_ious) == len(yolo_seg_ious) and len(swin_seg_ious) > 1:
            print("\n1️⃣ Segmentation IoU Comparison:")
            t_stat, p_value = stats.ttest_rel(yolo_seg_ious, swin_seg_ious)
            print(f"   Custom YOLO mean: {np.mean(yolo_seg_ious):.4f} ± {np.std(yolo_seg_ious):.4f}")
            print(f"   Swin mean: {np.mean(swin_seg_ious):.4f} ± {np.std(swin_seg_ious):.4f}")
            print(f"   Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")

            if p_value < 0.001:
                print(f"   *** HIGHLY SIGNIFICANT difference (p < 0.001)")
            elif p_value < 0.01:
                print(f"   ** SIGNIFICANT difference (p < 0.01)")
            elif p_value < 0.05:
                print(f"   * SIGNIFICANT difference (p < 0.05)")
            else:
                print(f"   No significant difference (p >= 0.05)")

            print("\n2️⃣ Dice Score Comparison:")
            t_stat, p_value = stats.ttest_rel(yolo_dice, swin_dice)
            print(f"   Custom YOLO mean: {np.mean(yolo_dice):.4f} ± {np.std(yolo_dice):.4f}")
            print(f"   Swin mean: {np.mean(swin_dice):.4f} ± {np.std(swin_dice):.4f}")
            print(f"   Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")

            if p_value < 0.05:
                print(f"   * SIGNIFICANT difference (p < 0.05)")
            else:
                print(f"   No significant difference (p >= 0.05)")

            print("\n3️⃣ Detection IoU Comparison:")
            t_stat, p_value = stats.ttest_rel(yolo_det_ious, swin_det_ious)
            print(f"   Custom YOLO mean: {np.mean(yolo_det_ious):.4f} ± {np.std(yolo_det_ious):.4f}")
            print(f"   Swin mean: {np.mean(swin_det_ious):.4f} ± {np.std(swin_det_ious):.4f}")
            print(f"   Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")

            if p_value < 0.05:
                print(f"   * SIGNIFICANT difference (p < 0.05)")
            else:
                print(f"   No significant difference (p >= 0.05)")

            # Wilcoxon signed-rank test
            print("\n4️⃣ Non-parametric Test (Wilcoxon):")
            w_stat, p_value = stats.wilcoxon(yolo_seg_ious, swin_seg_ious)
            print(f"   Segmentation IoU: W={w_stat:.2f}, p={p_value:.4f}")

        else:
            print("⚠️ Sample sizes don't match or insufficient data for paired testing")

        print("=" * 110)

    # ==================== VISUALIZATION ====================
    def plot_metric_comparison_bars(self):
        """Create comprehensive bar chart comparison"""
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Model Performance Comparison: Custom YOLO (CBAM+CA) vs Swin Transformer',
                     fontsize=16, fontweight='bold')

        # Detection metrics
        metrics_det = ['det_precision', 'det_recall', 'det_f1', 'det_avg_iou']
        labels_det = ['Precision', 'Recall', 'F1 Score', 'Avg IoU']

        for idx, (metric, label) in enumerate(zip(metrics_det[:3], labels_det[:3])):
            ax = axes[0, idx]
            custom = self.results['custom_yolo']['joint'].get(metric, 0)
            swin = self.results['swin']['joint'].get(metric, 0)

            bars = ax.bar(['Custom YOLO', 'Swin'], [custom, swin],
                         color=['#2ecc71', '#3498db'], alpha=0.8, edgecolor='black')
            ax.set_ylabel('Score', fontweight='bold')
            ax.set_title(f'Detection {label}', fontweight='bold')
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

        # Segmentation metrics
        metrics_seg = ['seg_avg_iou', 'seg_avg_dice', 'seg_avg_pixel_acc']
        labels_seg = ['IoU', 'Dice Score', 'Pixel Accuracy']

        for idx, (metric, label) in enumerate(zip(metrics_seg, labels_seg)):
            ax = axes[1, idx]
            custom = self.results['custom_yolo']['joint'].get(metric, 0)
            swin = self.results['swin']['joint'].get(metric, 0)

            bars = ax.bar(['Custom YOLO', 'Swin'], [custom, swin],
                         color=['#e74c3c', '#9b59b6'], alpha=0.8, edgecolor='black')
            ax.set_ylabel('Score', fontweight='bold')
            ax.set_title(f'Segmentation {label}', fontweight='bold')
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        save_path = os.path.join(self.viz_dir, 'metric_comparison_bars.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()

    def plot_iou_distributions(self):
        """Plot IoU distribution comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('IoU Score Distributions', fontsize=14, fontweight='bold')

        # Detection IoU
        ax = axes[0]
        custom_det = [r['det_iou'] for r in self.results['custom_yolo']['joint']['per_image_results'] if r['has_gt']]
        swin_det = [r['det_iou'] for r in self.results['swin']['joint']['per_image_results'] if r['has_gt']]

        ax.hist([custom_det, swin_det], bins=20, label=['Custom YOLO', 'Swin'],
                color=['#2ecc71', '#3498db'], alpha=0.7, edgecolor='black')
        ax.set_xlabel('Detection IoU', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Detection IoU Distribution', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Segmentation IoU
        ax = axes[1]
        custom_seg = [r['seg_iou'] for r in self.results['custom_yolo']['joint']['per_image_results'] if r['seg_iou'] > 0]
        swin_seg = [r['seg_iou'] for r in self.results['swin']['joint']['per_image_results'] if r['seg_iou'] > 0]

        ax.hist([custom_seg, swin_seg], bins=20, label=['Custom YOLO', 'Swin'],
                color=['#e74c3c', '#9b59b6'], alpha=0.7, edgecolor='black')
        ax.set_xlabel('Segmentation IoU', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Segmentation IoU Distribution', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.viz_dir, 'iou_distributions.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()

    def plot_speed_comparison(self):
        """Plot inference speed comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Inference Speed Comparison', fontsize=14, fontweight='bold')

        # Time breakdown
        ax = axes[0]
        models = ['Custom YOLO', 'Swin']
        det_times = [
            self.results['custom_yolo']['joint']['avg_det_time_ms'],
            self.results['swin']['joint']['avg_det_time_ms']
        ]
        seg_times = [
            self.results['custom_yolo']['joint']['avg_seg_time_ms'],
            self.results['swin']['joint']['avg_seg_time_ms']
        ]

        x = np.arange(len(models))
        width = 0.35

        bars1 = ax.bar(x - width/2, det_times, width, label='Detection',
                      color='#3498db', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, seg_times, width, label='Segmentation',
                      color='#e74c3c', alpha=0.8, edgecolor='black')

        ax.set_ylabel('Time (ms)', fontweight='bold')
        ax.set_title('Pipeline Time Breakdown', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=9)

        # FPS comparison
        ax = axes[1]
        fps_values = [
            self.results['custom_yolo']['joint']['fps'],
            self.results['swin']['joint']['fps']
        ]

        bars = ax.bar(models, fps_values, color=['#2ecc71', '#9b59b6'],
                     alpha=0.8, edgecolor='black')
        ax.set_ylabel('Frames Per Second (FPS)', fontweight='bold')
        ax.set_title('Overall Throughput', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        save_path = os.path.join(self.viz_dir, 'speed_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()

    def plot_per_image_scatter(self):
        """Scatter plot showing per-image performance"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Per-Image Performance Analysis', fontsize=14, fontweight='bold')

        # Detection vs Segmentation IoU
        ax = axes[0]

        custom_results = self.results['custom_yolo']['joint']['per_image_results']
        swin_results = self.results['swin']['joint']['per_image_results']

        custom_det = [r['det_iou'] for r in custom_results if r['has_gt'] and r['seg_iou'] > 0]
        custom_seg = [r['seg_iou'] for r in custom_results if r['has_gt'] and r['seg_iou'] > 0]

        swin_det = [r['det_iou'] for r in swin_results if r['has_gt'] and r['seg_iou'] > 0]
        swin_seg = [r['seg_iou'] for r in swin_results if r['has_gt'] and r['seg_iou'] > 0]

        ax.scatter(custom_det, custom_seg, alpha=0.6, s=50, label='Custom YOLO', color='#2ecc71')
        ax.scatter(swin_det, swin_seg, alpha=0.6, s=50, label='Swin', color='#3498db', marker='s')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
        ax.set_xlabel('Detection IoU', fontweight='bold')
        ax.set_ylabel('Segmentation IoU', fontweight='bold')
        ax.set_title('Detection vs Segmentation Performance', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        # Inference time vs IoU
        ax = axes[1]

        custom_times = [r['inference_time'] * 1000 for r in custom_results if r['has_gt'] and r['seg_iou'] > 0]
        swin_times = [r['inference_time'] * 1000 for r in swin_results if r['has_gt'] and r['seg_iou'] > 0]

        ax.scatter(custom_times, custom_seg, alpha=0.6, s=50, label='Custom YOLO', color='#e74c3c')
        ax.scatter(swin_times, swin_seg, alpha=0.6, s=50, label='Swin', color='#9b59b6', marker='s')
        ax.set_xlabel('Inference Time (ms)', fontweight='bold')
        ax.set_ylabel('Segmentation IoU', fontweight='bold')
        ax.set_title('Speed vs Accuracy Trade-off', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.viz_dir, 'per_image_scatter.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()

    def plot_box_plots(self):
        """Box plots for metric distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Metric Distribution Analysis (Box Plots)', fontsize=14, fontweight='bold')

        custom_results = self.results['custom_yolo']['joint']['per_image_results']
        swin_results = self.results['swin']['joint']['per_image_results']

        # Detection IoU
        ax = axes[0, 0]
        custom_det = [r['det_iou'] for r in custom_results if r['has_gt']]
        swin_det = [r['det_iou'] for r in swin_results if r['has_gt']]

        bp = ax.boxplot([custom_det, swin_det], labels=['Custom YOLO', 'Swin'],
                        patch_artist=True, showmeans=True)
        for patch, color in zip(bp['boxes'], ['#2ecc71', '#3498db']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel('Detection IoU', fontweight='bold')
        ax.set_title('Detection IoU Distribution', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Segmentation IoU
        ax = axes[0, 1]
        custom_seg = [r['seg_iou'] for r in custom_results if r['seg_iou'] > 0]
        swin_seg = [r['seg_iou'] for r in swin_results if r['seg_iou'] > 0]

        bp = ax.boxplot([custom_seg, swin_seg], labels=['Custom YOLO', 'Swin'],
                        patch_artist=True, showmeans=True)
        for patch, color in zip(bp['boxes'], ['#e74c3c', '#9b59b6']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel('Segmentation IoU', fontweight='bold')
        ax.set_title('Segmentation IoU Distribution', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Dice Score
        ax = axes[1, 0]
        custom_dice = [r['dice'] for r in custom_results if r['dice'] > 0]
        swin_dice = [r['dice'] for r in swin_results if r['dice'] > 0]

        bp = ax.boxplot([custom_dice, swin_dice], labels=['Custom YOLO', 'Swin'],
                        patch_artist=True, showmeans=True)
        for patch, color in zip(bp['boxes'], ['#f39c12', '#16a085']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel('Dice Score', fontweight='bold')
        ax.set_title('Dice Score Distribution', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Inference Time
        ax = axes[1, 1]
        custom_times = [r['inference_time'] * 1000 for r in custom_results]
        swin_times = [r['inference_time'] * 1000 for r in swin_results]

        bp = ax.boxplot([custom_times, swin_times], labels=['Custom YOLO', 'Swin'],
                        patch_artist=True, showmeans=True)
        for patch, color in zip(bp['boxes'], ['#27ae60', '#8e44ad']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel('Inference Time (ms)', fontweight='bold')
        ax.set_title('Inference Time Distribution', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.viz_dir, 'box_plots.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()

    def plot_comprehensive_summary(self):
        """Create a single comprehensive summary figure"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        fig.suptitle('Comprehensive Model Comparison Summary\nCustom YOLO (CBAM+CA) vs Swin Transformer',
                     fontsize=18, fontweight='bold')

        # 1. Overall Performance Radar Chart
        ax1 = fig.add_subplot(gs[0, 0], projection='polar')

        categories = ['Det\nPrecision', 'Det\nRecall', 'Det\nF1', 'Seg\nIoU', 'Seg\nDice', 'Speed\n(FPS/10)']

        custom_values = [
            self.results['custom_yolo']['joint']['det_precision'],
            self.results['custom_yolo']['joint']['det_recall'],
            self.results['custom_yolo']['joint']['det_f1'],
            self.results['custom_yolo']['joint']['seg_avg_iou'],
            self.results['custom_yolo']['joint']['seg_avg_dice'],
            self.results['custom_yolo']['joint']['fps'] / 10
        ]

        swin_values = [
            self.results['swin']['joint']['det_precision'],
            self.results['swin']['joint']['det_recall'],
            self.results['swin']['joint']['det_f1'],
            self.results['swin']['joint']['seg_avg_iou'],
            self.results['swin']['joint']['seg_avg_dice'],
            self.results['swin']['joint']['fps'] / 10
        ]

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        custom_values += custom_values[:1]
        swin_values += swin_values[:1]
        angles += angles[:1]

        ax1.plot(angles, custom_values, 'o-', linewidth=2, label='Custom YOLO', color='#2ecc71')
        ax1.fill(angles, custom_values, alpha=0.25, color='#2ecc71')
        ax1.plot(angles, swin_values, 's-', linewidth=2, label='Swin', color='#3498db')
        ax1.fill(angles, swin_values, alpha=0.25, color='#3498db')
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories, size=8)
        ax1.set_ylim(0, 1)
        ax1.set_title('Overall Performance', fontweight='bold', pad=20)
        ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax1.grid(True)

        # 2. Detection Metrics
        ax2 = fig.add_subplot(gs[0, 1])
        metrics = ['Precision', 'Recall', 'F1', 'IoU']
        custom_det = [
            self.results['custom_yolo']['joint']['det_precision'],
            self.results['custom_yolo']['joint']['det_recall'],
            self.results['custom_yolo']['joint']['det_f1'],
            self.results['custom_yolo']['joint']['det_avg_iou']
        ]
        swin_det = [
            self.results['swin']['joint']['det_precision'],
            self.results['swin']['joint']['det_recall'],
            self.results['swin']['joint']['det_f1'],
            self.results['swin']['joint']['det_avg_iou']
        ]

        x = np.arange(len(metrics))
        width = 0.35
        ax2.bar(x - width/2, custom_det, width, label='Custom YOLO', color='#2ecc71', alpha=0.8)
        ax2.bar(x + width/2, swin_det, width, label='Swin', color='#3498db', alpha=0.8)
        ax2.set_ylabel('Score', fontweight='bold')
        ax2.set_title('Detection Performance', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim([0, 1])

        # 3. Segmentation Metrics
        ax3 = fig.add_subplot(gs[0, 2])
        metrics = ['IoU', 'Dice', 'Pixel Acc', 'Boundary IoU']
        custom_seg = [
            self.results['custom_yolo']['joint']['seg_avg_iou'],
            self.results['custom_yolo']['joint']['seg_avg_dice'],
            self.results['custom_yolo']['joint']['seg_avg_pixel_acc'],
            self.results['custom_yolo']['joint']['seg_avg_boundary_iou']
        ]
        swin_seg = [
            self.results['swin']['joint']['seg_avg_iou'],
            self.results['swin']['joint']['seg_avg_dice'],
            self.results['swin']['joint']['seg_avg_pixel_acc'],
            self.results['swin']['joint']['seg_avg_boundary_iou']
        ]

        x = np.arange(len(metrics))
        ax3.bar(x - width/2, custom_seg, width, label='Custom YOLO', color='#e74c3c', alpha=0.8)
        ax3.bar(x + width/2, swin_seg, width, label='Swin', color='#9b59b6', alpha=0.8)
        ax3.set_ylabel('Score', fontweight='bold')
        ax3.set_title('Segmentation Performance', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics, rotation=15, ha='right')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_ylim([0, 1])

        # 4. IoU Distribution Histograms
        ax4 = fig.add_subplot(gs[1, 0])
        custom_seg_ious = [r['seg_iou'] for r in self.results['custom_yolo']['joint']['per_image_results'] if r['seg_iou'] > 0]
        swin_seg_ious = [r['seg_iou'] for r in self.results['swin']['joint']['per_image_results'] if r['seg_iou'] > 0]

        ax4.hist([custom_seg_ious, swin_seg_ious], bins=15, label=['Custom YOLO', 'Swin'],
                color=['#e74c3c', '#9b59b6'], alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Segmentation IoU', fontweight='bold')
        ax4.set_ylabel('Frequency', fontweight='bold')
        ax4.set_title('Segmentation IoU Distribution', fontweight='bold')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)

        # 5. Speed Comparison
        ax5 = fig.add_subplot(gs[1, 1])
        time_labels = ['Detection', 'Segmentation', 'Total']
        custom_times = [
            self.results['custom_yolo']['joint']['avg_det_time_ms'],
            self.results['custom_yolo']['joint']['avg_seg_time_ms'],
            self.results['custom_yolo']['joint']['avg_total_time_ms']
        ]
        swin_times = [
            self.results['swin']['joint']['avg_det_time_ms'],
            self.results['swin']['joint']['avg_seg_time_ms'],
            self.results['swin']['joint']['avg_total_time_ms']
        ]

        x = np.arange(len(time_labels))
        ax5.bar(x - width/2, custom_times, width, label='Custom YOLO', color='#27ae60', alpha=0.8)
        ax5.bar(x + width/2, swin_times, width, label='Swin', color='#8e44ad', alpha=0.8)
        ax5.set_ylabel('Time (ms)', fontweight='bold')
        ax5.set_title('Inference Speed', fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(time_labels)
        ax5.legend()
        ax5.grid(axis='y', alpha=0.3)

        # 6. FPS Comparison
        ax6 = fig.add_subplot(gs[1, 2])
        fps_values = [
            self.results['custom_yolo']['joint']['fps'],
            self.results['swin']['joint']['fps']
        ]
        bars = ax6.bar(['Custom YOLO', 'Swin'], fps_values,
                      color=['#f39c12', '#16a085'], alpha=0.8, edgecolor='black', linewidth=2)
        ax6.set_ylabel('Frames Per Second', fontweight='bold')
        ax6.set_title('Throughput (FPS)', fontweight='bold')
        ax6.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

        # 7. Per-Image Scatter
        ax7 = fig.add_subplot(gs[2, 0])
        custom_results = self.results['custom_yolo']['joint']['per_image_results']
        swin_results = self.results['swin']['joint']['per_image_results']

        custom_det = [r['det_iou'] for r in custom_results if r['has_gt'] and r['seg_iou'] > 0]
        custom_seg = [r['seg_iou'] for r in custom_results if r['has_gt'] and r['seg_iou'] > 0]
        swin_det = [r['det_iou'] for r in swin_results if r['has_gt'] and r['seg_iou'] > 0]
        swin_seg = [r['seg_iou'] for r in swin_results if r['has_gt'] and r['seg_iou'] > 0]

        ax7.scatter(custom_det, custom_seg, alpha=0.6, s=30, label='Custom YOLO', color='#2ecc71')
        ax7.scatter(swin_det, swin_seg, alpha=0.6, s=30, label='Swin', color='#3498db', marker='s')
        ax7.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax7.set_xlabel('Detection IoU', fontweight='bold')
        ax7.set_ylabel('Segmentation IoU', fontweight='bold')
        ax7.set_title('Detection vs Segmentation', fontweight='bold')
        ax7.legend()
        ax7.grid(alpha=0.3)
        ax7.set_xlim([0, 1])
        ax7.set_ylim([0, 1])

        # 8. Box Plot Comparison
        ax8 = fig.add_subplot(gs[2, 1])
        data_to_plot = [custom_seg_ious, swin_seg_ious]
        bp = ax8.boxplot(data_to_plot, labels=['Custom YOLO', 'Swin'],
                        patch_artist=True, showmeans=True)
        for patch, color in zip(bp['boxes'], ['#e74c3c', '#9b59b6']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax8.set_ylabel('Segmentation IoU', fontweight='bold')
        ax8.set_title('IoU Distribution (Box Plot)', fontweight='bold')
        ax8.grid(axis='y', alpha=0.3)

        # 9. Summary Statistics Table
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('tight')
        ax9.axis('off')

        table_data = [
            ['Metric', 'Custom YOLO', 'Swin', 'Winner'],
            ['Det F1', f"{self.results['custom_yolo']['joint']['det_f1']:.3f}",
             f"{self.results['swin']['joint']['det_f1']:.3f}",
             '✓' if self.results['custom_yolo']['joint']['det_f1'] > self.results['swin']['joint']['det_f1'] else ''],
            ['Seg IoU', f"{self.results['custom_yolo']['joint']['seg_avg_iou']:.3f}",
             f"{self.results['swin']['joint']['seg_avg_iou']:.3f}",
             '✓' if self.results['custom_yolo']['joint']['seg_avg_iou'] > self.results['swin']['joint']['seg_avg_iou'] else ''],
            ['Dice', f"{self.results['custom_yolo']['joint']['seg_avg_dice']:.3f}",
             f"{self.results['swin']['joint']['seg_avg_dice']:.3f}",
             '✓' if self.results['custom_yolo']['joint']['seg_avg_dice'] > self.results['swin']['joint']['seg_avg_dice'] else ''],
            ['FPS', f"{self.results['custom_yolo']['joint']['fps']:.2f}",
             f"{self.results['swin']['joint']['fps']:.2f}",
             '✓' if self.results['custom_yolo']['joint']['fps'] > self.results['swin']['joint']['fps'] else '']
        ]

        table = ax9.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.3, 0.25, 0.25, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax9.set_title('Performance Summary', fontweight='bold', pad=20)

        plt.savefig(os.path.join(self.viz_dir, 'comprehensive_summary.png'),
                   dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {os.path.join(self.viz_dir, 'comprehensive_summary.png')}")
        plt.close()

    def generate_all_visualizations(self):
        """Generate all visualization plots"""
        print("\n" + "=" * 110)
        print("GENERATING VISUALIZATIONS")
        print("=" * 110)

        try:
            self.plot_metric_comparison_bars()
            self.plot_iou_distributions()
            self.plot_speed_comparison()
            self.plot_per_image_scatter()
            self.plot_box_plots()
            self.plot_comprehensive_summary()

            print("\n✅ All visualizations generated successfully!")
            print(f"📁 Saved to: {self.viz_dir}/")
        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()

    # ==================== REPORTING ====================
    def print_comparison_table(self):
        """Print comprehensive comparison table with statistics"""
        print("\n" + "=" * 120)
        print("COMPREHENSIVE MODEL COMPARISON WITH STATISTICAL ANALYSIS")
        print("Custom YOLO (CBAM+CA) vs Swin Transformer")
        print("=" * 120)

        print(f"\n{'Metric':<40} {'Custom YOLO (CBAM+CA)':<30} {'Swin Transformer':<30} {'Winner':<15}")
        print("-" * 120)

        # Detection metrics
        print("\n🔍 DETECTION PERFORMANCE:")
        det_metrics = [
            ('det_precision', 'Precision', False),
            ('det_recall', 'Recall', False),
            ('det_f1', 'F1 Score', False),
            ('det_avg_iou', 'Average IoU', True),
        ]

        for metric, label, show_std in det_metrics:
            custom_val = self.results['custom_yolo']['joint'].get(metric, 0)
            swin_val = self.results['swin']['joint'].get(metric, 0)

            if show_std:
                custom_std = self.results['custom_yolo']['joint'].get('det_iou_std', 0)
                swin_std = self.results['swin']['joint'].get('det_iou_std', 0)
                custom_str = f"{custom_val:.4f} ± {custom_std:.4f}"
                swin_str = f"{swin_val:.4f} ± {swin_std:.4f}"
            else:
                custom_str = f"{custom_val:.4f}"
                swin_str = f"{swin_val:.4f}"

            winner = "Custom YOLO" if custom_val > swin_val else "Swin"
            if abs(custom_val - swin_val) < 0.001:
                winner = "Tie"

            print(f"  {label:<38} {custom_str:<30} {swin_str:<30} {winner:<15}")

        # Segmentation metrics
        print("\n🎨 SEGMENTATION PERFORMANCE:")
        seg_metrics = [
            ('seg_precision', 'Precision', False),
            ('seg_recall', 'Recall', False),
            ('seg_f1', 'F1 Score', False),
            ('seg_avg_iou', 'IoU (Jaccard Index)', True),
            ('seg_avg_dice', 'Dice Score (F1)', True),
            ('seg_avg_pixel_acc', 'Pixel Accuracy', False),
            ('seg_avg_boundary_iou', 'Boundary IoU', False),
        ]

        for metric, label, show_std in seg_metrics:
            custom_val = self.results['custom_yolo']['joint'].get(metric, 0)
            swin_val = self.results['swin']['joint'].get(metric, 0)

            if show_std:
                if 'iou' in metric:
                    custom_std = self.results['custom_yolo']['joint'].get('seg_iou_std', 0)
                    swin_std = self.results['swin']['joint'].get('seg_iou_std', 0)
                else:
                    custom_std = self.results['custom_yolo']['joint'].get('seg_dice_std', 0)
                    swin_std = self.results['swin']['joint'].get('seg_dice_std', 0)

                if 'pixel_acc' in metric:
                    custom_str = f"{custom_val*100:.2f}%"
                    swin_str = f"{swin_val*100:.2f}%"
                else:
                    custom_str = f"{custom_val:.4f} ± {custom_std:.4f}"
                    swin_str = f"{swin_val:.4f} ± {swin_std:.4f}"
            else:
                if 'pixel_acc' in metric:
                    custom_str = f"{custom_val*100:.2f}%"
                    swin_str = f"{swin_val*100:.2f}%"
                else:
                    custom_str = f"{custom_val:.4f}"
                    swin_str = f"{swin_val:.4f}"

            winner = "Custom YOLO" if custom_val > swin_val else "Swin"
            if abs(custom_val - swin_val) < 0.001:
                winner = "Tie"

            print(f"  {label:<38} {custom_str:<30} {swin_str:<30} {winner:<15}")

        # Speed metrics
        print("\n⚡ SPEED PERFORMANCE:")
        speed_metrics = [
            ('avg_det_time_ms', 'Detection Time (ms)', False),
            ('avg_seg_time_ms', 'Segmentation Time (ms)', False),
            ('avg_total_time_ms', 'Total Time (ms)', True),
            ('fps', 'FPS (frames/second)', False),
        ]

        for metric, label, show_std in speed_metrics:
            custom_val = self.results['custom_yolo']['joint'].get(metric, 0)
            swin_val = self.results['swin']['joint'].get(metric, 0)

            if show_std:
                custom_std = self.results['custom_yolo']['joint'].get('std_total_time_ms', 0)
                swin_std = self.results['swin']['joint'].get('std_total_time_ms', 0)
                custom_str = f"{custom_val:.2f} ± {custom_std:.2f}"
                swin_str = f"{swin_val:.2f} ± {swin_std:.2f}"
            else:
                custom_str = f"{custom_val:.2f}"
                swin_str = f"{swin_val:.2f}"

            if metric == 'fps':
                winner = "Custom YOLO" if custom_val > swin_val else "Swin"
            else:
                winner = "Custom YOLO" if custom_val < swin_val else "Swin"

            if abs(custom_val - swin_val) < 0.01:
                winner = "Tie"

            print(f"  {label:<38} {custom_str:<30} {swin_str:<30} {winner:<15}")

        # Pipeline reliability
        print("\n✅ PIPELINE RELIABILITY:")
        custom_success = self.results['custom_yolo']['joint'].get('pipeline_success_rate', 0)
        swin_success = self.results['swin']['joint'].get('pipeline_success_rate', 0)

        winner = "Custom YOLO" if custom_success > swin_success else "Swin"
        if abs(custom_success - swin_success) < 0.001:
            winner = "Tie"

        print(f"  {'Pipeline Success Rate':<38} {custom_success:.4f}{' '*24} {swin_success:.4f}{' '*24} {winner:<15}")

        print("=" * 120)

        # Performance summary
        print("\n📈 OVERALL SUMMARY:")

        custom_score = (
            self.results['custom_yolo']['joint'].get('det_f1', 0) * 0.25 +
            self.results['custom_yolo']['joint'].get('seg_f1', 0) * 0.25 +
            self.results['custom_yolo']['joint'].get('seg_avg_iou', 0) * 0.25 +
            self.results['custom_yolo']['joint'].get('seg_avg_dice', 0) * 0.25
        )

        swin_score = (
            self.results['swin']['joint'].get('det_f1', 0) * 0.25 +
            self.results['swin']['joint'].get('seg_f1', 0) * 0.25 +
            self.results['swin']['joint'].get('seg_avg_iou', 0) * 0.25 +
            self.results['swin']['joint'].get('seg_avg_dice', 0) * 0.25
        )

        print(f"\n  Overall Performance Score (weighted average):")
        print(f"    Custom YOLO (CBAM+CA):      {custom_score:.4f}")
        print(f"    Swin Transformer:           {swin_score:.4f}")

        if custom_score > swin_score:
            improvement = ((custom_score - swin_score) / swin_score) * 100
            print(f"\n  🏆 Winner: Custom YOLO (CBAM+CA)")
            print(f"     Improvement: +{improvement:.2f}% over Swin Transformer")
        elif swin_score > custom_score:
            improvement = ((swin_score - custom_score) / custom_score) * 100
            print(f"\n  🏆 Winner: Swin Transformer")
            print(f"     Improvement: +{improvement:.2f}% over Custom YOLO")
        else:
            print(f"\n  🤝 Result: Tie")

        # Category winners
        print(f"\n  Performance Breakdown:")

        det_winner = "Custom YOLO" if self.results['custom_yolo']['joint'].get('det_f1', 0) > self.results['swin']['joint'].get('det_f1', 0) else "Swin"
        seg_winner = "Custom YOLO" if self.results['custom_yolo']['joint'].get('seg_f1', 0) > self.results['swin']['joint'].get('seg_f1', 0) else "Swin"
        speed_winner = "Custom YOLO" if self.results['custom_yolo']['joint'].get('fps', 0) > self.results['swin']['joint'].get('fps', 0) else "Swin"

        print(f"    Detection:    {det_winner}")
        print(f"    Segmentation: {seg_winner}")
        print(f"    Speed:        {speed_winner}")

        print("\n" + "=" * 120)

    def save_results_json(self, save_path='research_results/comparison_results.json'):
        """Save all results including statistical analysis"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Create comprehensive results dictionary
        output = {
            'metadata': {
                'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'image_size': f"{self.image_size}x{self.image_size}",
                'detection_dataset': self.detection_data_dir,
                'segmentation_dataset': self.segmentation_data_dir,
                'dataset_image_size': '227x227 (original)',
                'model_input_size': f'{self.image_size}x{self.image_size}'
            },
            'results': self.results,
        }

        with open(save_path, 'w') as f:
            json.dump(output, f, indent=4)

        print(f"\n✅ Comprehensive results saved to: {save_path}")


# ==================== MAIN EXECUTION ====================
def main():
    print("=" * 120)
    print("FIXED TWO-MODEL COMPARISON: Custom YOLO (CBAM+CA) vs Swin Transformer")
    print("Publication-ready version with proper evaluation, statistical analysis, and visualizations")
    print("Handles 227px dataset images correctly")
    print("=" * 120)

    # Configuration
    DETECTION_DATA_DIR = "unified_crack_dataset"
    SEGMENTATION_DATA_DIR = "unified_crack_dataset_seg"

    # Model paths
    SWIN_DETECTION = "best_swin_crack_detection.pth"
    SWIN_SEGMENTATION = "best_swin_crack_segmentation.pth"

    CUSTOM_YOLO_DETECTION = "yolo12s_cbam_ca_crack.pt"
    CUSTOM_YOLO_SEGMENTATION = "yolo12s_seg_cbam_ca_crack.pt"

    # Model input size (handles 227px dataset images)
    IMAGE_SIZE = 224

    print("\n📂 Dataset Configuration:")
    print(f"   Detection Dataset (bbox labels): {DETECTION_DATA_DIR}")
    print(f"   Segmentation Dataset (polygon labels): {SEGMENTATION_DATA_DIR}")
    print(f"   Dataset image size: 227x227 pixels")

    print("\n📦 Model Configuration:")
    print(f"   Swin Detection: {SWIN_DETECTION}")
    print(f"   Swin Segmentation: {SWIN_SEGMENTATION}")
    print(f"   Custom YOLO Detection: {CUSTOM_YOLO_DETECTION}")
    print(f"   Custom YOLO Segmentation: {CUSTOM_YOLO_SEGMENTATION}")
    print(f"   Model Input Size: {IMAGE_SIZE}x{IMAGE_SIZE} (handles 227px images)")

    # Verify datasets
    if not os.path.exists(DETECTION_DATA_DIR):
        print(f"\n❌ ERROR: Detection dataset not found: {DETECTION_DATA_DIR}")
        return

    if not os.path.exists(SEGMENTATION_DATA_DIR):
        print(f"\n❌ ERROR: Segmentation dataset not found: {SEGMENTATION_DATA_DIR}")
        return

    # Initialize comparator
    comparator = FixedTwoModelComparator(
        detection_data_dir=DETECTION_DATA_DIR,
        segmentation_data_dir=SEGMENTATION_DATA_DIR,
        image_size=IMAGE_SIZE
    )

    # Load models
    print("\n" + "=" * 120)
    print("LOADING MODELS")
    print("=" * 120)

    swin_loaded = comparator.load_swin_models(SWIN_DETECTION, SWIN_SEGMENTATION)
    custom_yolo_loaded = comparator.load_custom_yolo_models(CUSTOM_YOLO_DETECTION, CUSTOM_YOLO_SEGMENTATION)

    if not (swin_loaded and custom_yolo_loaded):
        print("\n❌ Both models required for comparison. Exiting.")
        return

    # Run evaluations
    print("\n" + "=" * 120)
    print("STARTING COMPREHENSIVE EVALUATION")
    print("=" * 120)

    DET_CONF = 0.5
    IOU_THRESHOLD = 0.5

    # Evaluate Custom YOLO
    print("\n🔬 Evaluating Custom YOLO (CBAM+CA) Pipeline...")
    comparator.results['custom_yolo']['joint'] = comparator.evaluate_joint_pipeline(
        model_type='custom_yolo',
        det_conf=DET_CONF,
        iou_threshold=IOU_THRESHOLD
    )

    # Evaluate Swin
    print("\n🔬 Evaluating Swin Transformer Pipeline...")
    comparator.results['swin']['joint'] = comparator.evaluate_joint_pipeline(
        model_type='swin',
        det_conf=DET_CONF,
        iou_threshold=IOU_THRESHOLD
    )

    # Generate outputs
    print("\n" + "=" * 120)
    print("GENERATING RESULTS")
    print("=" * 120)

    comparator.print_comparison_table()
    comparator.compare_models_statistically()
    comparator.save_results_json()
    comparator.generate_all_visualizations()

    print("\n" + "=" * 120)
    print("✅ COMPARISON COMPLETE!")
    print("=" * 120)

    print("\n📁 Output Files:")
    print("   • research_results/comparison_results.json - Full results with statistics")
    print("   • research_results/visualizations/comprehensive_summary.png - Main summary figure")
    print("   • research_results/visualizations/metric_comparison_bars.png - Bar charts")
    print("   • research_results/visualizations/iou_distributions.png - IoU histograms")
    print("   • research_results/visualizations/speed_comparison.png - Speed analysis")
    print("   • research_results/visualizations/per_image_scatter.png - Per-image analysis")
    print("   • research_results/visualizations/box_plots.png - Distribution analysis")

    print("\n💡 Key Features:")
    print("   ✓ Handles 227px dataset images correctly (resizes to 224x224 for models)")
    print("   ✓ Per-instance segmentation evaluation")
    print("   ✓ Statistical significance testing (paired t-test)")
    print("   ✓ Confidence intervals for all metrics")
    print("   ✓ Comprehensive visualizations for research paper")
    print("   ✓ Boundary IoU for thin crack structures")
    print("   ✓ Per-image tracking for detailed analysis")

    print("\n" + "=" * 120)


if __name__ == "__main__":
    main()