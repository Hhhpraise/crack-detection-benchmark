"""
FIXED SWIN TRANSFORMER COMPARISON SCRIPT
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

try:
    from transformers import SwinConfig, SwinModel
except ImportError:
    print("Please install transformers: pip install transformers")
    exit()

try:
    from ultralytics import YOLO
    import ultralytics

    YOLO_AVAILABLE = True
    print(f"Ultralytics version: {ultralytics.__version__}")
except ImportError:
    print("YOLO not available. Install with: pip install ultralytics")
    YOLO_AVAILABLE = False


# ==================== SWIN TRANSFORMER MODEL (FIXED) ====================
class SwinDetectionModel(nn.Module):
    def __init__(self, num_classes=1, hidden_dim=256, max_detections=10, backbone_pretrained=True):
        super().__init__()
        self.max_detections = max_detections

        try:
            if backbone_pretrained:
                self.backbone = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
            else:
                raise Exception("Using random initialization")
        except Exception as e:
            print(f"Could not load pretrained model: {e}")
            self.backbone = SwinModel(config=SwinConfig(
                image_size=224, patch_size=4, num_channels=3, embed_dim=96,
                depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7
            ))

        # FIXED: Use the same architecture as your training script
        self.bbox_head = nn.Sequential(
            nn.Linear(768, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, 4 * max_detections)
        )
        self.objectness_head = nn.Sequential(
            nn.Linear(768, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_detections)
        )
        self.class_head = nn.Sequential(
            nn.Linear(768, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        features = self.backbone(x).last_hidden_state
        features = features.mean(dim=1)  # Global average pooling

        bbox_pred = self.bbox_head(features).view(-1, self.max_detections, 4)
        objectness_pred = self.objectness_head(features)
        class_logits = self.class_head(features)

        # Apply sigmoid like in training
        bbox_pred = torch.sigmoid(bbox_pred)
        objectness_probs = torch.sigmoid(objectness_pred)
        class_probs = torch.sigmoid(class_logits)

        return bbox_pred, objectness_probs, class_probs


class FixedModelComparator:
    def __init__(self, data_dir, image_size=224):
        self.data_dir = data_dir
        self.image_size = image_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        with open(os.path.join(data_dir, 'annotations', 'dataset_info.json'), 'r') as f:
            self.dataset_info = json.load(f)

        self.swin_model = None
        self.yolo_models = {}

    def load_swin_model(self, checkpoint_path):
        """Load Swin Transformer model with proper error handling"""
        print(f"\nLoading Swin model from {checkpoint_path}...")

        # First try to load the model architecture from checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Check if it's a full model or just state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # Create model with same architecture as training
                self.swin_model = SwinDetectionModel(
                    num_classes=1,
                    hidden_dim=256,
                    max_detections=10,
                    backbone_pretrained=False  # We'll load the trained weights
                )
                self.swin_model.load_state_dict(state_dict)
                print("✓ Swin model loaded from model_state_dict")
            else:
                # Assume it's a direct state dict
                self.swin_model = SwinDetectionModel(
                    num_classes=1,
                    hidden_dim=256,
                    max_detections=10,
                    backbone_pretrained=False
                )
                self.swin_model.load_state_dict(checkpoint)
                print("✓ Swin model loaded from direct state_dict")

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying alternative loading method...")

            # Alternative loading method
            try:
                self.swin_model = SwinDetectionModel(
                    num_classes=1,
                    hidden_dim=256,
                    max_detections=10,
                    backbone_pretrained=False
                )
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.swin_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.swin_model.load_state_dict(checkpoint)
                print("✓ Swin model loaded with alternative method")
            except Exception as e2:
                print(f"Failed to load Swin model: {e2}")
                return False

        self.swin_model = self.swin_model.to(self.device).eval()
        print("✓ Swin model loaded and set to eval mode")
        return True

    def load_yolo_model(self, model_path, model_name, imgsz=192):
        """Load YOLO model using standard API"""
        if not YOLO_AVAILABLE:
            return False

        print(f"\nLoading YOLO model: {model_name}")
        try:
            model = YOLO(model_path)
            self.yolo_models[model_name] = {
                'model': model,
                'imgsz': imgsz
            }
            print(f"✓ {model_name} loaded (standard API)")
            return True
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {e}")
            return False

    def preprocess_image_swin(self, image_path):
        """Preprocess for Swin - matching training preprocessing"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Use the same transforms as in training
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return transform(image).unsqueeze(0), image

    def predict_swin(self, image_path, conf_threshold=0.3):  # Lower confidence threshold
        """Fixed Swin inference"""
        try:
            img_tensor, orig_img = self.preprocess_image_swin(image_path)
            img_tensor = img_tensor.to(self.device)

            with torch.no_grad():
                start_time = time.time()
                bbox_pred, objectness_probs, class_probs = self.swin_model(img_tensor)
                inference_time = time.time() - start_time

            detections = []
            for j in range(self.swin_model.max_detections):
                confidence = objectness_probs[0, j].item()
                if confidence > conf_threshold:
                    bbox = bbox_pred[0, j].cpu().numpy()

                    # The model outputs normalized coordinates [0, 1]
                    # Assuming it's already in the correct format
                    x1, y1, x2, y2 = bbox

                    # Clamp to [0, 1]
                    x1 = np.clip(x1, 0, 1)
                    y1 = np.clip(y1, 0, 1)
                    x2 = np.clip(x2, 0, 1)
                    y2 = np.clip(y2, 0, 1)

                    # Only add if box has reasonable size
                    if (x2 - x1) > 0.05 and (y2 - y1) > 0.05:  # At least 5% of image size
                        detections.append({
                            'bbox': np.array([x1, y1, x2, y2]),
                            'confidence': confidence,
                            'class': 0
                        })

            return {
                'detections': detections,
                'inference_time': inference_time,
                'has_crack_prob': class_probs[0].item()
            }

        except Exception as e:
            print(f"Error in Swin prediction: {e}")
            return {
                'detections': [],
                'inference_time': 0,
                'has_crack_prob': 0
            }

    def predict_yolo(self, model_name, image_path, conf_threshold=0.25):
        """YOLO inference using standard API"""
        if model_name not in self.yolo_models:
            return None

        model_info = self.yolo_models[model_name]
        model = model_info['model']
        imgsz = model_info['imgsz']

        try:
            start_time = time.time()
            results = model.predict(
                source=image_path,
                conf=conf_threshold,
                save=False,
                imgsz=imgsz,
                verbose=False
            )
            inference_time = time.time() - start_time

            detections = []
            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                orig_img = cv2.imread(image_path)
                img_h, img_w = orig_img.shape[:2]

                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1 = xyxy[0] / img_w
                    y1 = xyxy[1] / img_h
                    x2 = xyxy[2] / img_w
                    y2 = xyxy[3] / img_h

                    detections.append({
                        'bbox': np.array([x1, y1, x2, y2]),
                        'confidence': boxes.conf[i].item(),
                        'class': int(boxes.cls[i].item()) if hasattr(boxes, 'cls') else 0
                    })

            return {'detections': detections, 'inference_time': inference_time}

        except Exception as e:
            print(f"Error in YOLO prediction: {e}")
            return {'detections': [], 'inference_time': 0}

    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes (normalized coordinates)"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def evaluate_model(self, model_type, model_name=None, conf_threshold=0.3, iou_threshold=0.5):
        """Evaluate model on test set"""
        test_images_dir = os.path.join(self.data_dir, 'images/test')
        test_labels_dir = os.path.join(self.data_dir, 'labels/test')

        image_files = [f for f in os.listdir(test_images_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'total_images': len(image_files),
            'inference_times': [],
            'ious': []
        }

        print(f"\nEvaluating {model_type}{' - ' + model_name if model_name else ''}...")

        for img_file in tqdm(image_files, desc=f"Testing {model_type}"):
            img_path = os.path.join(test_images_dir, img_file)
            label_path = os.path.join(test_labels_dir, os.path.splitext(img_file)[0] + '.txt')

            # Load ground truth
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

            # Get predictions
            if model_type == 'swin':
                results = self.predict_swin(img_path, conf_threshold)
            else:
                results = self.predict_yolo(model_name, img_path, conf_threshold)

            if results is None:
                continue

            metrics['inference_times'].append(results['inference_time'])
            pred_boxes = [det['bbox'] for det in results['detections']]

            # Match predictions to ground truth
            matched_gt = set()
            for pred_box in pred_boxes:
                best_iou = 0
                best_gt_idx = -1

                for j, gt_box in enumerate(gt_boxes):
                    if j not in matched_gt:
                        iou = self.calculate_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = j

                if best_iou >= iou_threshold:
                    metrics['true_positives'] += 1
                    metrics['ious'].append(best_iou)
                    matched_gt.add(best_gt_idx)
                else:
                    metrics['false_positives'] += 1

            metrics['false_negatives'] += len(gt_boxes) - len(matched_gt)

        # Calculate metrics - FIXED VERSION
        tp, fp, fn = metrics['true_positives'], metrics['false_positives'], metrics['false_negatives']
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Fixed F1-score calculation
        if (metrics['precision'] + metrics['recall']) > 0:
            metrics['f1_score'] = (2 * metrics['precision'] * metrics['recall']) / (
                        metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0

        metrics['avg_iou'] = np.mean(metrics['ious']) if metrics['ious'] else 0
        metrics['avg_inference_time_ms'] = np.mean(metrics['inference_times']) * 1000

        return metrics

    def print_comparison_table(self, results):
        """Print comparison table"""
        print("\n" + "=" * 100)
        print("MODEL COMPARISON RESULTS")
        print("=" * 100)
        print(f"\n{'Model':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} "
              f"{'Avg IoU':<12} {'Time (ms)':<12}")
        print("-" * 100)

        for model_name, metrics in results.items():
            print(f"{model_name:<25} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                  f"{metrics['f1_score']:<12.4f} {metrics['avg_iou']:<12.4f} "
                  f"{metrics['avg_inference_time_ms']:<12.2f}")

        print("=" * 100)

    def visualize_predictions(self, image_path, save_path='comparison_visualization.png'):
        """Visualize predictions from all models"""
        original_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        h, w = original_img.shape[:2]

        models_to_visualize = []

        # Swin predictions with lower confidence threshold
        if self.swin_model:
            swin_results = self.predict_swin(image_path, 0.3)  # Lower threshold for visualization
            models_to_visualize.append(('Swin Transformer', swin_results))

        # YOLO predictions
        for model_name in self.yolo_models:
            yolo_results = self.predict_yolo(model_name, image_path, 0.25)
            if yolo_results:
                models_to_visualize.append((model_name, yolo_results))

        num_models = len(models_to_visualize) + 1
        fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 5))
        if num_models == 1:
            axes = [axes]

        # Original image
        axes[0].imshow(original_img)
        axes[0].set_title('Original', fontsize=10)
        axes[0].axis('off')

        # Model predictions
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        for idx, (model_name, results) in enumerate(models_to_visualize, 1):
            img_copy = original_img.copy()
            color = colors[idx % len(colors)]

            for det in results['detections']:
                bbox = det['bbox']
                x1 = int(bbox[0] * w)
                y1 = int(bbox[1] * h)
                x2 = int(bbox[2] * w)
                y2 = int(bbox[3] * h)

                cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_copy, f'{det["confidence"]:.2f}', (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            axes[idx].imshow(img_copy)
            axes[idx].set_title(
                f"{model_name}\n{len(results['detections'])} dets, "
                f"{results['inference_time'] * 1000:.1f}ms",
                fontsize=10
            )
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved visualization: {save_path}")


def main():
    # ==================== CONFIGURATION ====================
    DATA_DIR = "crack_detection_dataset"
    SWIN_CHECKPOINT = "best_swin_crack_detection.pth"

    # YOLO models configuration
    YOLO_MODELS = [
        {
            'name': 'YOLO12s-CBAM-CA',
            'path': 'yolo12s_cbam_ca_crack.pt',
            'imgsz': 192
        },
    ]

    print("=" * 100)
    print("FIXED CRACK DETECTION MODEL COMPARISON")
    print("=" * 100)

    comparator = FixedModelComparator(DATA_DIR, image_size=224)

    # Load Swin model with better error handling
    if os.path.exists(SWIN_CHECKPOINT):
        success = comparator.load_swin_model(SWIN_CHECKPOINT)
        if not success:
            print("Failed to load Swin model. Check the checkpoint file.")
    else:
        print(f"Warning: Swin checkpoint not found: {SWIN_CHECKPOINT}")

    # Load YOLO models
    for model_info in YOLO_MODELS:
        if os.path.exists(model_info['path']):
            comparator.load_yolo_model(
                model_info['path'],
                model_info['name'],
                imgsz=model_info.get('imgsz', 640)
            )
        else:
            print(f"Warning: {model_info['path']} not found")

    if not comparator.swin_model and not comparator.yolo_models:
        print("\nError: No models loaded!")
        return

    # Evaluate all models with adjusted confidence thresholds
    print("\n" + "=" * 100)
    print("STARTING EVALUATION")
    print("=" * 100)

    results = {}

    # Evaluate Swin with lower confidence threshold
    if comparator.swin_model:
        results['Swin Transformer'] = comparator.evaluate_model(
            'swin',
            conf_threshold=0.3,  # Lower threshold for Swin
            iou_threshold=0.5
        )

    # Evaluate YOLO models
    for model_name in comparator.yolo_models:
        results[model_name] = comparator.evaluate_model(
            'yolo',
            model_name,
            conf_threshold=0.25,
            iou_threshold=0.5
        )

    # Print and save results
    if results:
        comparator.print_comparison_table(results)

        # Save to JSON
        results_json = {
            name: {
                k: float(v) if isinstance(v, (np.floating, np.integer))
                else int(v) if isinstance(v, (int, np.int64))
                else v
                for k, v in metrics.items()
                if k not in ['inference_times', 'ious']
            }
            for name, metrics in results.items()
        }

        with open('fixed_comparison_results.json', 'w') as f:
            json.dump(results_json, f, indent=4)
        print("\n✓ Results saved to fixed_comparison_results.json")

        # Generate visualizations
        test_dir = os.path.join(DATA_DIR, 'images/test')
        if os.path.exists(test_dir):
            samples = [
                os.path.join(test_dir, f)
                for f in sorted(os.listdir(test_dir))[:3]
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]

            print("\nGenerating visualizations...")
            for i, img_path in enumerate(samples, 1):
                try:
                    comparator.visualize_predictions(
                        img_path,
                        f'fixed_comparison_sample_{i}.png'
                    )
                except Exception as e:
                    print(f"Error visualizing {img_path}: {e}")

        print("\n" + "=" * 100)
        print("COMPARISON COMPLETE!")
        print("=" * 100)
    else:
        print("\nNo results to display - check model loading errors above")


if __name__ == "__main__":
    main()