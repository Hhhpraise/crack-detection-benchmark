import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torchvision
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from scipy import ndimage
from skimage.morphology import skeletonize
import pandas as pd
import warnings
import sys
import time

warnings.filterwarnings('ignore')
from PIL import Image

print("=" * 80)
print("CRACK DETECTION INFERENCE SCRIPT")
print("=" * 80)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# ===== CHECK TRANSFORMERS AVAILABILITY =====
TRANSFORMERS_AVAILABLE = False
SWIN_MODELS_AVAILABLE = False

try:
    from transformers import __version__ as transformers_version

    print(f"Transformers version: {transformers_version}")

    # Try to import Swin components
    try:
        from transformers import SwinConfig, SwinModel

        TRANSFORMERS_AVAILABLE = True
        SWIN_MODELS_AVAILABLE = True
        print("✅ Transformers with PyTorch Swin support is available")
    except ImportError as e:
        print(f"⚠️  SwinModel import failed: {e}")
        # Try alternative import
        try:
            # Some versions might have different structure
            import transformers

            if hasattr(transformers, 'SwinModel'):
                TRANSFORMERS_AVAILABLE = True
                SWIN_MODELS_AVAILABLE = True
                print("✅ Found SwinModel in transformers")
        except:
            TRANSFORMERS_AVAILABLE = False
            SWIN_MODELS_AVAILABLE = False
except ImportError:
    print("⚠️  Transformers library not found")
    print("   Install with: pip install transformers")

# ===== USER CONFIGURATION =====
# YOLO Models
YOLO_DETECTION_MODEL = 'yolo12s_cbam_ca_crack.pt'
YOLO_SEGMENTATION_MODEL = 'yolo12s_seg_cbam_ca_crack.pt'

# Swin Transformer Models
SWIN_DETECTION_MODEL = 'best_swin_crack_detection.pth'
SWIN_SEGMENTATION_MODEL = 'best_swin_crack_segmentation.pth'

# Model settings
CONFIDENCE = 0.5
SWIN_DETECTION_CONFIDENCE = 0.5  # Adjusted for Swin
IMG_SIZE = 192  # Swin uses 224x224
USE_YOLO = True
USE_SWIN = SWIN_MODELS_AVAILABLE  # Only use Swin if available
DEBUG_SWIN = True

# Processing settings
SAVE_RESULTS = True
RESULTS_DIR = 'combined_results'
OVERLAY_ALPHA = 0.6
SAVE_SEMANTIC_MASK = True
SAVE_METRICS_CSV = True
SAVE_COMPARISON_PLOTS = True

# Crack measurement parameters
PIXELS_PER_MM = 10

# Batch processing
IMAGE_FOLDER = 'test_images'
BATCH_SIZE = 8


class SwinDetectionModel(torch.nn.Module):
    """Swin Detection Model - MATCHES YOUR TRAINING ARCHITECTURE"""

    def __init__(self, num_classes=1, hidden_dim=256, max_detections=10, backbone_pretrained=True):
        super().__init__()
        self.max_detections = max_detections

        print("  ℹ️  Creating Swin detection model with correct architecture...")

        # Load Swin Transformer backbone
        try:
            if SWIN_MODELS_AVAILABLE and backbone_pretrained:
                from transformers import SwinModel
                self.backbone = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
                print("  ✅ Loaded pretrained Swin-Tiny backbone")
            else:
                raise Exception("Swin not available or pretrained=False")
        except Exception as e:
            print(f"  ⚠️  Could not load pretrained Swin: {e}")
            print("  ⚠️  Using random initialization")
            from transformers import SwinConfig, SwinModel
            self.backbone = SwinModel(config=SwinConfig(
                image_size=224, patch_size=4, num_channels=3, embed_dim=96,
                depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7
            ))

        # CRITICAL: 3-layer heads to match training (not 2-layer!)
        # Hidden dim from Swin-Tiny is 768
        self.bbox_head = torch.nn.Sequential(
            torch.nn.Linear(768, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, hidden_dim),  # EXTRA LAYER - THIS WAS MISSING!
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, 4 * max_detections)
        )

        self.objectness_head = torch.nn.Sequential(
            torch.nn.Linear(768, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, max_detections)
        )

        self.class_head = torch.nn.Sequential(
            torch.nn.Linear(768, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # Extract features from Swin backbone
        features = self.backbone(x).last_hidden_state
        features = features.mean(dim=1)  # Global average pooling

        # Predictions through heads
        bbox_pred = self.bbox_head(features).view(-1, self.max_detections, 4)
        objectness_pred = self.objectness_head(features)
        class_logits = self.class_head(features)

        # Apply sigmoid like in training
        bbox_pred = torch.sigmoid(bbox_pred)
        objectness_probs = torch.sigmoid(objectness_pred)
        class_probs = torch.sigmoid(class_logits)

        return bbox_pred, objectness_probs, class_probs


class SimpleSwinDetectionModel(torch.nn.Module):
    """Simplified fallback model when Swin transformers unavailable"""

    def __init__(self, num_classes=1, hidden_dim=256, max_detections=10):
        super().__init__()
        self.max_detections = max_detections

        print("  ℹ️  Creating simplified detection model (fallback)...")

        # Simple CNN backbone
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten()
        )

        # Match the 3-layer architecture
        self.bbox_head = torch.nn.Sequential(
            torch.nn.Linear(256, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, 4 * max_detections)
        )

        self.objectness_head = torch.nn.Sequential(
            torch.nn.Linear(256, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, max_detections)
        )

        self.class_head = torch.nn.Sequential(
            torch.nn.Linear(256, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.backbone(x)

        bbox_pred = self.bbox_head(features).view(-1, self.max_detections, 4)
        objectness_pred = self.objectness_head(features)
        class_logits = self.class_head(features)

        bbox_pred = torch.sigmoid(bbox_pred)
        objectness_probs = torch.sigmoid(objectness_pred)
        class_probs = torch.sigmoid(class_logits)

        return bbox_pred, objectness_probs, class_probs


class SwinDetector:
    """Wrapper for Swin detection - FIXED VERSION"""

    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.image_size = 224
        self.model_loaded = False
        self.use_simple_model = False
        self.max_detections = 10

        print(f"  ℹ️  Loading Swin detection model from: {model_path}")

        if not os.path.exists(model_path):
            print(f"  ❌ Model file not found: {model_path}")
            self.model = None
            return

        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            # Extract state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("  ✅ Found model_state_dict in checkpoint")
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("  ✅ Found state_dict in checkpoint")
            else:
                state_dict = checkpoint
                print("  ⚠️  Using checkpoint directly as state_dict")

            # Try to load with full Swin Transformer
            if SWIN_MODELS_AVAILABLE:
                try:
                    print("  ℹ️  Attempting to load Swin Transformer model...")

                    self.model = SwinDetectionModel(
                        num_classes=1,
                        hidden_dim=256,
                        max_detections=10,
                        backbone_pretrained=False  # We'll load trained weights
                    )

                    missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

                    if missing_keys:
                        print(f"  ⚠️  Missing keys: {len(missing_keys)} (OK for backbone)")
                    if unexpected_keys:
                        print(f"  ⚠️  Unexpected keys: {len(unexpected_keys)}")

                    print("  ✅ Swin Transformer model loaded successfully")
                    self.use_simple_model = False

                except Exception as e:
                    print(f"  ❌ Could not load Swin Transformer: {e}")
                    print("  ℹ️  Falling back to simple model...")
                    self.use_simple_model = True
            else:
                print("  ⚠️  Swin Transformer library not available")
                print("  ℹ️  Using simple fallback model")
                self.use_simple_model = True

            if self.use_simple_model:
                self.model = SimpleSwinDetectionModel(
                    num_classes=1,
                    hidden_dim=256,
                    max_detections=10
                )

                try:
                    self.model.load_state_dict(state_dict, strict=False)
                    print("  ✅ Loaded weights into simple model")
                except Exception as e:
                    print(f"  ⚠️  Could not load weights: {e}")
                    print("  ⚠️  Using randomly initialized model")

            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            print(f"  ✅ Model ready on {self.device}")

        except Exception as e:
            print(f"  ❌ Critical error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.model_loaded = False

        # Preprocessing transforms
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((self.image_size, self.image_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image, confidence=0.3):
        """Predict using detection model"""
        if not self.model_loaded or self.model is None:
            print("    ⚠️  Swin detection model not loaded. Skipping.")
            return []

        # Prepare image
        if isinstance(image, np.ndarray):
            original_size = image.shape[:2]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            original_size = (image.height, image.width)
            image_rgb = np.array(image)

        # Preprocess
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            start_time = time.time()
            bbox_pred, objectness_probs, class_probs = self.model(input_tensor)
            inference_time = time.time() - start_time

            if DEBUG_SWIN:
                print(f"    [Swin Detection] Inference: {inference_time * 1000:.1f}ms")
                print(f"    [Swin Detection] Has crack prob: {class_probs[0].item():.4f}")
                print(
                    f"    [Swin Detection] Objectness: min={objectness_probs[0].min():.3f}, max={objectness_probs[0].max():.3f}")

        # Extract detections
        detections = []

        for j in range(self.max_detections):
            conf = objectness_probs[0, j].item()

            if conf > confidence:
                bbox = bbox_pred[0, j].cpu().numpy()
                x1, y1, x2, y2 = bbox

                # Clamp to [0, 1]
                x1 = np.clip(x1, 0, 1)
                y1 = np.clip(y1, 0, 1)
                x2 = np.clip(x2, 0, 1)
                y2 = np.clip(y2, 0, 1)

                box_width = x2 - x1
                box_height = y2 - y1

                if DEBUG_SWIN:
                    print(
                        f"    [Det {j}] Conf: {conf:.4f}, Box: [{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}], Size: {box_width:.3f}x{box_height:.3f}")

                # Only keep boxes with minimum size
                if box_width > 0.05 and box_height > 0.05:
                    box_pixels = np.array([
                        x1 * original_size[1],
                        y1 * original_size[0],
                        x2 * original_size[1],
                        y2 * original_size[0]
                    ])

                    detections.append({
                        'box': box_pixels,
                        'score': conf,
                        'class': 0
                    })
                elif DEBUG_SWIN:
                    print(f"    [Det {j}] Rejected: box too small")

        if len(detections) == 0:
            if DEBUG_SWIN:
                print(f"    ⚠️  No valid detections (threshold: {confidence})")
        else:
            print(f"    ✅ Found {len(detections)} detection(s)")

        return detections

# ==================== SWIN SEGMENTATION MODEL ====================
class SimpleSegmentationModel(torch.nn.Module):
    """Simple segmentation model as fallback"""

    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

        # Simple encoder-decoder architecture
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output


class SwinSegmenter:
    """Wrapper for Swin segmentation with fallback"""

    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_loaded = False
        self.use_simple_model = False

        if not os.path.exists(model_path):
            print(f"  ❌ Model file not found: {model_path}")
            self.model = None
            return

        print(f"  ℹ️  Loading segmentation model from: {model_path}")

        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            # First try to import and use SwinUNet
            try:
                from common.swin_seg import SwinUNet

                if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
                    model_config = checkpoint['model_config']
                    self.model = SwinUNet(
                        img_size=model_config.get('img_size', 224),
                        num_classes=model_config.get('num_classes', 2),
                        embed_dim=model_config.get('embed_dim', 96),
                        depths=model_config.get('depths', [2, 2, 6, 2]),
                        num_heads=model_config.get('num_heads', [3, 6, 12, 24]),
                        window_size=model_config.get('window_size', 7)
                    )
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.image_size = model_config.get('img_size', 224)
                    print("  ✅ Loaded SwinUNet from config")
                else:
                    # Try default config
                    self.image_size = 224
                    self.model = SwinUNet(
                        img_size=self.image_size,
                        num_classes=2,
                        embed_dim=96,
                        depths=[2, 2, 6, 2],
                        num_heads=[3, 6, 12, 24],
                        window_size=7
                    )
                    state_dict = checkpoint.get('model_state_dict', checkpoint)
                    self.model.load_state_dict(state_dict, strict=False)
                    print("  ✅ Loaded SwinUNet with default config")

                self.use_simple_model = False

            except ImportError:
                print("  ⚠️  SwinUNet not found, using simple segmentation model")
                self.use_simple_model = True
            except Exception as e:
                print(f"  ⚠️  Could not load SwinUNet: {e}")
                print("  ⚠️  Using simple segmentation model")
                self.use_simple_model = True

            # Use simple model if Swin failed
            if self.use_simple_model:
                self.image_size = 224
                self.model = SimpleSegmentationModel(num_classes=2)

                # Try to load weights
                try:
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    else:
                        self.model.load_state_dict(checkpoint, strict=False)
                    print("  ✅ Loaded weights into simple segmentation model")
                except:
                    print("  ⚠️  Could not load weights, using randomly initialized model")

            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            print(f"  ✅ Segmentation model loaded on {device}")

        except Exception as e:
            print(f"  ❌ Error loading segmentation model: {e}")
            self.model = None
            self.model_loaded = False

        # Preprocessing transforms
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((self.image_size, self.image_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image, confidence=0.5):
        """Predict using segmentation model"""
        if not self.model_loaded or self.model is None:
            print("    ⚠️  Segmentation model not loaded. Skipping.")
            return []

        # Store original size
        if isinstance(image, np.ndarray):
            original_size = image.shape[:2]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            original_size = (image.height, image.width)
            image_rgb = np.array(image)

        # Preprocess
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)

            # Get probabilities and predictions
            if output.shape[1] > 1:  # Multi-class
                probabilities = torch.nn.functional.softmax(output, dim=1)
                predictions = torch.argmax(probabilities, dim=1).squeeze(0).cpu().numpy()
                confidence_map = torch.max(probabilities, dim=1)[0].squeeze(0).cpu().numpy()
            else:  # Binary
                probabilities = torch.sigmoid(output)
                predictions = (probabilities > 0.5).squeeze(0).squeeze(0).cpu().numpy()
                confidence_map = probabilities.squeeze(0).squeeze(0).cpu().numpy()

        # Resize to original size
        segmentation_mask = cv2.resize(
            predictions.astype(np.uint8),
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST
        )

        confidence_resized = cv2.resize(
            confidence_map,
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # Apply confidence threshold
        segmentation_mask[confidence_resized < confidence] = 0

        # Convert to list format
        if np.any(segmentation_mask == 1):
            masks = [{
                'mask': (segmentation_mask == 1).astype(np.uint8),
                'score': float(np.mean(confidence_resized[segmentation_mask == 1]))
            }]
        else:
            masks = []

        return masks


# ==================== HELPER FUNCTIONS ====================
def calculate_crack_dimensions(mask, pixels_per_mm=PIXELS_PER_MM):
    """Calculate crack dimensions including length, width, and area"""
    binary_mask = mask > 0 if mask.dtype != np.bool_ else mask

    area_pixels = np.sum(binary_mask)
    area_mm2 = area_pixels / (pixels_per_mm ** 2)

    try:
        skeleton = skeletonize(binary_mask)
        length_pixels = np.sum(skeleton)
        length_mm = length_pixels / pixels_per_mm
    except:
        length_pixels = area_pixels
        length_mm = length_pixels / pixels_per_mm

    if length_pixels > 0:
        avg_width_pixels = area_pixels / length_pixels
        avg_width_mm = avg_width_pixels / pixels_per_mm
    else:
        avg_width_pixels = 0
        avg_width_mm = 0

    if area_pixels > 0:
        distance_transform = ndimage.distance_transform_edt(binary_mask)
        max_width_pixels = np.max(distance_transform) * 2
        max_width_mm = max_width_pixels / pixels_per_mm
        width_values = distance_transform[binary_mask] * 2
        width_std_mm = np.std(width_values) / pixels_per_mm
    else:
        max_width_pixels = 0
        max_width_mm = 0
        width_std_mm = 0

    return {
        'length_pixels': length_pixels,
        'length_mm': length_mm,
        'avg_width_pixels': avg_width_pixels,
        'avg_width_mm': avg_width_mm,
        'max_width_pixels': max_width_pixels,
        'max_width_mm': max_width_mm,
        'width_std_mm': width_std_mm,
        'area_pixels': area_pixels,
        'area_mm2': area_mm2
    }


def create_semantic_mask(seg_results, image_shape, model_type='yolo'):
    """Create semantic segmentation mask with white cracks on black background"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    if model_type == 'yolo':
        if seg_results and hasattr(seg_results[0], 'masks') and seg_results[0].masks is not None:
            for mask_data in seg_results[0].masks.xy:
                polygon = np.array(mask_data, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [polygon], 255)
    elif model_type == 'swin':
        if seg_results:
            for seg in seg_results:
                mask_resized = cv2.resize(seg['mask'], (image_shape[1], image_shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
                mask[mask_resized > 0] = 255

    return mask


def calculate_crack_metrics(seg_results, image_shape, model_type='yolo'):
    """Calculate crack coverage metrics from segmentation results"""
    if model_type == 'yolo':
        if not seg_results or not hasattr(seg_results[0], 'masks') or seg_results[0].masks is None:
            return {
                'total_crack_pixels': 0,
                'crack_percentage': 0.0,
                'total_image_pixels': image_shape[0] * image_shape[1],
                'num_crack_regions': 0,
                'crack_dimensions': None
            }

        total_pixels = image_shape[0] * image_shape[1]
        combined_mask = np.zeros(image_shape[:2], dtype=np.uint8)

        for mask_poly in seg_results[0].masks.xy:
            polygon = np.array(mask_poly, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(combined_mask, [polygon], 255)

    elif model_type == 'swin':
        if not seg_results:
            return {
                'total_crack_pixels': 0,
                'crack_percentage': 0.0,
                'total_image_pixels': image_shape[0] * image_shape[1],
                'num_crack_regions': 0,
                'crack_dimensions': None
            }

        total_pixels = image_shape[0] * image_shape[1]
        combined_mask = np.zeros(image_shape[:2], dtype=np.uint8)

        for seg in seg_results:
            mask_resized = cv2.resize(seg['mask'], (image_shape[1], image_shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
            combined_mask[mask_resized > 0] = 255

    crack_pixels = np.sum(combined_mask > 0)
    crack_percentage = (crack_pixels / total_pixels) * 100
    crack_dimensions = calculate_crack_dimensions(combined_mask)

    return {
        'total_crack_pixels': int(crack_pixels),
        'crack_percentage': crack_percentage,
        'total_image_pixels': total_pixels,
        'num_crack_regions': len(seg_results) if model_type == 'swin' else (
            len(seg_results[0].masks) if seg_results and hasattr(seg_results[0], 'masks') else 0),
        'crack_dimensions': crack_dimensions
    }


def visualize_model_results(image, det_results, seg_results, model_name, save_path=None):
    """Visualize results from a specific model"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Detection results
    if det_results:
        det_img = image.copy()
        if model_name == 'yolo' and det_results[0].boxes:
            for box in det_results[0].boxes:
                xyxy = box.xyxy[0].tolist()
                conf = box.conf.item()
                cv2.rectangle(det_img, (int(xyxy[0]), int(xyxy[1])),
                              (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(det_img, f'{conf:.2f}', (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        elif model_name == 'swin':
            for det in det_results:
                box = det['box']
                score = det['score']
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(det_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(det_img, f'{score:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        axes[1].imshow(cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB))
        det_count = len(det_results[0].boxes) if model_name == 'yolo' and det_results[0].boxes else len(det_results)
        axes[1].set_title(f'{model_name.upper()} Detection ({det_count} cracks)', fontsize=12, fontweight='bold')
    else:
        axes[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'{model_name.upper()} Detection (No cracks)', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # Segmentation results
    seg_img = image.copy()
    if seg_results:
        overlay = np.zeros_like(image)
        if model_name == 'yolo' and seg_results[0].masks:
            for mask_poly in seg_results[0].masks.xy:
                polygon = np.array(mask_poly, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [polygon], (255, 0, 0))
        elif model_name == 'swin':
            for seg in seg_results:
                mask_resized = cv2.resize(seg['mask'], (image.shape[1], image.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
                overlay[mask_resized > 0] = (0, 0, 255)

        seg_img = cv2.addWeighted(seg_img, 1 - OVERLAY_ALPHA, overlay, OVERLAY_ALPHA, 0)
        mask_count = len(seg_results[0].masks) if model_name == 'yolo' and seg_results[0].masks else len(seg_results)
        axes[2].imshow(cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f'{model_name.upper()} Segmentation ({mask_count} masks)', fontsize=12, fontweight='bold')
    else:
        axes[2].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f'{model_name.upper()} Segmentation (No masks)', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_comparison(image, yolo_det, yolo_seg, swin_det, swin_seg, save_path=None):
    """Create comparison visualization of all models"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # YOLO Detection
    det_img = image.copy()
    if yolo_det and yolo_det[0].boxes:
        for box in yolo_det[0].boxes:
            xyxy = box.xyxy[0].tolist()
            cv2.rectangle(det_img, (int(xyxy[0]), int(xyxy[1])),
                          (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
    axes[0, 1].imshow(cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB))
    det_count = len(yolo_det[0].boxes) if yolo_det and yolo_det[0].boxes else 0
    axes[0, 1].set_title(f'YOLO Detection ({det_count} cracks)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # Swin Detection
    det_img = image.copy()
    swin_det_count = 0
    if swin_det:
        for det in swin_det:
            box = det['box']
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(det_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        swin_det_count = len(swin_det)
    axes[0, 2].imshow(cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f'Swin Detection ({swin_det_count} cracks)', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')

    # YOLO Segmentation
    seg_img = image.copy()
    if yolo_seg and yolo_seg[0].masks:
        overlay = np.zeros_like(image)
        for mask_poly in yolo_seg[0].masks.xy:
            polygon = np.array(mask_poly, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [polygon], (255, 0, 0))
        seg_img = cv2.addWeighted(seg_img, 1 - OVERLAY_ALPHA, overlay, OVERLAY_ALPHA, 0)
    axes[1, 0].imshow(cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB))
    mask_count = len(yolo_seg[0].masks) if yolo_seg and yolo_seg[0].masks else 0
    axes[1, 0].set_title(f'YOLO Segmentation ({mask_count} masks)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # Swin Segmentation
    seg_img = image.copy()
    swin_mask_count = 0
    if swin_seg:
        overlay = np.zeros_like(image)
        for seg in swin_seg:
            mask_resized = cv2.resize(seg['mask'], (image.shape[1], image.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
            overlay[mask_resized > 0] = (0, 0, 255)
        seg_img = cv2.addWeighted(seg_img, 1 - OVERLAY_ALPHA, overlay, OVERLAY_ALPHA, 0)
        swin_mask_count = len(swin_seg)
    axes[1, 1].imshow(cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'Swin Segmentation ({swin_mask_count} masks)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    # Combined masks comparison
    combined_img = image.copy()
    if yolo_seg and yolo_seg[0].masks:
        overlay = np.zeros_like(image)
        for mask_poly in yolo_seg[0].masks.xy:
            polygon = np.array(mask_poly, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [polygon], (255, 0, 0))
        combined_img = cv2.addWeighted(combined_img, 0.7, overlay, 0.3, 0)

    if swin_seg:
        overlay = np.zeros_like(image)
        for seg in swin_seg:
            mask_resized = cv2.resize(seg['mask'], (image.shape[1], image.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
            overlay[mask_resized > 0] = (0, 0, 255)
        combined_img = cv2.addWeighted(combined_img, 0.7, overlay, 0.3, 0)

    axes[1, 2].imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('Combined Masks (Red: YOLO, Blue: Swin)', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def process_single_image(image_path, output_dir):
    """Process a single image with all models"""
    print(f"Processing: {os.path.basename(image_path)}")

    img = cv2.imread(image_path)
    if img is None:
        print(f"  ⚠️ Failed to load image")
        return None

    img_name = os.path.splitext(os.path.basename(image_path))[0]
    img_dir = os.path.join(output_dir, img_name)
    os.makedirs(img_dir, exist_ok=True)

    results = {
        'filename': os.path.basename(image_path),
        'image_size': f"{img.shape[1]}x{img.shape[0]}"
    }

    # Process with YOLO models
    yolo_det_results = None
    yolo_seg_results = None

    if USE_YOLO:
        if os.path.exists(YOLO_DETECTION_MODEL):
            try:
                yolo_det_model = YOLO(YOLO_DETECTION_MODEL)
                yolo_det_results = yolo_det_model.predict(
                    source=img, conf=CONFIDENCE, save=False, imgsz=IMG_SIZE, verbose=False)
                results['yolo_detections'] = len(yolo_det_results[0].boxes) if yolo_det_results and yolo_det_results[
                    0].boxes else 0
            except Exception as e:
                print(f"  ⚠️ YOLO detection error: {e}")
                results['yolo_detections'] = 0

        if os.path.exists(YOLO_SEGMENTATION_MODEL):
            try:
                yolo_seg_model = YOLO(YOLO_SEGMENTATION_MODEL)
                yolo_seg_results = yolo_seg_model.predict(
                    source=img, conf=CONFIDENCE, save=False, imgsz=IMG_SIZE, verbose=False)
                yolo_metrics = calculate_crack_metrics(yolo_seg_results, img.shape, 'yolo')
                results['yolo_mask_regions'] = yolo_metrics['num_crack_regions']
                results['yolo_crack_pixels'] = yolo_metrics['total_crack_pixels']
                results['yolo_coverage_percent'] = yolo_metrics['crack_percentage']
                if yolo_metrics['crack_dimensions']:
                    results['yolo_length_mm'] = yolo_metrics['crack_dimensions']['length_mm']
                    results['yolo_area_mm2'] = yolo_metrics['crack_dimensions']['area_mm2']

                if SAVE_SEMANTIC_MASK:
                    yolo_mask = create_semantic_mask(yolo_seg_results, img.shape, 'yolo')
                    mask_path = os.path.join(img_dir, f"{img_name}_yolo_mask.png")
                    cv2.imwrite(mask_path, yolo_mask)
            except Exception as e:
                print(f"  ⚠️ YOLO segmentation error: {e}")
                results['yolo_mask_regions'] = 0
                results['yolo_crack_pixels'] = 0
                results['yolo_coverage_percent'] = 0

    # Process with Swin models
    swin_det_results = None
    swin_seg_results = None

    if USE_SWIN:
        if os.path.exists(SWIN_DETECTION_MODEL):
            try:
                swin_detector = SwinDetector(SWIN_DETECTION_MODEL)
                swin_det_results = swin_detector.predict(img, SWIN_DETECTION_CONFIDENCE)
                results['swin_detections'] = len(swin_det_results) if swin_det_results else 0
            except Exception as e:
                print(f"  ⚠️ Swin detection error: {e}")
                results['swin_detections'] = 0
                swin_det_results = None

        if os.path.exists(SWIN_SEGMENTATION_MODEL):
            try:
                swin_segmenter = SwinSegmenter(SWIN_SEGMENTATION_MODEL)
                swin_seg_results = swin_segmenter.predict(img, CONFIDENCE)
                swin_metrics = calculate_crack_metrics(swin_seg_results, img.shape, 'swin')
                results['swin_mask_regions'] = swin_metrics['num_crack_regions']
                results['swin_crack_pixels'] = swin_metrics['total_crack_pixels']
                results['swin_coverage_percent'] = swin_metrics['crack_percentage']
                if swin_metrics['crack_dimensions']:
                    results['swin_length_mm'] = swin_metrics['crack_dimensions']['length_mm']
                    results['swin_area_mm2'] = swin_metrics['crack_dimensions']['area_mm2']

                if SAVE_SEMANTIC_MASK:
                    swin_mask = create_semantic_mask(swin_seg_results, img.shape, 'swin')
                    mask_path = os.path.join(img_dir, f"{img_name}_swin_mask.png")
                    cv2.imwrite(mask_path, swin_mask)
            except Exception as e:
                print(f"  ⚠️ Swin segmentation error: {e}")
                results['swin_mask_regions'] = 0
                results['swin_crack_pixels'] = 0
                results['swin_coverage_percent'] = 0
                swin_seg_results = None

    # Save visualizations
    if SAVE_RESULTS:
        if USE_YOLO:
            yolo_viz_path = os.path.join(img_dir, f"{img_name}_yolo_results.png")
            visualize_model_results(img, yolo_det_results, yolo_seg_results, 'yolo', yolo_viz_path)

        if USE_SWIN and (swin_det_results is not None or swin_seg_results is not None):
            swin_viz_path = os.path.join(img_dir, f"{img_name}_swin_results.png")
            visualize_model_results(img, swin_det_results, swin_seg_results, 'swin', swin_viz_path)

        if SAVE_COMPARISON_PLOTS:
            compare_path = os.path.join(img_dir, f"{img_name}_comparison.png")
            visualize_comparison(img, yolo_det_results, yolo_seg_results,
                                 swin_det_results, swin_seg_results, compare_path)

    print(f"  ✅ Processed successfully")
    return results


def batch_process_images(image_folder, output_folder="batch_results"):
    """Process multiple images in batch with all models"""
    print(f"\n{'=' * 60}")
    print("🔄 BATCH PROCESSING WITH YOLO AND SWIN MODELS")
    print(f"{'=' * 60}")

    os.makedirs(output_folder, exist_ok=True)

    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(image_folder)
                   if f.lower().endswith(image_extensions)]

    if not image_files:
        print(f"❌ No images found in {image_folder}")
        return

    print(f"📁 Found {len(image_files)} images to process")
    print(f"🔧 Models enabled: YOLO={USE_YOLO}, Swin={USE_SWIN}")
    print(f"{'=' * 60}")

    all_results = []
    batch_count = 0

    for i in range(0, len(image_files), BATCH_SIZE):
        batch = image_files[i:i + BATCH_SIZE]
        batch_count += 1

        print(f"\n📦 Processing batch {batch_count}: images {i + 1}-{i + len(batch)}")

        for j, img_file in enumerate(batch, 1):
            img_path = os.path.join(image_folder, img_file)
            print(f"  [{j}/{len(batch)}] {img_file}")

            results = process_single_image(img_path, output_folder)
            if results:
                all_results.append(results)

        print(f"  ✅ Batch {batch_count} completed")

    if SAVE_METRICS_CSV and all_results:
        df = pd.DataFrame(all_results)
        summary_path = os.path.join(output_folder, "batch_summary.csv")
        df.to_csv(summary_path, index=False)

        print(f"\n📊 Summary saved to: {summary_path}")

        # Print statistics
        print(f"\n{'=' * 60}")
        print("📈 BATCH PROCESSING SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total images processed: {len(all_results)}")

        if USE_YOLO:
            avg_yolo_det = df['yolo_detections'].mean() if 'yolo_detections' in df.columns else 0
            avg_yolo_cov = df['yolo_coverage_percent'].mean() if 'yolo_coverage_percent' in df.columns else 0
            print(f"YOLO - Avg detections: {avg_yolo_det:.2f}, Avg coverage: {avg_yolo_cov:.2f}%")

        if USE_SWIN:
            avg_swin_det = df['swin_detections'].mean() if 'swin_detections' in df.columns else 0
            avg_swin_cov = df['swin_coverage_percent'].mean() if 'swin_coverage_percent' in df.columns else 0
            print(f"Swin - Avg detections: {avg_swin_det:.2f}, Avg coverage: {avg_swin_cov:.2f}%")

    print(f"\n✅ Batch processing completed!")
    print(f"📁 Results saved to: {os.path.abspath(output_folder)}/")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CRACK DETECTION INFERENCE TOOL")
    print("=" * 80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers available: {TRANSFORMERS_AVAILABLE}")
    print(f"Swin models available: {SWIN_MODELS_AVAILABLE}")
    print(f"Using Swin: {USE_SWIN}")
    print("=" * 80)

    if not SWIN_MODELS_AVAILABLE:
        print("⚠️  Note: Swin Transformer models require PyTorch 2.1+ and transformers library")
        print("   To use Swin models, upgrade PyTorch:")
        print("   pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118")
        print("   pip install transformers[torch]")
        print("\n   Continuing with YOLO models only...")

    if len(sys.argv) > 1:
        if sys.argv[1] == "batch":
            folder = sys.argv[2] if len(sys.argv) > 2 else IMAGE_FOLDER
            batch_process_images(folder, RESULTS_DIR)
        elif sys.argv[1] == "single":
            image_path = sys.argv[2] if len(sys.argv) > 2 else 'test/0209.png'
            output_dir = 'single_image_results'
            os.makedirs(output_dir, exist_ok=True)
            process_single_image(image_path, output_dir)
        elif sys.argv[1] == "help":
            print("\nUsage:")
            print("  python inferencer.py batch [folder_path]")
            print("  python inferencer.py single [image_path]")
            print("  python inferencer.py help")
            print("\nDefault: batch processing of test_images folder")
    else:
        print("\nNo arguments provided. Running batch processing by default...")
        print(f"Image folder: {IMAGE_FOLDER}")
        print(f"Output folder: {RESULTS_DIR}")
        print("\nTo specify options, use:")
        print("  python inferencer.py batch [folder_path]")
        print("  python inferencer.py single [image_path]\n")
        batch_process_images(IMAGE_FOLDER, RESULTS_DIR)