"""
Debug script to check what your custom YOLO model actually outputs
Run this to understand the prediction format
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from ultralytics.nn.tasks import DetectionModel
import ultralytics
from common.attention import CBAM, CoordinateAttention, ChannelAttention, SpatialAttention


# Register modules
def register_modules():
    for name, cls in [('CBAM', CBAM), ('CoordinateAttention', CoordinateAttention),
                      ('ChannelAttention', ChannelAttention), ('SpatialAttention', SpatialAttention)]:
        setattr(ultralytics.nn.tasks, name, cls)
        globals()[name] = cls


# ==================== DEBUG FUNCTIONS ====================

def load_model(weights_path, config_path):
    """Load the model"""
    register_modules()

    model = DetectionModel(cfg=config_path, verbose=False)
    ckpt = torch.load(weights_path, map_location='cpu')

    if isinstance(ckpt, dict):
        if 'model' in ckpt:
            state_dict = ckpt['model'].state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt.state_dict() if hasattr(ckpt, 'state_dict') else ckpt

    model.load_state_dict(state_dict, strict=False)
    return model.eval()


def preprocess_image(image_path, imgsz=192):
    """Preprocess image"""
    img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (imgsz, imgsz))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return img_tensor, img.shape[:2]


def flatten_structure(obj, depth=0, max_depth=5):
    """Recursively flatten nested structure to find all tensors"""
    if depth > max_depth:
        return []

    tensors = []
    if isinstance(obj, torch.Tensor):
        tensors.append(obj)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            tensors.extend(flatten_structure(item, depth + 1, max_depth))
    elif isinstance(obj, dict):
        for value in obj.values():
            tensors.extend(flatten_structure(value, depth + 1, max_depth))

    return tensors


def analyze_predictions(pred, conf_threshold=0.25):
    """Analyze what the model outputs"""
    print("\n" + "=" * 80)
    print("MODEL OUTPUT ANALYSIS")
    print("=" * 80)

    # Check type
    print(f"\nOutput type: {type(pred)}")

    # Flatten to find all tensors
    all_tensors = flatten_structure(pred)
    print(f"\nFound {len(all_tensors)} tensor(s) in nested structure")

    # Analyze each tensor
    for i, tensor in enumerate(all_tensors):
        print(f"\n--- Tensor {i} ---")
        print(f"  Shape: {tensor.shape}")
        print(f"  Device: {tensor.device}")
        print(f"  Dtype: {tensor.dtype}")
        print(f"  Min value: {tensor.min().item():.4f}")
        print(f"  Max value: {tensor.max().item():.4f}")
        print(f"  Mean value: {tensor.mean().item():.4f}")

        # Show first few values if appropriate
        if len(tensor.shape) >= 2 and tensor.shape[0] > 0:
            sample_size = min(3, tensor.shape[0])
            print(f"  First {sample_size} sample(s):")
            for j in range(sample_size):
                if tensor.shape[1] <= 10:
                    print(f"    [{j}]: {tensor[j].tolist()}")
                else:
                    print(f"    [{j}]: {tensor[j, :10].tolist()}... (truncated)")

    print("\n" + "=" * 80)
    print("DETECTION EXTRACTION ATTEMPT")
    print("=" * 80)

    # Try to extract detections from each tensor
    all_detections = []

    for i, p in enumerate(all_tensors):
        print(f"\n--- Processing tensor {i} ---")

        # Handle different shapes
        original_shape = p.shape
        print(f"Original shape: {original_shape}")

        # Remove batch dimension if present
        if len(p.shape) == 3 and p.shape[0] == 1:
            p = p[0]
            print(f"After batch removal: {p.shape}")
        elif len(p.shape) == 4:
            # Reshape from (batch, anchors, h, w, features) to (batch*anchors*h*w, features)
            p = p.reshape(-1, p.shape[-1])
            print(f"After reshaping 4D: {p.shape}")

        # Check if this looks like detection output
        if len(p.shape) == 2 and p.shape[1] >= 5:
            print(f"✓ Looks like detection format: {p.shape[0]} predictions, {p.shape[1]} features")

            # Analyze columns
            print(f"\nColumn analysis:")
            for col_idx in range(min(10, p.shape[1])):
                col_data = p[:, col_idx]
                print(f"  Col {col_idx}: min={col_data.min().item():.4f}, "
                      f"max={col_data.max().item():.4f}, "
                      f"mean={col_data.mean().item():.4f}")

            # Try to identify confidence column (usually column 4)
            if p.shape[1] > 4:
                conf_col = p[:, 4]
                print(f"\nAssuming column 4 is confidence:")
                print(f"  Min: {conf_col.min().item():.4f}")
                print(f"  Max: {conf_col.max().item():.4f}")
                print(f"  Mean: {conf_col.mean().item():.4f}")
                print(f"  Detections > {conf_threshold}: {(conf_col > conf_threshold).sum().item()}")

                # Filter by confidence
                valid = conf_col > conf_threshold
                filtered = p[valid]
                print(f"  After filtering: {len(filtered)} detections")

                if len(filtered) > 0:
                    print(f"\n  Sample detection:")
                    det = filtered[0]
                    print(f"    Bbox (cols 0-3): [{det[0]:.4f}, {det[1]:.4f}, {det[2]:.4f}, {det[3]:.4f}]")
                    print(f"    Conf (col 4): {det[4]:.4f}")
                    if det.shape[0] > 5:
                        print(f"    Additional features: {det[5:min(10, det.shape[0])].tolist()}")

                    all_detections.append(filtered)
            else:
                print(f"⚠ Only {p.shape[1]} columns - expected at least 5 (xywh + conf)")
        else:
            print(f"✗ Shape doesn't match detection format: {p.shape}")

    print("\n" + "=" * 80)
    print(f"TOTAL DETECTIONS FOUND: {sum(len(d) for d in all_detections)}")
    print("=" * 80)

    return all_detections


def main():
    # UPDATE THESE PATHS TO MATCH YOUR SETUP
    WEIGHTS_PATH = "runs/detect/yolo12_cbam_ca_crack/weights/best.pt"
    CONFIG_PATH = "models/yolo12_cbam_ca.yaml"
    TEST_IMAGE = "crack_detection_dataset/images/test/test_00001.jpg"  # Or any test image

    print("=" * 80)
    print("CUSTOM YOLO MODEL DEBUG")
    print("=" * 80)

    # Check files exist
    import os
    for path, name in [(WEIGHTS_PATH, "Weights"), (CONFIG_PATH, "Config"), (TEST_IMAGE, "Test image")]:
        if os.path.exists(path):
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name} NOT FOUND: {path}")
            print("\nPlease update the paths at the top of this script!")
            return

    # Load model
    print("\nLoading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(WEIGHTS_PATH, CONFIG_PATH).to(device)
    print(f"✓ Model loaded on {device}")

    # Preprocess image
    print(f"\nPreprocessing image...")
    img_tensor, orig_shape = preprocess_image(TEST_IMAGE, imgsz=192)
    img_tensor = img_tensor.to(device)
    print(f"✓ Image shape: {img_tensor.shape}, Original: {orig_shape}")

    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        pred = model(img_tensor)
    print("✓ Inference complete")

    # Analyze output
    detections = analyze_predictions(pred, conf_threshold=0.25)

    # Additional check: Compare with normal YOLO inference
    print("\n" + "=" * 80)
    print("COMPARISON WITH STANDARD YOLO INFERENCE")
    print("=" * 80)

    try:
        from ultralytics import YOLO
        yolo_model = YOLO(WEIGHTS_PATH)
        yolo_results = yolo_model(TEST_IMAGE, conf=0.25, verbose=False)

        if len(yolo_results) > 0 and yolo_results[0].boxes is not None:
            num_dets = len(yolo_results[0].boxes)
            print(f"✓ Standard YOLO found {num_dets} detections")

            if num_dets > 0:
                boxes = yolo_results[0].boxes
                print("\nSample detection from standard YOLO:")
                print(f"  xywh (normalized): {boxes.xywhn[0].tolist()}")
                print(f"  confidence: {boxes.conf[0].item():.4f}")
        else:
            print("✗ Standard YOLO found no detections")
    except Exception as e:
        print(f"✗ Could not run standard YOLO: {e}")

    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)
    print("\nWhat to look for:")
    print("1. Are bbox coordinates in range [0, 1]? (normalized)")
    print("2. Are confidence values reasonable? (0-1 range)")
    print("3. Does the number of detections match standard YOLO?")
    print("4. What is the shape of each detection head output?")
    print("5. Which column contains the confidence scores?")


if __name__ == "__main__":
    main()