"""
Diagnostic tool to debug Swin detection model
"""
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

try:
    from transformers import SwinConfig, SwinModel

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("ERROR: transformers not installed")
    exit()

from common.swin_detection import SwinDetectionModel


def diagnose_swin_detection(model_path, test_image_path):
    """Comprehensive diagnostic of Swin detection model"""

    print("=" * 80)
    print("SWIN DETECTION MODEL DIAGNOSTIC")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n1. Device: {device}")

    # Load model
    print(f"\n2. Loading model from: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        print(f"   ✓ Checkpoint loaded successfully")
        print(f"   Checkpoint keys: {list(checkpoint.keys())}")

        model = SwinDetectionModel(
            num_classes=1,
            hidden_dim=256,
            max_detections=10,
            backbone_pretrained=False
        )

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   ✓ Loaded from 'model_state_dict'")
        else:
            model.load_state_dict(checkpoint)
            print(f"   ✓ Loaded from direct state_dict")

        model = model.to(device).eval()
        print(f"   ✓ Model loaded and set to eval mode")

    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        return

    # Load and preprocess image
    print(f"\n3. Loading test image: {test_image_path}")
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"   ✗ Could not load image")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"   ✓ Image loaded, shape: {image_rgb.shape}")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image_rgb).unsqueeze(0).to(device)
    print(f"   ✓ Image preprocessed, tensor shape: {input_tensor.shape}")

    # Run inference
    print(f"\n4. Running inference...")
    with torch.no_grad():
        bbox_pred, objectness_probs, class_probs = model(input_tensor)

    print(f"   ✓ Inference complete")
    print(f"   bbox_pred shape: {bbox_pred.shape}")
    print(f"   objectness_probs shape: {objectness_probs.shape}")
    print(f"   class_probs shape: {class_probs.shape}")

    # Analyze outputs
    print(f"\n5. Output Analysis:")
    print(f"   Class probability (has crack): {class_probs[0].item():.6f}")

    objectness = objectness_probs[0].cpu().numpy()
    bboxes = bbox_pred[0].cpu().numpy()

    print(f"\n   Objectness scores (all {len(objectness)} slots):")
    print(f"   Min: {objectness.min():.6f}")
    print(f"   Max: {objectness.max():.6f}")
    print(f"   Mean: {objectness.mean():.6f}")
    print(f"   Std: {objectness.std():.6f}")

    # Show top detections
    top_indices = np.argsort(objectness)[::-1][:5]
    print(f"\n   Top 5 detection slots:")
    print(f"   {'Rank':<6} {'Slot':<6} {'Confidence':<12} {'BBox (x1,y1,x2,y2)'}")
    print(f"   " + "-" * 60)

    for rank, idx in enumerate(top_indices, 1):
        conf = objectness[idx]
        bbox = bboxes[idx]
        print(f"   {rank:<6} {idx:<6} {conf:<12.6f} [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]")

    # Count detections at different thresholds
    print(f"\n6. Detections at different confidence thresholds:")
    for threshold in [0.05, 0.1, 0.2, 0.3, 0.5]:
        count = np.sum(objectness > threshold)
        print(f"   Threshold {threshold:.2f}: {count} detections")

    # Visualize top detections
    print(f"\n7. Generating visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Detections at different thresholds
    thresholds = [0.05, 0.1, 0.2, 0.3, 0.5]
    for i, threshold in enumerate(thresholds):
        row = (i + 1) // 3
        col = (i + 1) % 3

        img_copy = image_rgb.copy()
        h, w = img_copy.shape[:2]

        count = 0
        for j in range(len(objectness)):
            if objectness[j] > threshold:
                bbox = bboxes[j]
                x1, y1, x2, y2 = bbox

                # Convert to pixels
                x1_px = int(x1 * w)
                y1_px = int(y1 * h)
                x2_px = int(x2 * w)
                y2_px = int(y2 * h)

                cv2.rectangle(img_copy, (x1_px, y1_px), (x2_px, y2_px), (255, 0, 0), 2)
                cv2.putText(img_copy, f'{objectness[j]:.2f}', (x1_px, y1_px - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                count += 1

        axes[row, col].imshow(img_copy)
        axes[row, col].set_title(f'Threshold {threshold:.2f} ({count} dets)')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig('swin_detection_diagnostic.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved visualization to: swin_detection_diagnostic.png")

    # Recommendations
    print(f"\n8. Recommendations:")
    max_conf = objectness.max()

    if max_conf < 0.1:
        print(f"   ⚠️  WARNING: Maximum confidence is very low ({max_conf:.6f})")
        print(f"   → The model may not be properly trained")
        print(f"   → Consider retraining with more data or different hyperparameters")
    elif max_conf < 0.3:
        print(f"   ⚠️  NOTICE: Maximum confidence is low ({max_conf:.6f})")
        print(f"   → Use confidence threshold < 0.2 for this model")
        print(f"   → Consider fine-tuning the model")
    else:
        print(f"   ✓ Model produces reasonable confidence scores (max: {max_conf:.6f})")
        print(f"   → Use confidence threshold around 0.3-0.5")

    if class_probs[0].item() < 0.5:
        print(f"   ⚠️  Class head predicts no crack (prob: {class_probs[0].item():.4f})")
    else:
        print(f"   ✓ Class head predicts crack present (prob: {class_probs[0].item():.4f})")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # CONFIGURE THESE PATHS
    MODEL_PATH = "best_swin_crack_detection.pth"
    TEST_IMAGE = "test_images/0488.png"  # Use one of your test images

    diagnose_swin_detection(MODEL_PATH, TEST_IMAGE)