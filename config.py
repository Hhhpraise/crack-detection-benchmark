"""
Centralized path configuration for the Crack Detection research project.

All paths that were hardcoded across scripts are centralized here.
Override defaults by setting environment variables before importing.

Usage:
    from config import PROJECT_ROOT, WEIGHTS_DIR, DATASETS_DIR
    from config import YOLO_DETECTION_WEIGHTS, SWIN_DETECTION_WEIGHTS
"""

import os
from pathlib import Path

# ---- Project root ----
PROJECT_ROOT = Path(os.getenv("CRACK_PROJECT_ROOT", Path(__file__).parent.resolve()))

# ---- Source data directories (external, read-only) ----
# Original raw data location — set CRACK_WORK_IMAGES to override
WORK_IMAGES_ROOT = Path(os.getenv("CRACK_WORK_IMAGES", "D:/work images"))

# Detection source (YOLO bbox labels)
DETECTION_SOURCE = WORK_IMAGES_ROOT / "crack_detection"
DETECTION_POSITIVE = DETECTION_SOURCE / "positive_batch"
DETECTION_NEGATIVE = DETECTION_SOURCE / "negative_batch"

# Segmentation source (YOLO polygon labels)
SEGMENTATION_SOURCE = WORK_IMAGES_ROOT / "crack_seg" / "yolo-seg"
SEGMENTATION_POSITIVE = SEGMENTATION_SOURCE / "positive_batch"
SEGMENTATION_NEGATIVE = SEGMENTATION_SOURCE / "negative_batch"
SEGMENTATION_LABELS = SEGMENTATION_SOURCE / "labels" / "train"

# Swin source (COCO JSON + PNG masks)
SWIN_SOURCE = WORK_IMAGES_ROOT / "swinn"
SWIN_POSITIVE = SWIN_SOURCE / "positive_batch"
SWIN_NEGATIVE = SWIN_SOURCE / "negative_batch"
SWIN_MASKS = SWIN_SOURCE / "SegmentationClass"

# ---- Datasets directory (generated / organized data) ----
DATASETS_DIR = PROJECT_ROOT / "datasets"

# YOLO detection dataset (YOLO bbox labels)
DETECTION_DATASET = DATASETS_DIR / "detection"

# YOLO segmentation dataset (YOLO polygon labels)
SEGMENTATION_DATASET = DATASETS_DIR / "segmentation"

# Swin detection dataset (YOLO bbox, 224x224)
SWIN_DETECTION_DATASET = DATASETS_DIR / "swin_detection"

# Swin segmentation dataset (PNG masks, 224x224)
SWIN_SEGMENTATION_DATASET = DATASETS_DIR / "swin_segmentation"

# Unified comparison datasets (from order.py output)
UNIFIED_DETECTION_DATASET = DATASETS_DIR / "unified" / "unified_crack_dataset"
UNIFIED_SEGMENTATION_DATASET = DATASETS_DIR / "unified" / "unified_crack_dataset_seg"

# ---- Weights directory (trained model checkpoints) ----
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# Canonical YOLO weights
YOLO_DETECTION_WEIGHTS = WEIGHTS_DIR / "yolo12s_cbam_ca_crack.pt"
YOLO_SEGMENTATION_WEIGHTS = WEIGHTS_DIR / "yolo12s_seg_cbam_ca_crack.pt"

# Canonical Swin weights
SWIN_DETECTION_WEIGHTS = WEIGHTS_DIR / "best_swin_crack_detection.pth"
SWIN_SEGMENTATION_WEIGHTS = WEIGHTS_DIR / "best_swin_crack_segmentation.pth"

# YOLO base models (auto-downloaded by ultralytics, no need to store)
YOLO_BASE_MODELS = {
    "yolo11n": "yolo11n.pt",
    "yolo11s": "yolo11s.pt",
    "yolo12s": "yolo12s.pt",
    "yolo11s-seg": "yolo11s-seg.pt",
}

# ---- YOLO model configs ----
YOLO_DETECTION_CONFIG = "yolo12_cbam_ca.yaml"
YOLO_SEGMENTATION_CONFIG = "yolo12_seg_cbam_ca.yaml"

# ---- Output directories ----
RUNS_DIR = PROJECT_ROOT / "runs"
OUTPUT_DIR = PROJECT_ROOT / "output"
RESEARCH_RESULTS_DIR = PROJECT_ROOT / "research_results"

# ---- Swin model hyperparameters (single source of truth) ----
SWIN_CONFIG = {
    "img_size": 224,
    "num_classes": 2,
    "embed_dim": 96,
    "depths": [2, 2, 6, 2],
    "num_heads": [3, 6, 12, 24],
    "window_size": 7,
    "mlp_ratio": 4.0,
    "drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "drop_path_rate": 0.1,
}

# ---- Camera calibration (for crack_analysis.py) ----
CAMERA_DISTANCE_M = 1.0
HORIZONTAL_FOV_DEG = 65.0
IMAGE_WIDTH_PIXELS = 4032
IMAGE_HEIGHT_PIXELS = 3024


def ensure_dirs():
    """Create all output directories if they don't exist."""
    for d in [DATASETS_DIR, WEIGHTS_DIR, RUNS_DIR, OUTPUT_DIR, RESEARCH_RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
