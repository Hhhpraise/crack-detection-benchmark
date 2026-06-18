# Crack Detection Benchmark

**YOLOv12 + Attention vs Swin Transformer for crack detection and segmentation in concrete structures.**

---
This code accompanies a manuscript currently under review. The citation will be updated upon publication. 

本代码随附于一篇目前正在审稿中的论文。论文发表后，引用信息将予以更新。
---

**Authors:** [Praise O. Arowolo](https://github.com/Hhhpraise) & Ellie Akahoho Banks

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.0+-00b894.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive benchmark comparing two deep learning architectures for automated crack detection and segmentation in concrete infrastructure. Trained and evaluated on 10,000 images with paired detection (bounding box) and segmentation (polygon mask) annotations.

## Overview

This repository accompanies a deep learning for crack damage assessment. It provides:

- Two full training pipelines (YOLOv12+Attention and Swin Transformer)
- Detection and segmentation for both architectures
- A unified comparison framework with statistical significance testing
- Publication-quality visualizations and metrics
- Pre-trained model weights (YOLO) and instructions for Swin weights

### Architectures

| Pipeline | Detection | Segmentation |
|----------|-----------|---------------|
| **YOLOv12 + Attention** | YOLOv12s with CBAM + Coordinate Attention in backbone | YOLOv12s-seg with CBAM + Coordinate Attention |
| **Swin Transformer** | Swin-Tiny backbone with detection heads | SwinUNet (Swin Transformer U-Net) |

## Project Structure

```
├── config.py                  # Centralized configuration (paths, hyperparameters)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── common/                    # Shared library (no duplicated code)
│   ├── attention.py           # CBAM, CoordinateAttention modules
│   ├── improved_attention.py  # CrackCBAM, CrackASPP, CrackCoordinateAttention
│   ├── swin_seg.py            # Full SwinUNet + CrackSegmentationTrainer
│   └── swin_detection.py      # SwinDetectionModel (shared base class)
│
├── yolo/                      # YOLO detection + segmentation
│   ├── detection/
│   │   ├── train.py           # Train YOLO detection model
│   │   ├── detect.py          # Single-image detection inference
│   │   ├── organize.py        # Dataset organization (train/val split)
│   │   ├── exporter.py        # Model format converter (→ Ultralytics format)
│   │   ├── checker.py         # YAML config validator
│   │   └── models/            # YOLO model YAML configs
│   │
│   └── segmentation/
│       ├── train.py           # Train YOLO segmentation model
│       ├── inference.py       # Combined detection+segmentation + crack metrics
│       ├── organize.py        # Segmentation dataset organization
│       └── models/            # YOLO seg model YAML configs
│
├── swin/                      # Swin Transformer detection + segmentation
│   ├── detection/
│   │   ├── train.py           # Swin detection training
│   │   ├── compare.py         # Swin vs YOLO detection comparison
│   │   ├── debug_inference.py # YOLO output debugging tool
│   │   ├── correct.py         # COCO JSON annotation fixer
│   │   └── organizer.py       # Detection dataset organizer (COCO→YOLO)
│   │
│   └── segmentation/
│       ├── inference.py       # Swin segmentation inference
│       ├── model_comparison.py # Joint comparison (detection + segmentation)
│       └── organizer.py       # Segmentation dataset organizer
│
├── comparison/                # Research comparison framework
│   ├── model_comparison.py    # Main comparison script (t-test, visualization)
│   ├── crack_analysis.py      # Quantitative crack measurement (camera-calibrated)
│   ├── inferencer.py          # All-in-one inference (YOLO + Swin)
│   ├── yolo_trainer.py        # Unified YOLO dual-task trainer
│   ├── visualizer.py          # Publication-grade figure generation
│   ├── diagnose.py            # Swin model diagnostics
│   ├── cache_cleaner.py       # Cache cleanup utility
│   └── order.py               # Dataset merging & splitting utility
│
├── demo/                      # Small test subset (20 annotated images)
│   ├── detection/             # Images + YOLO bbox labels
│   └── segmentation/          # Images + YOLO polygon labels + masks
│
└── weights/                   # Trained model weights
    ├── yolo12s_cbam_ca_crack.pt          # YOLO detection model (~19 MB)
    └── yolo12s_seg_cbam_ca_crack.pt      # YOLO segmentation model (~20 MB)
```

## Quick Start

### Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

```bash
# Clone the repository
git clone https://github.com/Hhhpraise/crack-detection-benchmark.git
cd crack-detection-benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
# or
venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

### Test with the Demo

```bash
# Run YOLO detection on a demo image
cd yolo/detection
python detect.py   # Edit IMAGE_PATH in the script to point to demo/detection/test/0001.png

# Run combined detection + segmentation
cd ../segmentation
python inference.py   # Uses demo images for testing
```

## Configuration

All paths and hyperparameters are centralized in `config.py`. Override defaults via environment variables:

```bash
# Set custom data source directory
export CRACK_WORK_IMAGES=/path/to/your/data

# Set custom project root
export CRACK_PROJECT_ROOT=/path/to/project
```

## Dataset

### Demo Subset

This repository includes a small demo subset of 20 annotated images under `demo/` for testing the code.

### Full Dataset

The full dataset contains 10,000 images (5,000 crack + 5,000 non-crack) with:
- **Detection**: YOLO bounding box format (`class_id x_center y_center width height`, normalized [0,1])
- **Segmentation**: YOLO polygon format (`class_id x1 y1 x2 y2 ...`, normalized [0,1])
- **Swin datasets**: Images resized to 224×224, with PNG grayscale masks for segmentation

The full dataset is available separately:
- **Kaggle**: [link to be added]
- **Zenodo**: [link to be added]

### Dataset Preparation

Original data should be placed in a directory referenced by `CRACK_WORK_IMAGES` (default: `D:/work images/`). Run the organizer scripts:

```bash
# YOLO detection dataset
cd yolo/detection && python organize.py

# YOLO segmentation dataset
cd yolo/segmentation && python organize.py

# Swin detection dataset
cd swin/detection && python organizer.py

# Swin segmentation dataset
cd swin/segmentation && python organizer.py
```

### Annotation Formats

| Dataset | Format | Notes |
|---------|--------|-------|
| Detection | YOLO bbox: `class_id x_c y_c w h` | Normalized [0,1] |
| Segmentation | YOLO polygon: `class_id x1 y1 x2 y2 ...` | Normalized [0,1] |
| Swin detection | YOLO bbox | Resized 224×224 |
| Swin segmentation | PNG mask | Grayscale, >127 = crack |

## Training

### YOLO Detection

```bash
cd yolo/detection
python train.py
```

Uses YOLOv12s pretrained weights, trains 150 epochs with CBAM and Coordinate Attention modules.

### YOLO Segmentation

```bash
cd yolo/segmentation
python train.py
```

Transfers weights from the trained detection model, trains 150 epochs.

### Swin Detection

```bash
cd swin/detection
python train.py
```

Swin-Tiny backbone with 3-layer detection heads, 150 epochs, CosineAnnealingLR.

### Swin Segmentation

```bash
cd swin/segmentation
# The training script is in common/swin_seg.py, run via main():
python -c "from common.swin_seg import main; main()"
```

SwinUNet architecture, Dice + CrossEntropy loss, 150 epochs.

## Inference

### YOLO

```bash
cd yolo/detection && python detect.py          # Single image detection
cd yolo/segmentation && python inference.py     # Combined detection + segmentation
cd yolo/segmentation && python inference.py batch <dir>  # Batch processing
```

### Swin

```bash
cd swin/segmentation && python inference.py     # Edit INPUT_PATH in script
```

### Combined (YOLO + Swin)

```bash
cd comparison && python inferencer.py           # Runs both pipelines
```

## Model Comparison

```bash
cd comparison
python model_comparison.py
```

This script:
1. Loads both YOLO and Swin models
2. Evaluates detection and segmentation on the test set
3. Computes metrics: Precision, Recall, F1, IoU, Dice, Pixel Accuracy, Boundary IoU
4. Performs paired t-test and Wilcoxon signed-rank test
5. Generates publication-quality figures:
   - Comprehensive summary radar chart
   - Detection and segmentation bar charts
   - IoU distribution histograms
   - Speed comparison (FPS)
   - Per-image scatter plots
   - Box plots with statistical annotations

Results are saved to `comparison/research_results/`.

### Quantitative Crack Analysis

```bash
cd comparison
python crack_analysis.py <image_folder>
```

Calculates crack length, width, area, and severity classification using FOV-based camera calibration.

## Pre-trained Weights

### YOLO (included in this repo)

- `weights/yolo12s_cbam_ca_crack.pt` — Detection model (~19 MB)
- `weights/yolo12s_seg_cbam_ca_crack.pt` — Segmentation model (~20 MB)

YOLO base models (`yolo11n.pt`, `yolo12s.pt`, etc.) are auto-downloaded by Ultralytics.

### Swin (download separately)

The Swin Transformer weights are too large for GitHub (>300 MB each):

- `best_swin_crack_detection.pth` — Detection model (~340 MB)
- `best_swin_crack_segmentation.pth` — Segmentation model (~380 MB)

Download from the [Releases page](https://github.com/Hhhpraise/crack-detection-benchmark/releases) or Zenodo (links coming soon).

## Research Results

Key evaluation outputs stored in `comparison/research_results/`:

- `comparison_results.json` — Complete evaluation data with per-image metrics
- `visualizations/comprehensive_summary.png` — Main summary figure
- `visualizations/metric_comparison_bars.png` — Detection & segmentation bar charts
- `visualizations/iou_distributions.png` — IoU distribution histograms
- `visualizations/speed_comparison.png` — Inference speed analysis
- `visualizations/per_image_scatter.png` — Per-image performance analysis
- `visualizations/box_plots.png` — Distribution box plots

## Troubleshooting

### Path Issues

All paths are configured through `config.py`. Use environment variables to override:
```bash
export CRACK_WORK_IMAGES=/path/to/your/data
```

### CUDA Out of Memory

Reduce `batch_size` or `imgsz` in the training config dictionary, or run on CPU.

### Swin Model Loading Issues

- Ensure `transformers` is installed: `pip install transformers`
- Swin uses `microsoft/swin-tiny-patch4-window7-224` (auto-downloaded on first use)
- For the pre-trained `.pth` weights, download from the releases page

### Missing `model_config` in Checkpoint

This bug has been fixed. The training script in `common/swin_seg.py` now saves `model_config` alongside weights, enabling proper inference reconstruction.

## Citation

If you use this code or benchmark in your research, please cite:

```bibtex
@misc{crack-detection-benchmark,
  author = {Arowolo, Praise O. and Banks, Ellie Akahoho},
  title  = {Crack Detection Benchmark: YOLOv12+Attention vs Swin Transformer},
  year   = {2025},
  url    = {https://github.com/Hhhpraise/crack-detection-benchmark}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
