import os
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
from common.attention import CBAM, CoordinateAttention
import requests
from pathlib import Path


def register_custom_modules():
    """Register custom attention modules with Ultralytics"""
    # Add our modules to Ultralytics' global namespace
    ultralytics.nn.tasks.CBAM = CBAM
    ultralytics.nn.tasks.CoordinateAttention = CoordinateAttention

    # Also add to the current global namespace
    globals()["CBAM"] = CBAM
    globals()["CoordinateAttention"] = CoordinateAttention

    print("✅ Custom attention modules registered with Ultralytics")


def get_fallback_weights():
    """Get fallback segmentation model weights, prioritizing YOLO12-seg"""
    try:
        # Priority order: YOLO12-seg variants first, then YOLO11, then YOLO8
        fallback_weights = [
            "yolo12n-seg.pt", "yolo12s-seg.pt", "yolo12m-seg.pt", "yolo12l-seg.pt",  # YOLO12-seg priority
            "yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt",  # YOLO11-seg fallback
            "yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt"  # YOLO8-seg last resort
        ]

        # First try to find local weights
        for weight in fallback_weights:
            if os.path.exists(weight):
                print(f"✅ Found local segmentation weights: {weight}")
                return weight

        # If no local weights found, return the preferred YOLO12-seg model for download
        preferred_weight = "yolo12s-seg.pt"  # Use YOLO12s-seg as default
        print(f"📥 Will attempt to download: {preferred_weight}")
        return preferred_weight

    except Exception as e:
        print(f"⚠️  Error determining fallback weights: {e}")
        return "yolo12s-seg.pt"  # Default to YOLO12s-seg


def load_pretrained_weights(model, pretrained_path):
    """Load weights from detection model to segmentation model with compatibility checks"""
    print(f"Loading pretrained weights from {pretrained_path}")

    if not os.path.exists(pretrained_path):
        print(f"❌ Pretrained weights not found: {pretrained_path}")
        return model

    try:
        # Load from your trained detection model
        ckpt = torch.load(pretrained_path, map_location="cpu")
        if 'model' in ckpt:
            pretrained_sd = ckpt["model"].state_dict()
        else:
            pretrained_sd = ckpt.state_dict() if hasattr(ckpt, 'state_dict') else ckpt

        current_sd = model.state_dict()

        # Create new state dict with matching layers
        new_sd = {}
        matched_keys = 0
        skipped_keys = 0

        for k, v in current_sd.items():
            if k in pretrained_sd:
                pretrained_weight = pretrained_sd[k]
                if v.shape == pretrained_weight.shape:
                    new_sd[k] = pretrained_weight
                    matched_keys += 1
                    print(f"✅ Loaded: {k}")
                else:
                    print(f"⚠️  Size mismatch: {k} (current: {v.shape}, pretrained: {pretrained_weight.shape})")
                    new_sd[k] = v  # Keep current initialization
                    skipped_keys += 1
            else:
                # Try without model prefix for detection->segmentation transfer
                k_short = k.replace('model.', '') if 'model.' in k else k
                if k_short in pretrained_sd and v.shape == pretrained_sd[k_short].shape:
                    new_sd[k] = pretrained_sd[k_short]
                    matched_keys += 1
                    print(f"✅ Loaded (prefix adjusted): {k}")
                else:
                    print(f"🆕 New layer: {k} (shape: {v.shape})")
                    new_sd[k] = v
                    skipped_keys += 1

        # Load the new state dict
        model.load_state_dict(new_sd, strict=False)
        print(f"\n📊 Weight loading summary:")
        print(f"   Matched: {matched_keys} layers")
        print(f"   Skipped/New: {skipped_keys} layers")
        print(f"   Total: {len(current_sd)} layers in model")

        return model

    except Exception as e:
        print(f"❌ Error loading pretrained weights: {e}")
        return model


def main():
    print("🚀 Starting YOLOv12 Segmentation Model Training")
    print("=" * 60)

    # Register custom modules first
    register_custom_modules()

    # Configuration
    config_file = "models/yolo12_seg_cbam_ca.yaml"  # Use the fixed YAML
    pretrained_detection = "yolo12s_cbam_ca_crack.pt"  # Your trained detection model

    print(f"📁 Config file: {config_file}")
    print(f"🎯 Detection weights: {pretrained_detection}")

    # Check if config file exists
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        print("Please make sure the fixed YAML configuration is saved properly.")
        return

    try:
        print("\n🏗️  Creating segmentation model from custom config...")

        # Method 1: Create model from scratch with custom config
        trainer = YOLO(config_file, task='segment')
        print("✅ Model created from custom configuration")

        # Method 2: Load detection weights if available
        if os.path.exists(pretrained_detection):
            print(f"\n📥 Loading compatible weights from detection model...")
            trainer.model = load_pretrained_weights(trainer.model, pretrained_detection)
        else:
            print(f"⚠️  Detection weights not found: {pretrained_detection}")
            print("🔥 Training from scratch with custom architecture...")

    except Exception as e:
        print(f"❌ Error with custom configuration: {e}")
        print("🔄 Falling back to YOLO12-seg model...")

        # Fallback: Use YOLO12-seg model
        fallback_weights = get_fallback_weights()
        try:
            print(f"📥 Attempting to load: {fallback_weights}")
            trainer = YOLO(fallback_weights, task='segment')
            print(f"✅ Successfully loaded YOLO12-seg model: {fallback_weights}")

            # If we have custom detection weights, try to transfer compatible layers
            if os.path.exists(pretrained_detection):
                print(f"🔄 Attempting to transfer weights from detection model...")
                try:
                    trainer.model = load_pretrained_weights(trainer.model, pretrained_detection)
                except Exception as transfer_error:
                    print(f"⚠️  Weight transfer failed: {transfer_error}")
                    print("✅ Continuing with YOLO12-seg pretrained weights only")

        except Exception as fallback_error:
            print(f"❌ YOLO12-seg fallback failed: {fallback_error}")
            print("🔄 Trying YOLO11-seg as secondary fallback...")

            try:
                trainer = YOLO('yolo11s-seg.pt', task='segment')
                print("✅ Loaded YOLO11s-seg as secondary fallback")
            except Exception as final_error:
                print(f"❌ All fallback methods failed: {final_error}")
                print("🔥 Training from scratch with basic YOLO configuration...")
                trainer = YOLO('yolo11n.yaml', task='segment')  # Most basic config

    # Verify data configuration
    data_config = "dataset/data.yaml"
    if not os.path.exists(data_config):
        print(f"❌ Data config not found: {data_config}")
        print("Please check your dataset configuration.")
        return

    # Training configuration optimized for segmentation
    config = {
        # Data
        "data": data_config,

        # Training parameters
        "epochs": 150,
        "imgsz": 192,
        "batch": 16,  # Further reduced batch size
        "name": "yolo12_seg_cbam_ca_crack",
        "patience": 40,

        # Optimization
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "optimizer": "AdamW",
        "momentum": 0.937,

        # Hardware
        "device": "0" if torch.cuda.is_available() else "cpu",
        "workers": 4,  #

        # Data loading - More conservative settings
        "single_cls": True,  # Single class (crack)
        "cache": "disk",

        # Augmentation - Optimized for crack segmentation
        "augment": True,
        "copy_paste": 0.3,  # Helpful for crack patterns - copy crack segments to other areas
        "mixup": 0.1,  # Light mixup to improve generalization
        "mosaic": 0.8,  # Higher mosaic - good for learning crack context
        "degrees": 15.0,  # Moderate rotation - cracks can appear at various angles
        "translate": 0.2,  # More translation - cracks can appear anywhere
        "scale": 0.6,  # More aggressive scaling - cracks vary in size
        "shear": 5.0,  # Light shear - simulates perspective changes
        "flipud": 0.3,  # Some vertical flip - cracks can be vertical or horizontal
        "fliplr": 0.5,  # Horizontal flip - maintains crack orientation variety
        "hsv_h": 0.02,  # Slight hue variation - different concrete/surface colors
        "hsv_s": 0.8,  # Higher saturation variation - weathered vs new surfaces
        "hsv_v": 0.6,  # Higher value variation - shadows, lighting conditions

        # Training schedule
        "cos_lr": True,
        "amp": True,
        "close_mosaic": 10,
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,

        # Validation
        "val": True,
        "save_period": 25,
        "plots": True,

        # Segmentation specific
        "overlap_mask": True,
        "mask_ratio": 4,

        # Other
        "verbose": True,
        "seed": 42,
        "deterministic": False,  # Allow some randomness for stability
    }

    print("\n⚙️  Training Configuration:")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Image Size: {config['imgsz']}")
    print(f"   Batch Size: {config['batch']}")
    print(f"   Device: {config['device']}")
    print(f"   AMP: {config['amp']}")

    print("\n🚀 Starting training...")
    print("=" * 60)

    try:
        results = trainer.train(**config)

        # Print results
        print("\n🎉 Training completed!")
        print("=" * 60)
        print("📊 Final Results:")

        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            for key, value in metrics.items():
                if 'mAP' in key or 'precision' in key or 'recall' in key:
                    print(f"   {key}: {value:.4f}")

        print(f"\n💾 Model saved in: runs/segment/{config['name']}")
        print("🚀 Model ready for inference!")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

        # Suggest debugging steps
        print("\n🔧 Debugging suggestions:")
        print("1. Check if your dataset images and labels exist")
        print("2. Verify the data.yaml file paths are correct")
        print("3. Try reducing batch size further (batch=1)")
        print("4. Check GPU memory usage")
        print("5. Try using standard YOLO12-seg model: trainer = YOLO('yolo12s-seg.pt', task='segment')")


if __name__ == "__main__":
    main()