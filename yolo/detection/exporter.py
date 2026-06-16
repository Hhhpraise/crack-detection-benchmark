"""
This script converts your trained custom YOLO model to a proper format
that can be loaded by Ultralytics without the YOLOv5 compatibility error.
"""

import torch
import os
from pathlib import Path
import yaml
from ultralytics.nn.tasks import DetectionModel
import ultralytics
import torch.nn as nn
from common.attention import CBAM, CoordinateAttention, ChannelAttention, SpatialAttention


def register_custom_modules():
    """Register custom modules with Ultralytics"""
    ultralytics.nn.tasks.CBAM = CBAM
    ultralytics.nn.tasks.CoordinateAttention = CoordinateAttention
    ultralytics.nn.tasks.ChannelAttention = ChannelAttention
    ultralytics.nn.tasks.SpatialAttention = SpatialAttention

    globals()["CBAM"] = CBAM
    globals()["CoordinateAttention"] = CoordinateAttention
    globals()["ChannelAttention"] = ChannelAttention
    globals()["SpatialAttention"] = SpatialAttention

    print("✓ Custom modules registered")


def find_trained_weights():
    """Find the trained model weights"""

    # Check common locations
    possible_paths = [
        "yolo12s_cbam_ca_crack.pt",
        "yolo12s_cbam_ca_exported.pt",
        "runs/detect/yolo12_cbam_ca_crack/weights/best.pt",
        "runs/detect/yolo12_cbam_ca_crack2/weights/best.pt",
        "runs/detect/yolo12_cbam_ca_crack3/weights/best.pt",
    ]

    # Search in runs directory
    runs_dir = Path('runs/detect')
    if runs_dir.exists():
        model_runs = sorted([d for d in runs_dir.iterdir()
                           if d.is_dir() and 'yolo12_cbam_ca' in d.name],
                          key=lambda x: x.stat().st_mtime)

        for run in reversed(model_runs):  # Check most recent first
            best_weights = run / 'weights' / 'best.pt'
            if best_weights.exists():
                possible_paths.insert(0, str(best_weights))

    # Find first existing path
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✓ Found trained weights: {path}")
            return path

    return None


def convert_model_format(source_path, output_path="yolo12s_cbam_ca_final.pt"):
    """
    Convert the model to proper Ultralytics format by reconstructing it
    """

    print(f"\n{'='*70}")
    print("CONVERTING MODEL TO ULTRALYTICS FORMAT")
    print(f"{'='*70}\n")

    register_custom_modules()

    print(f"Source: {source_path}")
    print(f"Output: {output_path}\n")

    try:
        # Load the checkpoint
        print("Loading checkpoint...")
        ckpt = torch.load(source_path, map_location='cpu')
        print("✓ Checkpoint loaded")

        # Extract model state dict
        if isinstance(ckpt, dict):
            if 'model' in ckpt:
                # It's a training checkpoint
                if hasattr(ckpt['model'], 'state_dict'):
                    state_dict = ckpt['model'].state_dict()
                    model_obj = ckpt['model']
                else:
                    state_dict = ckpt['model']
                    model_obj = None

                # Get other training info
                epoch = ckpt.get('epoch', -1)
                best_fitness = ckpt.get('best_fitness', None)

                print(f"  Type: Training checkpoint")
                if epoch >= 0:
                    print(f"  Epoch: {epoch}")
                if best_fitness:
                    print(f"  Best fitness: {best_fitness}")
            else:
                state_dict = ckpt
                model_obj = None
                print(f"  Type: State dict")
        else:
            print(f"  Type: Unknown format")
            state_dict = None
            model_obj = ckpt

        # Try to reconstruct model using YAML config
        config_path = "models/yolo12_cbam_ca.yaml"

        if os.path.exists(config_path):
            print(f"\n✓ Found config: {config_path}")
            print("Reconstructing model from config...")

            # Create fresh model from config
            new_model = DetectionModel(cfg=config_path, verbose=False)
            print("✓ Model reconstructed")

            # Load the trained weights
            if state_dict is not None:
                print("\nLoading trained weights...")
                # Try strict loading first
                try:
                    new_model.load_state_dict(state_dict, strict=True)
                    print("✓ Weights loaded (strict)")
                except Exception as e:
                    print(f"  Strict loading failed: {e}")
                    print("  Trying non-strict loading...")
                    missing, unexpected = new_model.load_state_dict(state_dict, strict=False)
                    print(f"✓ Weights loaded (non-strict)")
                    if missing:
                        print(f"  Missing keys: {len(missing)}")
                    if unexpected:
                        print(f"  Unexpected keys: {len(unexpected)}")

            # Create proper Ultralytics checkpoint format
            print("\nCreating Ultralytics checkpoint...")

            # Get model metadata
            import datetime
            from ultralytics import __version__

            # IMPORTANT: Save the model object, not state_dict
            save_dict = {
                'epoch': epoch if 'epoch' in locals() else -1,
                'best_fitness': best_fitness if 'best_fitness' in locals() else None,
                'model': new_model,  # Save model object (not state_dict)
                'ema': None,
                'updates': None,
                'optimizer': None,
                'train_args': {
                    'task': 'detect',
                    'mode': 'train',
                    'model': config_path,
                    'data': 'dataset/data.yaml',
                },
                'date': datetime.datetime.now().isoformat(),
                'version': __version__,
            }

            # Save the new checkpoint
            torch.save(save_dict, output_path)
            print(f"✓ Model saved to: {output_path}")

            # Verify it can be loaded
            print("\nVerifying converted model...")
            from ultralytics import YOLO

            try:
                test_model = YOLO(output_path, task='detect')
                print("✓ Model loads successfully with YOLO()")

                # Test inference
                import numpy as np
                from PIL import Image

                print("\nTesting inference...")
                dummy_img = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
                dummy_img.save("temp_test.jpg")

                results = test_model("temp_test.jpg", verbose=False)
                print("✓ Inference successful")

                os.remove("temp_test.jpg")

                print(f"\n{'='*70}")
                print("CONVERSION SUCCESSFUL!")
                print(f"{'='*70}")
                print(f"\nConverted model: {output_path}")
                print("\nUpdate your comparison script to use this file:")
                print(f"  'YOLO12s-CBAM-CA': '{output_path}'")

                return output_path

            except Exception as e:
                print(f"✗ Verification failed: {e}")
                print("\nThe model was saved but may still have compatibility issues.")
                print("Try loading it directly in your comparison script.")
                return output_path

        else:
            print(f"\n✗ Config file not found: {config_path}")
            print("Cannot reconstruct model without config file.")
            return None

    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("=" * 70)
    print("CUSTOM YOLO MODEL CONVERTER")
    print("=" * 70)

    # Find trained weights
    print("\nSearching for trained model...")
    source_path = find_trained_weights()

    if source_path:
        # Convert the model
        output_path = convert_model_format(source_path, "yolo12s_cbam_ca_final.pt")

        if output_path:
            print(f"\n{'='*70}")
            print("NEXT STEPS")
            print(f"{'='*70}")
            print("\n1. Update your comparer.py YOLO_MODELS dictionary:")
            print("   YOLO_MODELS = {")
            print(f"       'YOLO12s-CBAM-CA': '{output_path}',")
            print("       'YOLOv11n': 'yolo11n.pt',")
            print("       'YOLOv12s': 'yolo12s.pt',")
            print("   }")
            print("\n2. Run the comparison:")
            print("   python comparer.py")
    else:
        print("\n✗ No trained model found!")
        print("\nPlease specify the path manually:")
        print("  Edit this script and set:")
        print("  source_path = 'path/to/your/trained/model.pt'")
        print("\nOr train the model first using:")
        print("  python train.py")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        source = sys.argv[1]
        output = sys.argv[2] if len(sys.argv) > 2 else "yolo12s_cbam_ca_final.pt"

        if os.path.exists(source):
            convert_model_format(source, output)
        else:
            print(f"✗ File not found: {source}")
    else:
        main()