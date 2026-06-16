import os
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from common.attention import CBAM, CoordinateAttention


# Register custom modules with Ultralytics
def register_custom_modules():
    # Add our modules to Ultralytics' global namespace
    ultralytics.nn.tasks.CBAM = CBAM
    ultralytics.nn.tasks.CoordinateAttention = CoordinateAttention

    # Also add to the current global namespace
    globals()["CBAM"] = CBAM
    globals()["CoordinateAttention"] = CoordinateAttention

    print("Custom modules registered with Ultralytics")


def load_pretrained_weights(model, pretrained_path):
    """Load weights with compatibility checks and size matching"""
    print(f"Loading pretrained weights from {pretrained_path}")
    ckpt = torch.load(pretrained_path, map_location="cpu")
    pretrained_sd = ckpt["model"].state_dict()
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
            else:
                print(f"Size mismatch: {k} (model: {v.shape}, pretrained: {pretrained_weight.shape})")
                new_sd[k] = v  # Keep current initialization
                skipped_keys += 1
        else:
            # Initialize new layers (attention modules)
            print(f"New layer: {k} (shape: {v.shape})")
            new_sd[k] = v
            skipped_keys += 1

    # Load the new state dict
    model.load_state_dict(new_sd, strict=False)
    print(f"Weight loading complete: {matched_keys} matched, {skipped_keys} skipped/initialized")
    return model


def main():
    register_custom_modules()

    config_file = "models/yolo12_cbam_ca.yaml"
    pretrained = "yolo12s.pt"
    custom_model_path = "yolo12s_custom.pt"

    # Create model from YAML
    model = DetectionModel(cfg=config_file, verbose=True)

    # Load pretrained weights
    if os.path.exists(pretrained):
        model = load_pretrained_weights(model, pretrained)
    else:
        print("Pretrained weights not found, training from scratch")

    # Save using Ultralytics' proper method
    torch.save({
        'model': model,
        'yaml': config_file  # Add YAML config to the checkpoint
    }, custom_model_path)
    print(f"Saved custom model to {custom_model_path}")

    # Initialize YOLO trainer with the model file
    trainer = YOLO(custom_model_path)

    # Training configuration
    config = {
        "data": "dataset/data.yaml",
        "epochs": 150,
        "imgsz": 192,
        "batch": 16,
        "name": "yolo12_cbam_ca_crack",
        "patience": 30,
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "optimizer": "AdamW",
        "label_smoothing": 0.1,
        "device": "0" if torch.cuda.is_available() else "cpu",
        "cache": "disk",
        "single_cls": True,
        "augment": True,
        "cos_lr": True,
        "amp": True,  # Ensure this is enabled (default)
        "close_mosaic": 10,
        "warmup_epochs": 5,
        "workers": 4,
    }

    print("Starting training...")
    results = trainer.train(**config)

    # Print and save results
    print("\nTraining Results:")
    print(f"mAP@0.5: {results.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"Precision: {results.results_dict['metrics/precision(B)']:.4f}")
    print(f"Recall: {results.results_dict['metrics/recall(B)']:.4f}")

    # Save the final model
    trainer.save("yolo12s_cbam_ca_crack.pt")
    print("Custom model saved as 'yolo12s_cbam_ca_crack.pt'")


if __name__ == "__main__":
    main()
