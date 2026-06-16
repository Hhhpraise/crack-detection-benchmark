# Add this to your training script before model creation
import yaml
with open("models/yolo12_cbam_ca.yaml") as f:
    config = yaml.safe_load(f)
print("YAML loaded successfully:", config)