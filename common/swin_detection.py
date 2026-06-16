import torch
import torch.nn as nn
from torch.nn import init

try:
    from transformers import SwinConfig, SwinModel
except ImportError:
    SwinConfig = None
    SwinModel = None


class SwinDetectionModel(nn.Module):
    """Swin Transformer-based detection model with bbox/objectness/class heads."""

    def __init__(self, num_classes=1, hidden_dim=256, max_detections=10):
        super().__init__()
        self.max_detections = max_detections

        try:
            self.backbone = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        except Exception:
            self.backbone = SwinModel(config=SwinConfig(
                image_size=224, patch_size=4, num_channels=3, embed_dim=96,
                depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7
            ))

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
        features = features.mean(dim=1)

        bbox_pred = self.bbox_head(features).view(-1, self.max_detections, 4)
        objectness_pred = self.objectness_head(features)
        class_logits = self.class_head(features)

        bbox_pred = torch.sigmoid(bbox_pred)
        objectness_probs = torch.sigmoid(objectness_pred)
        class_probs = torch.sigmoid(class_logits)

        return bbox_pred, objectness_probs, class_probs


__all__ = ['SwinDetectionModel']
