import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.ops import boxes as box_ops
import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from PIL import Image
import glob
import time
import multiprocessing

try:
    from transformers import SwinConfig, SwinModel
except ImportError:
    print("Please install transformers: pip install transformers")
    exit()

warnings.filterwarnings('ignore')


class CrackDetectionDataset(Dataset):
    def __init__(self, images_dir, labels_dir, image_size=224, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_size = image_size
        self.transform = transform

        self.image_files = [f for f in os.listdir(images_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.labels_dir, label_name)

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        # Convert YOLO format to [x1, y1, x2, y2] normalized
                        x1 = (x_center - width / 2)
                        y1 = (y_center - height / 2)
                        x2 = (x_center + width / 2)
                        y2 = (y_center + height / 2)

                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_id)

        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
            image = transforms.Resize((self.image_size, self.image_size))(image)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])(image)

        target = {
            'boxes': boxes,
            'labels': labels,
            'has_crack': len(boxes) > 0
        }

        return image, target


class SwinDetectionModel(nn.Module):
    def __init__(self, num_classes=1, hidden_dim=256, max_detections=10, backbone_pretrained=True):
        """
        Swin Transformer based object detection model
        Supports multiple crack detections per image
        """
        super().__init__()

        self.max_detections = max_detections

        try:
            if backbone_pretrained:
                print("Loading pretrained Swin Transformer...")
                self.backbone = SwinModel.from_pretrained(
                    "microsoft/swin-tiny-patch4-window7-224"
                )
                print("Pretrained model loaded successfully")
            else:
                raise Exception("Using random initialization")
        except Exception as e:
            print(f"Could not load pretrained model: {e}")
            print("Using randomly initialized model...")
            self.backbone = SwinModel(config=SwinConfig(
                image_size=224,
                patch_size=4,
                num_channels=3,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7
            ))

        # Detection heads for multiple objects
        self.bbox_head = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 4 * max_detections)  # Multiple bboxes
        )

        self.objectness_head = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_detections)  # Objectness score for each detection
        )

        self.class_head = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)  # Binary: has crack or not
        )

    def forward(self, x, targets=None):
        features = self.backbone(x).last_hidden_state
        features = features.mean(dim=1)

        # Predictions
        bbox_pred = self.bbox_head(features).view(-1, self.max_detections, 4)
        objectness_pred = self.objectness_head(features)
        class_logits = self.class_head(features)

        # Apply sigmoid to bbox coordinates to keep them in [0, 1]
        bbox_pred = torch.sigmoid(bbox_pred)

        if self.training:
            loss = self.compute_loss(bbox_pred, objectness_pred, class_logits, targets)
            return loss
        else:
            objectness_probs = torch.sigmoid(objectness_pred)
            class_probs = torch.sigmoid(class_logits)
            return bbox_pred, objectness_probs, class_probs

    def compute_loss(self, bbox_pred, objectness_pred, class_logits, targets):
        """Compute detection loss supporting multiple objects per image"""
        bbox_loss = 0
        objectness_loss = 0
        class_loss = 0
        num_objects = 0

        for i, target in enumerate(targets):
            num_gt_boxes = len(target['boxes'])

            # Binary classification loss
            has_crack = float(num_gt_boxes > 0)
            class_loss += F.binary_cross_entropy_with_logits(
                class_logits[i],
                torch.tensor([has_crack], device=class_logits.device)
            )

            if num_gt_boxes > 0:
                # Match predictions to ground truth boxes
                gt_boxes = target['boxes']

                # Use up to max_detections or num_gt_boxes, whichever is smaller
                num_matched = min(num_gt_boxes, self.max_detections)

                for j in range(num_matched):
                    # Compute IoU-based matching (simple: use first N boxes)
                    bbox_loss += F.smooth_l1_loss(bbox_pred[i, j], gt_boxes[j])

                    # Objectness: should be 1 for matched boxes
                    objectness_loss += F.binary_cross_entropy_with_logits(
                        objectness_pred[i, j],
                        torch.tensor(1.0, device=objectness_pred.device)
                    )
                    num_objects += 1

                # Objectness: should be 0 for unmatched boxes
                for j in range(num_matched, self.max_detections):
                    objectness_loss += F.binary_cross_entropy_with_logits(
                        objectness_pred[i, j],
                        torch.tensor(0.0, device=objectness_pred.device)
                    )
            else:
                # No objects: all objectness should be 0
                for j in range(self.max_detections):
                    objectness_loss += F.binary_cross_entropy_with_logits(
                        objectness_pred[i, j],
                        torch.tensor(0.0, device=objectness_pred.device)
                    )

        batch_size = len(targets)
        total_loss = class_loss / batch_size + objectness_loss / (batch_size * self.max_detections)

        if num_objects > 0:
            total_loss += bbox_loss / num_objects

        return total_loss


class CrackDetectionTrainer:
    def __init__(self, data_dir, batch_size=8, learning_rate=1e-4, num_epochs=100):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

        with open(os.path.join(data_dir, 'annotations', 'dataset_info.json'), 'r') as f:
            self.dataset_info = json.load(f)

        self.image_size = self.dataset_info['image_size']
        self.num_classes = 1

        self.setup_data_loaders()
        self.setup_model()
        self.setup_training()

    def setup_data_loaders(self):
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.3),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        train_dataset = CrackDetectionDataset(
            os.path.join(self.data_dir, 'images/train'),
            os.path.join(self.data_dir, 'labels/train'),
            image_size=self.image_size,
            transform=train_transform
        )

        val_dataset = CrackDetectionDataset(
            os.path.join(self.data_dir, 'images/val'),
            os.path.join(self.data_dir, 'labels/val'),
            image_size=self.image_size,
            transform=val_transform
        )

        test_dataset = CrackDetectionDataset(
            os.path.join(self.data_dir, 'images/test'),
            os.path.join(self.data_dir, 'labels/test'),
            image_size=self.image_size,
            transform=val_transform
        )

        num_workers = 0 if os.name == 'nt' else 4
        pin_memory = torch.cuda.is_available()

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        print(f"Dataset loaded:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Validation: {len(val_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")

    def collate_fn(self, batch):
        images = []
        targets = []
        for img, target in batch:
            images.append(img)
            targets.append(target)
        images = torch.stack(images)
        return images, targets

    def setup_model(self):
        print("Initializing Swin Transformer Detection Model...")
        self.model = SwinDetectionModel(
            num_classes=self.num_classes,
            hidden_dim=256,
            max_detections=10,
            backbone_pretrained=True
        )
        self.model = self.model.to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    def setup_training(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs,
            eta_min=self.learning_rate * 0.01
        )

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_times': []
        }

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        start_time = time.time()

        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)

            device_targets = []
            for target in targets:
                device_target = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                                 for k, v in target.items()}
                device_targets.append(device_target)

            self.optimizer.zero_grad()
            loss = self.model(images, device_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        epoch_loss = running_loss / len(self.train_loader)
        epoch_time = time.time() - start_time
        return epoch_loss, epoch_time

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for images, targets in pbar:
                images = images.to(self.device, non_blocking=True)

                device_targets = []
                for target in targets:
                    device_target = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                                     for k, v in target.items()}
                    device_targets.append(device_target)

                self.model.train()
                loss = self.model(images, device_targets)
                self.model.eval()

                running_loss += loss.item()
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        epoch_loss = running_loss / len(self.val_loader)
        return epoch_loss

    def train(self):
        print("\n" + "=" * 60)
        print("STARTING CRACK DETECTION TRAINING")
        print("=" * 60)

        best_val_loss = float('inf')
        start_train_time = time.time()

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            train_loss, epoch_time = self.train_epoch()
            val_loss = self.validate_epoch()

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            self.history['epoch_times'].append(epoch_time)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'history': self.history
                }, 'best_swin_crack_detection.pth')
                print(f"Best model saved! Val Loss: {val_loss:.4f}")

            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'history': self.history
                }, f'checkpoint_epoch_{epoch + 1}.pth')

        total_time = time.time() - start_train_time
        print(f"\nTraining completed in {total_time / 60:.2f} minutes")
        print(f"Best validation loss: {best_val_loss:.4f}")

    def visualize_predictions(self, num_samples=6):
        checkpoint = torch.load('best_swin_crack_detection.pth', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        data_iter = iter(self.test_loader)
        images, targets = next(data_iter)
        num_samples = min(num_samples, len(images))

        with torch.no_grad():
            bbox_pred, objectness_probs, class_probs = self.model(images[:num_samples].to(self.device))

        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

        fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))
        if num_samples == 1:
            axes = axes.reshape(2, 1)

        for i in range(num_samples):
            img = images[i].clone()
            for t, m, s in zip(img, mean, std):
                t.mul_(s).add_(m)
            img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()

            # Ground truth
            gt_img = (img * 255).astype(np.uint8).copy()
            for bbox in targets[i]['boxes']:
                x1, y1, x2, y2 = (bbox * self.image_size).int().numpy()
                cv2.rectangle(gt_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Predictions
            pred_img = (img * 255).astype(np.uint8).copy()
            for j in range(self.model.max_detections):
                if objectness_probs[i, j] > 0.5:
                    bbox = bbox_pred[i, j]
                    x1, y1, x2, y2 = (bbox * self.image_size).cpu().int().numpy()
                    cv2.rectangle(pred_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(pred_img, f'{objectness_probs[i, j]:.2f}',
                                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            axes[0, i].imshow(gt_img)
            axes[0, i].set_title('Ground Truth')
            axes[0, i].axis('off')

            axes[1, i].imshow(pred_img)
            axes[1, i].set_title(f'Pred (crack: {class_probs[i].item():.2f})')
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.savefig('predictions.png', dpi=200, bbox_inches='tight')
        plt.show()


def main():
    DATA_DIR = "crack_detection_dataset"
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 150

    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory '{DATA_DIR}' not found!")
        return

    trainer = CrackDetectionTrainer(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS
    )

    trainer.train()
    trainer.visualize_predictions()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()