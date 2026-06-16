import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import warnings
import math
from typing import Tuple, Optional
import torch.nn.functional as F
from torch.nn import init
import time
import multiprocessing

warnings.filterwarnings('ignore')


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer"""

    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block"""

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Identity() if drop_path <= 0. else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage"""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 downsample=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class SwinTransformer(nn.Module):
    """Swin Transformer backbone"""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            init.trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)

        return features

    def forward(self, x):
        features = self.forward_features(x)
        return features


class SwinUNet(nn.Module):
    """Swin Transformer U-Net for semantic segmentation"""

    def __init__(self, img_size=224, num_classes=2, embed_dim=96, depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_layers = len(depths)

        self.encoder = SwinTransformer(
            img_size=img_size, embed_dim=embed_dim, depths=depths, num_heads=num_heads,
            window_size=window_size, mlp_ratio=mlp_ratio, drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, num_classes=0
        )

        # Decoder layers
        self.up3_2 = nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2, bias=False)
        self.conv3_2 = nn.Sequential(
            nn.GroupNorm(32, 384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, 384),
            nn.ReLU(inplace=True)
        )

        self.reduce2_1 = nn.Conv2d(576, 384, kernel_size=1, bias=False)
        self.up2_1 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2, bias=False)
        self.conv2_1 = nn.Sequential(
            nn.GroupNorm(32, 192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, 192),
            nn.ReLU(inplace=True)
        )

        self.reduce1_0 = nn.Conv2d(288, 192, kernel_size=1, bias=False)
        self.up1_0 = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2, bias=False)
        self.conv1_0 = nn.Sequential(
            nn.GroupNorm(32, 96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, 96),
            nn.ReLU(inplace=True)
        )

        self.final_upsample = nn.ConvTranspose2d(96, 96, kernel_size=4, stride=4, bias=False)
        self.final_conv = nn.Conv2d(96, num_classes, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        encoder_features = self.encoder.forward_features(x)

        skip_features = []
        patches_resolution = self.encoder.patches_resolution

        for i, feat in enumerate(encoder_features):
            h = patches_resolution[0] // (2 ** i)
            w = patches_resolution[1] // (2 ** i)
            feat_spatial = feat.view(B, h, w, -1).permute(0, 3, 1, 2).contiguous()
            skip_features.append(feat_spatial)

        x4 = skip_features[3]

        x3 = self.up3_2(x4)
        x3 = self.conv3_2(x3)

        skip2 = skip_features[2]
        if skip2.size(2) != x3.size(2) or skip2.size(3) != x3.size(3):
            skip2 = F.interpolate(skip2, size=(x3.size(2), x3.size(3)),
                                  mode='bilinear', align_corners=False)

        x2_concat = torch.cat([x3, skip2], dim=1)
        x2_reduced = self.reduce2_1(x2_concat)
        x2 = self.up2_1(x2_reduced)
        x2 = self.conv2_1(x2)

        skip1 = skip_features[1]
        if skip1.size(2) != x2.size(2) or skip1.size(3) != x2.size(3):
            skip1 = F.interpolate(skip1, size=(x2.size(2), x2.size(3)),
                                  mode='bilinear', align_corners=False)

        x1_concat = torch.cat([x2, skip1], dim=1)
        x1_reduced = self.reduce1_0(x1_concat)
        x1 = self.up1_0(x1_reduced)
        x1 = self.conv1_0(x1)

        x0 = self.final_upsample(x1)
        output = self.final_conv(x0)

        return output


class CrackDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=224, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
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

        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask = (mask > 127).astype(np.uint8)

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
            image = transforms.Resize((self.image_size, self.image_size))(image)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])(image)

        mask = torch.from_numpy(
            cv2.resize(mask, (self.image_size, self.image_size),
                       interpolation=cv2.INTER_NEAREST)
        ).long()

        return image, mask


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target_one_hot = torch.zeros_like(pred)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)

        pred_flat = pred.contiguous().view(-1)
        target_flat = target_one_hot.contiguous().view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)

        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice


class CrackSegmentationTrainer:
    def __init__(self, data_dir, batch_size=16, learning_rate=1e-4, num_epochs=150):
        """
        Initialize the trainer - ALIGNED WITH DETECTION PARAMETERS

        Args:
            data_dir: Directory containing organized dataset
            batch_size: 16 (same as detection)
            learning_rate: 1e-4 (same as detection)
            num_epochs: 150 (same as detection)
        """
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
        self.num_classes = self.dataset_info['num_classes']

        self.setup_data_loaders()
        self.setup_model()
        self.setup_training()

    def setup_data_loaders(self):
        """Setup data loaders - ALIGNED WITH DETECTION FORMAT"""
        # Same transforms as detection
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

        train_dataset = CrackDataset(
            os.path.join(self.data_dir, 'images/train'),
            os.path.join(self.data_dir, 'masks/train'),
            image_size=self.image_size,
            transform=train_transform
        )

        val_dataset = CrackDataset(
            os.path.join(self.data_dir, 'images/val'),
            os.path.join(self.data_dir, 'masks/val'),
            image_size=self.image_size,
            transform=val_transform
        )

        test_dataset = CrackDataset(
            os.path.join(self.data_dir, 'images/test'),
            os.path.join(self.data_dir, 'masks/test'),
            image_size=self.image_size,
            transform=val_transform
        )

        # Same worker settings as detection
        num_workers = 0 if os.name == 'nt' else 4
        pin_memory = torch.cuda.is_available()

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        print(f"Dataset loaded:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Validation: {len(val_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")

    def setup_model(self):
        """Setup the Swin Transformer model"""
        print("Initializing Swin Transformer U-Net...")

        self.model = SwinUNet(
            img_size=self.image_size,
            num_classes=self.num_classes,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1
        )

        self.model = self.model.to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    def setup_training(self):
        """Setup training components - ALIGNED WITH DETECTION"""
        self.criterion = CombinedLoss(ce_weight=1.0, dice_weight=1.0)

        # Same optimizer settings as detection
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        # Same scheduler as detection
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs,
            eta_min=self.learning_rate * 0.01
        )

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_iou': [],
            'val_iou': [],
            'learning_rate': [],
            'epoch_times': []
        }

    def calculate_iou(self, pred, target, num_classes=2):
        """Calculate IoU score"""
        pred = torch.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)

        ious = []
        for cls in range(num_classes):
            pred_cls = pred == cls
            target_cls = target == cls

            intersection = (pred_cls & target_cls).float().sum()
            union = (pred_cls | target_cls).float().sum()

            if union == 0:
                iou = 1.0 if intersection == 0 else 0.0
            else:
                iou = (intersection / union).item()

            ious.append(iou)

        return np.mean(ious)

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        running_iou = 0.0
        start_time = time.time()

        pbar = tqdm(self.train_loader, desc="Training")
        for images, masks in pbar:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            iou = self.calculate_iou(outputs, masks)

            running_loss += loss.item()
            running_iou += iou

            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'IoU': f'{iou:.4f}'})

        epoch_loss = running_loss / len(self.train_loader)
        epoch_iou = running_iou / len(self.train_loader)
        epoch_time = time.time() - start_time

        return epoch_loss, epoch_iou, epoch_time

    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        running_iou = 0.0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for images, masks in pbar:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                iou = self.calculate_iou(outputs, masks)

                running_loss += loss.item()
                running_iou += iou

                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'IoU': f'{iou:.4f}'})

        epoch_loss = running_loss / len(self.val_loader)
        epoch_iou = running_iou / len(self.val_loader)

        return epoch_loss, epoch_iou

    def train(self):
        """Main training loop - ALIGNED WITH DETECTION FORMAT"""
        print("\n" + "=" * 60)
        print("STARTING CRACK SEGMENTATION TRAINING")
        print("=" * 60)

        best_val_iou = 0.0
        start_train_time = time.time()

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            train_loss, train_iou, epoch_time = self.train_epoch()
            val_loss, val_iou = self.validate_epoch()

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_iou'].append(train_iou)
            self.history['val_iou'].append(val_iou)
            self.history['learning_rate'].append(current_lr)
            self.history['epoch_times'].append(epoch_time)

            print(f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | LR: {current_lr:.6f}")

            if val_iou > best_val_iou:
                best_val_iou = val_iou
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_iou': val_iou,
                    'history': self.history,
                    'model_config': {
                        'img_size': self.image_size,
                        'num_classes': self.num_classes,
                        'embed_dim': 96,
                        'depths': [2, 2, 6, 2],
                        'num_heads': [3, 6, 12, 24],
                        'window_size': 7,
                    }
                }, 'best_swin_crack_segmentation.pth')
                print(f"Best model saved! Val IoU: {val_iou:.4f}")

            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'history': self.history
                }, f'checkpoint_epoch_{epoch + 1}.pth')

        total_time = time.time() - start_train_time
        print(f"\nTraining completed in {total_time / 60:.2f} minutes")
        print(f"Best validation IoU: {best_val_iou:.4f}")

    def test(self, model_path='best_swin_crack_segmentation.pth'):
        """Test the trained model"""
        print("\n" + "=" * 60)
        print("TESTING SWIN TRANSFORMER MODEL")
        print("=" * 60)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']} with Val IoU: {checkpoint['val_iou']:.4f}")

        self.model.eval()
        test_loss = 0.0
        test_iou = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Testing")
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                iou = self.calculate_iou(outputs, masks)
                test_loss += loss.item()
                test_iou += iou

                pred = torch.softmax(outputs, dim=1)
                pred = torch.argmax(pred, dim=1)
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(masks.cpu().numpy().flatten())

                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'IoU': f'{iou:.4f}'})

        test_loss /= len(self.test_loader)
        test_iou /= len(self.test_loader)

        print(f"\nFINAL TEST RESULTS:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test IoU: {test_iou:.4f}")

        print("\nDetailed Classification Report:")
        print(classification_report(all_targets, all_predictions,
                                    target_names=['Background', 'Crack']))

        return test_loss, test_iou

    def plot_training_history(self):
        """Plot training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(1, len(self.history['train_loss']) + 1)

        ax1.plot(epochs, self.history['train_loss'], 'bo-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 'ro-', label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, self.history['train_iou'], 'bo-', label='Training IoU', linewidth=2)
        ax2.plot(epochs, self.history['val_iou'], 'ro-', label='Validation IoU', linewidth=2)
        ax2.set_title('Model IoU', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('IoU')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3.plot(epochs, self.history['learning_rate'], 'go-', label='Learning Rate', linewidth=2)
        ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')

        loss_diff = np.array(self.history['val_loss']) - np.array(self.history['train_loss'])
        ax4.plot(epochs, loss_diff, 'mo-', label='Val Loss - Train Loss', linewidth=2)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax4.set_title('Overfitting Indicator', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss Difference')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('swin_segmentation_training_history.png', dpi=200, bbox_inches='tight')
        plt.show()

    def visualize_predictions(self, num_samples=6, model_path='best_swin_crack_segmentation.pth'):
        """Visualize model predictions"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        data_iter = iter(self.test_loader)
        images, masks = next(data_iter)

        num_samples = min(num_samples, len(images))
        images = images[:num_samples]
        masks = masks[:num_samples]

        with torch.no_grad():
            images_gpu = images.to(self.device)
            outputs = self.model(images_gpu)
            predictions = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(predictions, dim=1).cpu()

        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            img = images[i].clone()
            for t, m, s in zip(img, mean, std):
                t.mul_(s).add_(m)
            img = torch.clamp(img, 0, 1)
            img = img.permute(1, 2, 0).numpy()

            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(masks[i], cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(predictions[i], cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')

        plt.suptitle('Swin Transformer - Crack Segmentation Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('swin_segmentation_predictions.png', dpi=200, bbox_inches='tight')
        plt.show()


def main():
    """Main training function - ALIGNED WITH DETECTION PARAMETERS"""

    # SAME CONFIGURATION AS DETECTION
    DATA_DIR = "crack_segmentation_dataset"
    BATCH_SIZE = 16  # Same as detection
    LEARNING_RATE = 1e-4  # Same as detection
    NUM_EPOCHS = 150  # Same as detection

    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory '{DATA_DIR}' not found!")
        return

    print("\n" + "=" * 80)
    print("CRACK SEGMENTATION TRAINING WITH SWIN TRANSFORMER")
    print("=" * 80)
    print("ALIGNED WITH DETECTION TRAINING PARAMETERS:")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Learning Rate: {LEARNING_RATE}")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print(f"  - Optimizer: AdamW (weight_decay=0.01, betas=(0.9, 0.999))")
    print(f"  - Scheduler: CosineAnnealingLR")
    print(f"  - Gradient Clipping: max_norm=1.0")
    print("=" * 80)

    trainer = CrackSegmentationTrainer(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS
    )

    trainer.train()
    trainer.test()
    trainer.plot_training_history()
    trainer.visualize_predictions()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("Files generated:")
    print("  - best_swin_crack_segmentation.pth: Best model weights")
    print("  - checkpoint_epoch_*.pth: Training checkpoints")
    print("  - swin_segmentation_training_history.png: Training curves")
    print("  - swin_segmentation_predictions.png: Sample predictions")
    print("=" * 80)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()