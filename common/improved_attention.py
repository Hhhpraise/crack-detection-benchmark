import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleSpatialAttention(nn.Module):
    """Multi-scale spatial attention for varying crack widths"""

    def __init__(self, kernel_sizes=[3, 5, 7]):
        super(MultiScaleSpatialAttention, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(2, 1, k, padding=k // 2, bias=False) for k in kernel_sizes
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)

        # Multi-scale convolutions
        outputs = [conv(concat) for conv in self.convs]
        out = sum(outputs) / len(outputs)
        return self.sigmoid(out)


class EdgeEnhancementModule(nn.Module):
    """Edge enhancement for boundary precision in crack detection

    FIXED: Unified constructor signature (c1, c2=None) for YOLO compatibility
    """

    def __init__(self, c1, c2=None):
        super(EdgeEnhancementModule, self).__init__()
        in_channels = c1
        # Use depthwise separable convolution for efficiency
        mid_channels = max(8, in_channels // 4)

        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        edge_weights = self.edge_conv(x)
        return x * (1 + edge_weights)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling for multi-scale context

    FIXED: Uses GroupNorm instead of BatchNorm to avoid batch size=1 errors
    """

    def __init__(self, in_channels, out_channels, dilation_rates=[6, 12, 18]):
        super(ASPP, self).__init__()

        # Use GroupNorm instead of BatchNorm (avoids batch_size=1 issues)
        num_groups = min(32, out_channels)  # Ensure divisibility

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True)
        )

        self.atrous_convs = nn.ModuleList()
        for rate in dilation_rates:
            self.atrous_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.GroupNorm(num_groups, out_channels),
                nn.ReLU(inplace=True)
            ))

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True)
        )

        total_channels = out_channels * (2 + len(dilation_rates))
        self.project = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        size = x.shape[2:]

        # 1x1 conv
        feat1 = self.conv1(x)

        # Atrous convolutions
        atrous_feats = [conv(x) for conv in self.atrous_convs]

        # Global pooling
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=size, mode='bilinear', align_corners=False)

        # Concatenate all features
        concat_feat = torch.cat([feat1] + atrous_feats + [global_feat], dim=1)

        return self.project(concat_feat)


class CrackCBAM(nn.Module):
    """Enhanced CBAM specifically for crack detection

    FIXED: Consistent constructor (c1, c2=None, reduction_ratio=16)
    """

    def __init__(self, c1, c2=None, reduction_ratio=16):
        super(CrackCBAM, self).__init__()

        self.c1 = c1
        self.c2 = c2 if c2 is not None else c1
        mid_channels = max(8, c1 // reduction_ratio)

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(c1, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, c1, 1, bias=False)
        )

        # Multi-scale Spatial Attention
        self.spatial_attn = MultiScaleSpatialAttention([3, 5, 7])

        # Edge Enhancement
        self.edge_enhance = EdgeEnhancementModule(c1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_out = self.sigmoid(avg_out + max_out)
        x = x * channel_out

        # Spatial attention
        spatial_out = self.spatial_attn(x)
        x = x * spatial_out

        # Edge enhancement
        x = self.edge_enhance(x)

        return x


class CrackCoordinateAttention(nn.Module):
    """Enhanced Coordinate Attention for crack patterns

    FIXED: Consistent constructor (c1, c2=None, reduction_ratio=32)
    """

    def __init__(self, c1, c2=None, reduction_ratio=32):
        super(CrackCoordinateAttention, self).__init__()

        in_channels = c1
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mid_channels = max(8, in_channels // reduction_ratio)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mid_channels, in_channels, 1, 1, 0, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, 1, 1, 0, bias=False)

        # Edge enhancement for directional features
        self.edge_conv_h = nn.Conv2d(in_channels, in_channels, (3, 1), padding=(1, 0), groups=in_channels, bias=False)
        self.edge_conv_w = nn.Conv2d(in_channels, in_channels, (1, 3), padding=(0, 1), groups=in_channels, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Directional pooling
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Combine and process
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # Generate attention with edge enhancement
        att_h = self.sigmoid(self.conv_h(x_h) + self.edge_conv_h(identity))
        att_w = self.sigmoid(self.conv_w(x_w) + self.edge_conv_w(identity))

        return identity * att_h * att_w


class CrackASPP(nn.Module):
    """Lightweight ASPP specifically tuned for crack detection

    FIXED: Consistent constructor (c1, c2=None, reduction_ratio=4)
    """

    def __init__(self, c1, c2=None, reduction_ratio=4):
        super(CrackASPP, self).__init__()
        in_channels = c1
        out_channels = max(in_channels // reduction_ratio, 64)

        # Use smaller dilation rates suitable for cracks
        self.aspp = ASPP(in_channels, out_channels, dilation_rates=[3, 6, 9])

        # Project back to original channels
        self.project_back = nn.Sequential(
            nn.Conv2d(out_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        aspp_out = self.aspp(x)
        out = self.project_back(aspp_out)
        return x + out  # Residual connection


class LightCrackCBAM(nn.Module):
    """Lightweight version of CrackCBAM for faster inference

    FIXED: Consistent constructor (c1, c2=None, reduction_ratio=16)
    """

    def __init__(self, c1, c2=None, reduction_ratio=16):
        super(LightCrackCBAM, self).__init__()

        in_channels = c1
        # Simplified channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid_channels = max(in_channels // reduction_ratio, 8)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        # Simplified spatial attention
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.fc(self.avg_pool(x))
        x = x * ca

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial(torch.cat([avg_out, max_out], dim=1))
        x = x * sa

        return x


# Export all modules
__all__ = [
    'CrackCBAM',
    'CrackCoordinateAttention',
    'CrackASPP',
    'LightCrackCBAM',
    'MultiScaleSpatialAttention',
    'EdgeEnhancementModule',
    'ASPP'
]
