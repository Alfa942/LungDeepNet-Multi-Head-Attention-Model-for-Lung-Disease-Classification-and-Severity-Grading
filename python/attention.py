"""
Dual-Head Attention Modules for LungDeepNet
- Spatial Attention Head: Inter-spatial relationship of features (Where lesions reside)
- Channel Attention Head: Inter-channel relationship of features (Which features quantify severity)
"""

import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    Applies channel-wise MaxPool & AvgPool concatenated across channel dimension,
    followed by a 7x7 Convolution and Sigmoid activation to generate spatial weights.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, C, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)     # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)   # (B, 1, H, W)
        combined = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        
        spatial_map = self.conv(combined)
        return self.sigmoid(spatial_map)                  # (B, 1, H, W)


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation Channel Attention Module.
    Aggregates spatial dimensions via Global Avg & Max Pooling, passes through a two-layer
    MLP with reduction ratio r, and generates channel importance weights via Sigmoid.
    """
    def __init__(self, in_channels=1024, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        reduced_channels = max(in_channels // reduction_ratio, 32)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP across both average and maximum pooled features
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, C, H, W)
        b, c, _, _ = x.size()
        
        avg_feat = self.avg_pool(x).view(b, c)
        max_feat = self.max_pool(x).view(b, c)
        
        avg_out = self.mlp(avg_feat)
        max_out = self.mlp(max_feat)
        
        channel_weights = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return channel_weights
