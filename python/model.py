"""
LungDeepNet Architecture
DenseNet121 Shared Feature Extractor + Dual-Head (Spatial & Channel) Attention + Multi-Task Heads
"""

import torch
import torch.nn as nn
import torchvision.models as models
from attention import SpatialAttention, ChannelAttention


class LungDeepNet(nn.Module):
    """
    LungDeepNet: Multi-Task Chest X-Ray Analysis System.
    - Shared Backbone: DenseNet121 (DenseBlock 1-4, 1024 feature channels)
    - Dual-Head Attention:
        * Spatial Attention: Focuses on 'WHERE' abnormal lesions are located (feeds Classification)
        * Channel Attention: Focuses on 'WHICH' feature channels represent disease-specific burden (feeds Severity)
    - Multi-Task Outputs:
        * Disease Classification: 4 classes (Normal, Pneumonia, Emphysema, Pulmonary Fibrosis)
        * Radiographic Severity/Burden: Continuous score in [0.0, 1.0]
    """
    def __init__(self, num_classes=4, reduction_ratio=16, dropout_rate=0.25, pretrained=True):
        super(LungDeepNet, self).__init__()
        
        # 1. Pretrained DenseNet121 Shared Backbone
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        densenet = models.densenet121(weights=weights)
        self.features = densenet.features  # Outputs (B, 1024, 7, 7) for 224x224 input
        self.in_channels = 1024
        
        # 2. Dual-Head Attention Mechanisms
        # Spatial Attention Head for Pathology Localization & Disease Discrimination
        self.spatial_attention = SpatialAttention(kernel_size=7)
        
        # Channel Attention Head for Disease-Aware Radiographic Severity/Burden
        self.channel_attention = ChannelAttention(in_channels=self.in_channels, reduction_ratio=reduction_ratio)
        
        # 3. Global Pooling & Regularization
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Head 1: Disease Classification Head (Multi-class cross-entropy)
        self.classification_head = nn.Sequential(
            nn.Linear(self.in_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Head 2: Radiographic Severity Head (Continuous regression [0.0, 1.0])
        self.severity_head = nn.Sequential(
            nn.Linear(self.in_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Restricts predicted burden strictly to continuous [0.0, 1.0] range
        )
        
    def forward(self, x):
        """
        Forward pass through shared representation and dual attention heads.
        Args:
            x (torch.Tensor): Input batch of shape (B, 3, 224, 224)
        Returns:
            dict with classification logits, continuous severity score, and attention maps.
        """
        # Shared high-dimensional convolutional representation: (B, 1024, 7, 7)
        feat = self.features(x)
        
        # Spatial Attention Branch -> Focus on pathological coordinates (lesions)
        spatial_map = self.spatial_attention(feat)      # (B, 1, 7, 7)
        feat_spatial = feat * spatial_map               # Element-wise spatial gating
        
        # Channel Attention Branch -> Focus on disease-aware semantic channels
        channel_weights = self.channel_attention(feat)  # (B, 1024, 1, 1)
        feat_channel = feat * channel_weights           # Element-wise channel gating
        
        # 1. Classification Path
        pooled_cls = self.global_pool(feat_spatial).flatten(1)
        pooled_cls = self.dropout(pooled_cls)
        cls_logits = self.classification_head(pooled_cls)
        
        # 2. Severity Path
        pooled_sev = self.global_pool(feat_channel).flatten(1)
        pooled_sev = self.dropout(pooled_sev)
        severity_score = self.severity_head(pooled_sev)
        
        return {
            'cls_logits': cls_logits,
            'severity': severity_score,
            'spatial_map': spatial_map,
            'channel_weights': channel_weights,
            'shared_features': feat
        }

    def freeze_backbone(self, freeze=True):
        """Freeze DenseNet121 convolutional layers for Phase 1 warmup."""
        for param in self.features.parameters():
            param.requires_grad = not freeze

    def unfreeze_denseblock4(self):
        """Unfreeze DenseBlock 4 and Norm5 for Phase 2 fine-tuning."""
        for name, param in self.features.named_parameters():
            if 'denseblock4' in name or 'norm5' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
