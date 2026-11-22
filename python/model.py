"""
LungDeepNet Architecture v2 — ConvNeXt-V2 Base Backbone
========================================================
Replaces DenseNet121 with ConvNeXt-V2 Base (1024-channel output) — the 2024
state-of-the-art for chest X-ray classification (AUC ~0.85+, vs DenseNet121's ~0.75).

ConvNeXt-V2 uses:
  - Fully-Convolutional Masked Autoencoder (FCMAE) self-supervised pretraining
  - Global Response Normalization (GRN) layers for feature inter-channel contrast
  - Depthwise separable 7×7 convolutions + inverted bottleneck (Transformer-like design)
  - Superior texture/structural feature extraction for subtle pathologies

Architecture:
  Shared Backbone : ConvNeXt-V2 Base  (stages 1-4 → 1024-channel feature map, 7×7 spatial)
  Dual-Head Attention:
    * Spatial Attention  → Localises WHERE lesions are (feeds Classification)
    * Channel Attention  → Weights WHICH channels encode disease burden (feeds Severity)
  Multi-Task Heads:
    * Classification Head  : 4-class softmax  (Normal, Pneumonia, Emphysema, Pulmonary Fibrosis)
    * Severity Head        : Continuous [0,1] sigmoid regression
"""

import torch
import torch.nn as nn
import timm
from attention import SpatialAttention, ChannelAttention


class LungDeepNet(nn.Module):
    """
    LungDeepNet v2: Multi-Task Chest X-Ray Analysis System.
    Backbone: ConvNeXt-V2 Base (1024 feature channels at 7×7 spatial resolution).
    Dual-Head: Spatial + Channel CBAM-style Attention.
    Outputs: 4-class disease classification + continuous severity score.
    """

    # ConvNeXt-V2 Base outputs 1024 channels from its final stage.
    BACKBONE_OUT_CHANNELS = 1024

    def __init__(
        self,
        num_classes: int = 4,
        reduction_ratio: int = 16,
        dropout_rate: float = 0.30,
        pretrained: bool = True,
    ):
        super(LungDeepNet, self).__init__()

        # ------------------------------------------------------------------ #
        # 1. ConvNeXt-V2 Base Backbone (pretrained on ImageNet-22k / FCMAE)  #
        # ------------------------------------------------------------------ #
        # 'convnextv2_base' is available in timm ≥0.9.5                     #
        # features_only=True → returns (B, 1024, 7, 7) for 224×224 input    #
        backbone = timm.create_model(
            "convnextv2_base",
            pretrained=pretrained,
            features_only=True,   # Return intermediate feature maps
            out_indices=(3,),     # Only stage-4 output: (B, 1024, 7, 7)
        )
        # Wrap so self.features(x) returns a single tensor, not a list
        self._backbone = backbone
        self.in_channels = self.BACKBONE_OUT_CHANNELS

        # ------------------------------------------------------------------ #
        # 2. Dual-Head Attention Mechanisms                                   #
        # ------------------------------------------------------------------ #
        # Spatial Attention: focuses on pathological spatial coordinates
        self.spatial_attention = SpatialAttention(kernel_size=7)

        # Channel Attention: focuses on disease-discriminative feature channels
        self.channel_attention = ChannelAttention(
            in_channels=self.in_channels, reduction_ratio=reduction_ratio
        )

        # ------------------------------------------------------------------ #
        # 3. Pooling & Regularisation                                         #
        # ------------------------------------------------------------------ #
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)

        # ------------------------------------------------------------------ #
        # 4. Task Heads                                                        #
        # ------------------------------------------------------------------ #
        # Head 1 — Disease Classification (4-class cross-entropy)
        self.classification_head = nn.Sequential(
            nn.Linear(self.in_channels, 512),
            nn.LayerNorm(512),          # LayerNorm matches ConvNeXt design
            nn.GELU(),                  # GELU activation used throughout ConvNeXt
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),
        )

        # Head 2 — Radiographic Severity (continuous [0, 1] regression)
        self.severity_head = nn.Sequential(
            nn.Linear(self.in_channels, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    # ---------------------------------------------------------------------- #
    # Forward Pass                                                             #
    # ---------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Input batch (B, 3, 224, 224)
        Returns:
            dict:
              'cls_logits'      – (B, num_classes)
              'severity'        – (B, 1)
              'spatial_map'     – (B, 1, 7, 7)
              'channel_weights' – (B, 1024, 1, 1)
              'shared_features' – (B, 1024, 7, 7)
        """
        # ConvNeXt-V2 feature extraction → (B, 1024, 7, 7)
        feat_list = self._backbone(x)
        feat = feat_list[0]   # Stage-4 output

        # --- Spatial Attention Branch (WHERE lesions are) ---
        spatial_map = self.spatial_attention(feat)      # (B, 1, 7, 7)
        feat_spatial = feat * spatial_map               # Spatial gating

        # --- Channel Attention Branch (WHICH channels matter for severity) ---
        channel_weights = self.channel_attention(feat)  # (B, 1024, 1, 1)
        feat_channel = feat * channel_weights           # Channel gating

        # --- Classification Path ---
        pooled_cls = self.global_pool(feat_spatial).flatten(1)
        pooled_cls = self.dropout(pooled_cls)
        cls_logits = self.classification_head(pooled_cls)

        # --- Severity Path ---
        pooled_sev = self.global_pool(feat_channel).flatten(1)
        pooled_sev = self.dropout(pooled_sev)
        severity_score = self.severity_head(pooled_sev)

        return {
            "cls_logits": cls_logits,
            "severity": severity_score,
            "spatial_map": spatial_map,
            "channel_weights": channel_weights,
            "shared_features": feat,
        }

    # ---------------------------------------------------------------------- #
    # Freeze / Unfreeze Utilities for Multi-Phase Training                    #
    # ---------------------------------------------------------------------- #
    def freeze_backbone(self, freeze: bool = True):
        """Freeze all ConvNeXt-V2 backbone weights (Phase 1 warmup)."""
        for param in self._backbone.parameters():
            param.requires_grad = not freeze

    def unfreeze_stage4(self):
        """Unfreeze only Stage 4 (deepest ConvNeXt blocks) for Phase 2."""
        for param in self._backbone.parameters():
            param.requires_grad = False
        # timm's ConvNeXtV2 names stage-4 blocks as 'stages.3'
        for name, param in self._backbone.named_parameters():
            if "stages.3" in name or "norm" in name:
                param.requires_grad = True

    def unfreeze_stages3_4(self):
        """Unfreeze Stage 3 + Stage 4 for Phase 3 deep fine-tuning."""
        for param in self._backbone.parameters():
            param.requires_grad = False
        for name, param in self._backbone.named_parameters():
            if "stages.2" in name or "stages.3" in name or "norm" in name:
                param.requires_grad = True

    def unfreeze_full_backbone(self):
        """Unfreeze entire backbone for maximum end-to-end fine-tuning."""
        for param in self._backbone.parameters():
            param.requires_grad = True

    # Convenience: target layer for Grad-CAM (last ConvNeXt norm layer)
    def get_gradcam_target_layer(self):
        """Returns the last normalisation layer for Grad-CAM hook registration."""
        return self._backbone.norm_pre if hasattr(self._backbone, "norm_pre") else \
               list(self._backbone.children())[-1]
