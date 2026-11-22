"""
Explainability & Localisation Engine — LungDeepNet v2
=======================================================
Updated for ConvNeXt-V2 Base backbone (replaces DenseNet121 hooks).

ConvNeXt-V2 Base Grad-CAM target layer:
  → The normalisation layer after the last ConvNeXt stage block:
    model._backbone.stages[-1] (last ConvNeXtV2 block in Stage 4).

All public APIs (generate_gradcam, generate_gradcam_plus_plus,
get_spatial_attention_map, _normalize_cam) remain identical for
backward compatibility with server.ts and infer.py.
"""

import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    from PIL import Image


class ExplainabilityEngine:
    """
    Computes Grad-CAM and Grad-CAM++ heatmaps on ConvNeXt-V2 feature maps.

    The default target layer is the depthwise conv inside the last block
    of Stage 4 (the deepest spatial feature map before global pooling).
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()

        # Resolve target layer for ConvNeXt-V2
        if target_layer is not None:
            self.target_layer = target_layer
        else:
            self.target_layer = self._resolve_target_layer()

        self.gradients  = None
        self.activations = None
        self._register_hooks()

    def _resolve_target_layer(self):
        """
        Auto-resolve the last convolutional block of ConvNeXt-V2 Stage 4.
        Falls back gracefully to any available last layer.
        """
        try:
            # timm ConvNeXtV2 structure: _backbone.stages[3] is Stage 4
            stage4 = self.model._backbone.stages[3]
            # Last block in Stage 4
            last_block = list(stage4.children())[-1]
            # Depthwise conv is .conv_dw inside the block
            if hasattr(last_block, "conv_dw"):
                return last_block.conv_dw
            elif hasattr(last_block, "dwconv"):
                return last_block.dwconv
            else:
                return last_block
        except (AttributeError, IndexError):
            # Fallback: last module in backbone
            modules = list(self.model._backbone.modules())
            conv_layers = [m for m in modules if isinstance(m, torch.nn.Conv2d)]
            return conv_layers[-1] if conv_layers else modules[-1]

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_gradcam(self, input_tensor, target_class_idx=None):
        """Standard Grad-CAM using globally-averaged gradients."""
        self.model.zero_grad()
        out = self.model(input_tensor)
        logits = out["cls_logits"]

        if target_class_idx is None:
            target_class_idx = torch.argmax(logits, dim=1).item()

        score = logits[0, target_class_idx]
        score.backward(retain_graph=True)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        return self._normalize_cam(cam)

    def generate_gradcam_plus_plus(self, input_tensor, target_class_idx=None):
        """Grad-CAM++ with second/third-order gradient weighting for finer detail."""
        self.model.zero_grad()
        out = self.model(input_tensor)
        logits = out["cls_logits"]

        if target_class_idx is None:
            target_class_idx = torch.argmax(logits, dim=1).item()

        score = logits[0, target_class_idx]
        score.backward(retain_graph=True)

        grads = self.gradients
        acts  = self.activations

        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3
        sum_acts = torch.sum(acts, dim=(2, 3), keepdim=True)

        eps = 1e-7
        aij = grads_power_2 / (2 * grads_power_2 + sum_acts * grads_power_3 + eps)
        aij = torch.where(grads != 0, aij, torch.zeros_like(aij))

        weights = torch.sum(aij * F.relu(grads), dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = F.relu(cam)
        return self._normalize_cam(cam)

    def get_spatial_attention_map(self, input_tensor):
        """Returns the learned spatial attention map from the dual-head module."""
        with torch.no_grad():
            out = self.model(input_tensor)
            spatial_map = out["spatial_map"]
            return self._normalize_cam(spatial_map)

    def _normalize_cam(self, cam_tensor):
        cam = cam_tensor.squeeze().cpu().numpy()
        # Handle scalar edge-case
        if cam.ndim < 2:
            cam = np.zeros((7, 7))
        if HAS_CV2:
            cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)
        else:
            pil_img = Image.fromarray((cam * 255).clip(0, 255).astype(np.uint8))
            resized  = pil_img.resize((224, 224), Image.Resampling.BILINEAR)
            cam = np.array(resized, dtype=np.float32) / 255.0
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam
