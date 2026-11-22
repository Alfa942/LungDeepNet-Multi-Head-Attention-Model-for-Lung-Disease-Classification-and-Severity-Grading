"""
Explainability & Localization Engine
Implements Grad-CAM, Grad-CAM++, LayerCAM, and Spatial Attention Map extraction.
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
    Computes explainability heatmaps on target DenseNet convolutional feature maps.
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer if target_layer is not None else self.model.features.norm5
        
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_gradcam(self, input_tensor, target_class_idx=None):
        """Standard Grad-CAM using global-averaged gradients."""
        self.model.zero_grad()
        out = self.model(input_tensor)
        logits = out['cls_logits']
        
        if target_class_idx is None:
            target_class_idx = torch.argmax(logits, dim=1).item()
            
        score = logits[0, target_class_idx]
        score.backward(retain_graph=True)
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        return self._normalize_cam(cam)

    def generate_gradcam_plus_plus(self, input_tensor, target_class_idx=None):
        """Grad-CAM++ using second- and third-order gradient weighting for fine detail."""
        self.model.zero_grad()
        out = self.model(input_tensor)
        logits = out['cls_logits']
        
        if target_class_idx is None:
            target_class_idx = torch.argmax(logits, dim=1).item()
            
        score = logits[0, target_class_idx]
        score.backward(retain_graph=True)
        
        grads = self.gradients
        acts = self.activations
        
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
        """Extracts the learned spatial attention map directly from the dual head."""
        with torch.no_grad():
            out = self.model(input_tensor)
            spatial_map = out['spatial_map']
            return self._normalize_cam(spatial_map)

    def _normalize_cam(self, cam_tensor):
        cam = cam_tensor.squeeze().cpu().numpy()
        if HAS_CV2:
            cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)
        else:
            # Resize with PIL without requiring OpenCV
            pil_img = Image.fromarray((cam * 255).astype(np.uint8))
            resized = pil_img.resize((224, 224), Image.Resampling.BILINEAR)
            cam = np.array(resized, dtype=np.float32) / 255.0
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam
