"""
Multi-Task Loss Function for Joint Classification and Severity Regression
Loss_total = Loss_cls + lambda * Loss_sev (masked for unannotated/normal cases)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """
    Jointly optimizes:
    1. Cross-Entropy Classification Loss for disease prediction
    2. Masked Smooth L1 Loss for disease-aware radiographic burden regression
    """
    def __init__(self, cls_weight=1.0, sev_weight=0.4, class_weights=None):
        super(MultiTaskLoss, self).__init__()
        self.cls_weight = cls_weight
        self.sev_weight = sev_weight
        
        if class_weights is not None:
            self.cls_criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.cls_criterion = nn.CrossEntropyLoss()
            
        self.sev_criterion = nn.SmoothL1Loss(reduction='none', beta=0.05)

    def forward(self, cls_logits, pred_severity, true_cls, true_severity, severity_mask):
        """
        Args:
            cls_logits: (B, num_classes)
            pred_severity: (B, 1)
            true_cls: (B,)
            true_severity: (B,)
            severity_mask: (B,) 1.0 if valid severity target exists, else 0.0
        """
        # 1. Classification Loss (Cross Entropy)
        loss_cls = self.cls_criterion(cls_logits, true_cls)
        
        # 2. Masked Severity Regression Loss (Smooth L1)
        pred_sev_flat = pred_severity.squeeze(1)
        raw_sev_loss = self.sev_criterion(pred_sev_flat, true_severity)
        
        # Mask out samples without validated severity ground truth
        masked_sev_loss = raw_sev_loss * severity_mask
        valid_count = torch.sum(severity_mask)
        
        if valid_count > 0:
            loss_sev = torch.sum(masked_sev_loss) / valid_count
        else:
            loss_sev = torch.tensor(0.0, device=cls_logits.device)
            
        # Total Multi-Task Objective
        loss_total = (self.cls_weight * loss_cls) + (self.sev_weight * loss_sev)
        
        return {
            'loss_total': loss_total,
            'loss_cls': loss_cls,
            'loss_sev': loss_sev
        }
