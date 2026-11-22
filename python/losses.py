"""
Multi-Task Loss — LungDeepNet v2
=================================
Loss_total = w_cls * FocalLoss(cls) + w_sev * MaskedSmoothL1(sev)

Improvements over v1 (plain CrossEntropy):
  • Focal Loss (γ=2.0, α=class_weights): Heavily penalises confident mis-classifications
    of rare/subtle classes (Emphysema, Pulmonary Fibrosis) that the network often collapses
    to "Normal" with standard CE.
  • Class-weighted CE baseline: Inverse-frequency weights supplied at training time.
  • Masked SmoothL1 severity loss: unchanged, unchanged masking for Normal samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.
    FL(p_t) = -α_t * (1 − p_t)^γ * log(p_t)

    Args:
        gamma      : Focusing parameter. Higher = harder examples get more weight. Default 2.0.
        class_weights: 1-D Tensor of per-class α weights (inverse frequency).
        reduction  : 'mean' or 'sum'.
    """

    def __init__(self, gamma: float = 2.0, class_weights: torch.Tensor = None, reduction: str = "mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.class_weights = class_weights  # (num_classes,)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, C) raw class logits
            targets : (B,)   integer class labels
        """
        log_probs = F.log_softmax(logits, dim=1)                 # (B, C)
        probs = torch.exp(log_probs)                             # (B, C)

        # Gather probabilities of the true class
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)   # (B,)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)           # (B,)

        # Focusing term
        focal_weight = (1.0 - pt) ** self.gamma                 # (B,)

        # Optional per-class α
        if self.class_weights is not None:
            alpha_t = self.class_weights.to(logits.device).gather(0, targets)  # (B,)
            focal_weight = alpha_t * focal_weight

        loss = -focal_weight * log_pt                            # (B,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MultiTaskLoss(nn.Module):
    """
    Jointly optimises:
      1. Focal Classification Loss   — robust to class imbalance & subtle pathologies
      2. Masked Smooth-L1 Severity Regression — severity only for diseased samples
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        sev_weight: float = 0.4,
        focal_gamma: float = 2.0,
        class_weights: torch.Tensor = None,
    ):
        super(MultiTaskLoss, self).__init__()
        self.cls_weight = cls_weight
        self.sev_weight = sev_weight

        # Focal loss for classification (replaces plain CrossEntropy)
        self.cls_criterion = FocalLoss(gamma=focal_gamma, class_weights=class_weights)

        # Smooth-L1 for severity regression
        self.sev_criterion = nn.SmoothL1Loss(reduction="none", beta=0.05)

    def forward(
        self,
        cls_logits: torch.Tensor,
        pred_severity: torch.Tensor,
        true_cls: torch.Tensor,
        true_severity: torch.Tensor,
        severity_mask: torch.Tensor,
    ) -> dict:
        """
        Args:
            cls_logits    : (B, num_classes)
            pred_severity : (B, 1)
            true_cls      : (B,)   integer disease labels
            true_severity : (B,)   ground-truth severity scores
            severity_mask : (B,)   1.0 if severity is valid; 0.0 for Normal samples
        """
        # 1. Focal Classification Loss
        loss_cls = self.cls_criterion(cls_logits, true_cls)

        # 2. Masked Severity Regression
        pred_sev_flat = pred_severity.squeeze(1)
        raw_sev_loss = self.sev_criterion(pred_sev_flat, true_severity)
        masked_sev_loss = raw_sev_loss * severity_mask
        valid_count = torch.sum(severity_mask)

        if valid_count > 0:
            loss_sev = torch.sum(masked_sev_loss) / valid_count
        else:
            loss_sev = torch.tensor(0.0, device=cls_logits.device, requires_grad=False)

        # 3. Combined Multi-Task Objective
        loss_total = (self.cls_weight * loss_cls) + (self.sev_weight * loss_sev)

        return {
            "loss_total": loss_total,
            "loss_cls": loss_cls,
            "loss_sev": loss_sev,
        }
