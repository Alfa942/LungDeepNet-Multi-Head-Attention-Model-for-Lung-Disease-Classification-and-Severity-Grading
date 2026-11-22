"""
Three-Phase Training Pipeline — LungDeepNet v2 (ConvNeXt-V2 Base)
===================================================================
Phase 1  — Warmup  : Backbone fully frozen. Train attention modules + task heads only.
           Epochs  : 3   LR = 1e-3  (AdamW)
Phase 2  — Stage 4 : Unfreeze ConvNeXt Stage 4 + Norm.
           Epochs  : 8   LR = 5e-5 backbone / 5e-4 heads  (differential LR)
Phase 3  — Stages 3+4: Unfreeze ConvNeXt Stages 3 & 4 for deep fine-tuning.
           Epochs  : 9   LR = 1e-5 backbone / 1e-4 heads  Cosine Annealing

Improvements over v1:
  - Three-phase training  → allows gradual backbone adaptation without catastrophic forgetting
  - Differential learning rates per parameter group  → backbone gets ×10 lower LR than heads
  - Focal Loss (γ=2.0) with inverse-frequency class weights  → fixes Normal-prior dominance
  - Cosine Annealing LR scheduler  → smooth convergence without hard LR drops
  - Label smoothing (ε=0.05) in focal loss  → prevents over-confident wrong predictions
"""

import os
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from losses import MultiTaskLoss


# ---------------------------------------------------------------------- #
# Class weights: inverse-frequency for 4-class balanced 40k dataset      #
# Normal (10k) : weight = 1.0 (most common, suppress dominance slightly)  #
# Pneumonia    : weight = 1.0                                             #
# Emphysema    : weight = 1.5 (harder to detect; mild texture change)     #
# Pulmonary F. : weight = 2.0 (most subtle; worst false-negative rate)    #
# ---------------------------------------------------------------------- #
DEFAULT_CLASS_WEIGHTS = torch.tensor([0.8, 1.0, 1.5, 2.0], dtype=torch.float32)


def _build_param_groups(model, backbone_lr: float, head_lr: float) -> list:
    """Split model parameters into backbone and head groups with differential LR."""
    backbone_params = list(model._backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params,     "lr": head_lr},
    ]


def train_epoch(model, dataloader, optimizer, criterion, scaler, device):
    """One mixed-precision training epoch."""
    model.train()
    total_loss = cls_loss_acc = sev_loss_acc = 0.0
    correct = total_samples = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        images    = batch["image"].to(device)
        labels    = batch["disease_label"].to(device)
        severities = batch["severity_score"].to(device)
        masks     = batch["severity_mask"].to(device)

        optimizer.zero_grad()
        with autocast():
            outputs   = model(images)
            loss_dict = criterion(
                cls_logits=outputs["cls_logits"],
                pred_severity=outputs["severity"],
                true_cls=labels,
                true_severity=severities,
                severity_mask=masks,
            )
            loss = loss_dict["loss_total"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        total_loss    += loss.item() * bs
        cls_loss_acc  += loss_dict["loss_cls"].item() * bs
        sev_loss_acc  += loss_dict["loss_sev"].item() * bs
        preds = torch.argmax(outputs["cls_logits"], dim=1)
        correct       += (preds == labels).sum().item()
        total_samples += bs

    n = max(total_samples, 1)
    return {
        "loss":     total_loss    / n,
        "cls_loss": cls_loss_acc  / n,
        "sev_loss": sev_loss_acc  / n,
        "accuracy": correct       / n,
    }


def validate(model, dataloader, criterion, device):
    """Validation without gradient tracking."""
    model.eval()
    total_loss = cls_loss_acc = sev_loss_acc = 0.0
    correct = total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            images     = batch["image"].to(device)
            labels     = batch["disease_label"].to(device)
            severities = batch["severity_score"].to(device)
            masks      = batch["severity_mask"].to(device)

            outputs   = model(images)
            loss_dict = criterion(
                cls_logits=outputs["cls_logits"],
                pred_severity=outputs["severity"],
                true_cls=labels,
                true_severity=severities,
                severity_mask=masks,
            )

            bs = images.size(0)
            total_loss    += loss_dict["loss_total"].item() * bs
            cls_loss_acc  += loss_dict["loss_cls"].item()   * bs
            sev_loss_acc  += loss_dict["loss_sev"].item()   * bs
            preds = torch.argmax(outputs["cls_logits"], dim=1)
            correct       += (preds == labels).sum().item()
            total_samples += bs

    n = max(total_samples, 1)
    return {
        "val_loss":     total_loss    / n,
        "val_cls_loss": cls_loss_acc  / n,
        "val_sev_loss": sev_loss_acc  / n,
        "val_accuracy": correct       / n,
    }


def run_training_pipeline(
    model,
    train_loader,
    val_loader,
    epochs_phase1: int = 3,
    epochs_phase2: int = 8,
    epochs_phase3: int = 9,
    device: str = "cuda",
    save_path: str = "checkpoints/best_model.pth",
    class_weights: torch.Tensor = None,
):
    """
    Full three-phase training pipeline for LungDeepNet v2 (ConvNeXt-V2 Base).

    Total default epochs: 20  (3 + 8 + 9)
    On T4 GPU with 40k images (batch 64):  ~18-25 minutes total.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if class_weights is None:
        class_weights = DEFAULT_CLASS_WEIGHTS

    cw = class_weights.to(device)
    criterion = MultiTaskLoss(
        cls_weight=1.0,
        sev_weight=0.4,
        focal_gamma=2.0,
        class_weights=cw,
    )
    scaler = GradScaler()
    best_val_loss = float("inf")

    # ------------------------------------------------------------------ #
    # Phase 1 — Warmup: backbone frozen, heads + attention only           #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print(" PHASE 1 / 3 — Warmup  (Frozen Backbone, Heads Only)")
    print(f"  LR=1e-3  |  Focal Loss γ=2  |  Epochs {epochs_phase1}")
    print("=" * 60)

    model.freeze_backbone(True)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4,
    )

    for epoch in range(1, epochs_phase1 + 1):
        tr = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        va = validate(model, val_loader, criterion, device)
        print(
            f"  P1 Epoch [{epoch:02d}/{epochs_phase1}]  "
            f"Loss={tr['loss']:.4f}  Acc={tr['accuracy']*100:.2f}%  |  "
            f"Val Loss={va['val_loss']:.4f}  Val Acc={va['val_accuracy']*100:.2f}%"
        )
        if va["val_loss"] < best_val_loss:
            best_val_loss = va["val_loss"]
            torch.save(model.state_dict(), save_path)
            print(f"    ✓ Saved best checkpoint (Phase 1 — val_loss={best_val_loss:.4f})")

    # ------------------------------------------------------------------ #
    # Phase 2 — Unfreeze Stage 4: differential LR                        #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print(" PHASE 2 / 3 — Stage-4 Fine-Tuning (Differential LR)")
    print(f"  Backbone LR=5e-5  |  Heads LR=5e-4  |  Epochs {epochs_phase2}")
    print("=" * 60)

    model.unfreeze_stage4()
    param_groups = _build_param_groups(model, backbone_lr=5e-5, head_lr=5e-4)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs_phase2, eta_min=1e-6
    )

    for epoch in range(1, epochs_phase2 + 1):
        tr = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        va = validate(model, val_loader, criterion, device)
        scheduler.step()
        print(
            f"  P2 Epoch [{epoch:02d}/{epochs_phase2}]  "
            f"Loss={tr['loss']:.4f}  Acc={tr['accuracy']*100:.2f}%  |  "
            f"Val Loss={va['val_loss']:.4f}  Val Acc={va['val_accuracy']*100:.2f}%"
        )
        if va["val_loss"] < best_val_loss:
            best_val_loss = va["val_loss"]
            torch.save(model.state_dict(), save_path)
            print(f"    ✓ Saved best checkpoint (Phase 2 — val_loss={best_val_loss:.4f})")

    # ------------------------------------------------------------------ #
    # Phase 3 — Unfreeze Stages 3+4: deep end-to-end fine-tuning         #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print(" PHASE 3 / 3 — Stages 3+4 Deep Fine-Tuning")
    print(f"  Backbone LR=1e-5  |  Heads LR=1e-4  |  Epochs {epochs_phase3}")
    print("=" * 60)

    model.unfreeze_stages3_4()
    param_groups = _build_param_groups(model, backbone_lr=1e-5, head_lr=1e-4)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs_phase3, eta_min=5e-7
    )

    for epoch in range(1, epochs_phase3 + 1):
        tr = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        va = validate(model, val_loader, criterion, device)
        scheduler.step()
        print(
            f"  P3 Epoch [{epoch:02d}/{epochs_phase3}]  "
            f"Loss={tr['loss']:.4f}  Acc={tr['accuracy']*100:.2f}%  |  "
            f"Val Loss={va['val_loss']:.4f}  Val Acc={va['val_accuracy']*100:.2f}%"
        )
        if va["val_loss"] < best_val_loss:
            best_val_loss = va["val_loss"]
            torch.save(model.state_dict(), save_path)
            print(f"    ✓ Saved best checkpoint (Phase 3 — val_loss={best_val_loss:.4f})")

    print(f"\n[DONE] Training Complete. Best Validation Loss: {best_val_loss:.4f}")
    print(f"[DONE] Best checkpoint saved to: {save_path}")
