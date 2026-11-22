"""
Two-Phase Multi-Task Training Pipeline for LungDeepNet
- Phase 1: Warmup with frozen DenseNet121 backbone (train attention & task heads)
- Phase 2: End-to-end fine-tuning with unfrozen DenseBlock 4 & Cosine Annealing LR
"""

import os
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from losses import MultiTaskLoss
from model import LungDeepNet


def train_epoch(model, dataloader, optimizer, criterion, scaler, device):
    """Executes one training epoch with mixed precision (AMP)."""
    model.train()
    total_loss, cls_loss_acc, sev_loss_acc = 0.0, 0.0, 0.0
    correct, total_samples = 0, 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch['image'].to(device)
        labels = batch['disease_label'].to(device)
        severities = batch['severity_score'].to(device)
        masks = batch['severity_mask'].to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss_dict = criterion(
                cls_logits=outputs['cls_logits'],
                pred_severity=outputs['severity'],
                true_cls=labels,
                true_severity=severities,
                severity_mask=masks
            )
            loss = loss_dict['loss_total']
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        bs = images.size(0)
        total_loss += loss.item() * bs
        cls_loss_acc += loss_dict['loss_cls'].item() * bs
        sev_loss_acc += loss_dict['loss_sev'].item() * bs
        
        preds = torch.argmax(outputs['cls_logits'], dim=1)
        correct += (preds == labels).sum().item()
        total_samples += bs
        
    return {
        'loss': total_loss / max(total_samples, 1),
        'cls_loss': cls_loss_acc / max(total_samples, 1),
        'sev_loss': sev_loss_acc / max(total_samples, 1),
        'accuracy': correct / max(total_samples, 1)
    }


def validate(model, dataloader, criterion, device):
    """Executes validation evaluation without gradient tracking."""
    model.eval()
    total_loss, cls_loss_acc, sev_loss_acc = 0.0, 0.0, 0.0
    correct, total_samples = 0, 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            images = batch['image'].to(device)
            labels = batch['disease_label'].to(device)
            severities = batch['severity_score'].to(device)
            masks = batch['severity_mask'].to(device)
            
            outputs = model(images)
            loss_dict = criterion(
                cls_logits=outputs['cls_logits'],
                pred_severity=outputs['severity'],
                true_cls=labels,
                true_severity=severities,
                severity_mask=masks
            )
            
            bs = images.size(0)
            total_loss += loss_dict['loss_total'].item() * bs
            cls_loss_acc += loss_dict['loss_cls'].item() * bs
            sev_loss_acc += loss_dict['loss_sev'].item() * bs
            
            preds = torch.argmax(outputs['cls_logits'], dim=1)
            correct += (preds == labels).sum().item()
            total_samples += bs
            
    return {
        'val_loss': total_loss / max(total_samples, 1),
        'val_cls_loss': cls_loss_acc / max(total_samples, 1),
        'val_sev_loss': sev_loss_acc / max(total_samples, 1),
        'val_accuracy': correct / max(total_samples, 1)
    }


def run_training_pipeline(model, train_loader, val_loader, epochs_phase1=5, epochs_phase2=20, device='cuda', save_path='checkpoints/best_model.pth'):
    """Full two-phase training loop."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    criterion = MultiTaskLoss(cls_weight=1.0, sev_weight=0.4)
    scaler = GradScaler()
    
    # Phase 1: Warmup attention heads with backbone frozen
    print("\n--- Starting Phase 1: Attention & Task Heads Warmup ---")
    model.freeze_backbone(True)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    
    for epoch in range(1, epochs_phase1 + 1):
        train_res = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_res = validate(model, val_loader, criterion, device)
        print(f"Phase 1 - Epoch [{epoch}/{epochs_phase1}] Loss: {train_res['loss']:.4f} Acc: {train_res['accuracy']*100:.2f}% | Val Loss: {val_res['val_loss']:.4f} Val Acc: {val_res['val_accuracy']*100:.2f}%")
        
    # Phase 2: Unfreeze DenseBlock 4 and fine-tune
    print("\n--- Starting Phase 2: Fine-Tuning DenseBlock 4 & Norm5 ---")
    model.unfreeze_denseblock4()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_phase2, eta_min=1e-6)
    
    best_val_loss = float('inf')
    for epoch in range(1, epochs_phase2 + 1):
        train_res = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_res = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Phase 2 - Epoch [{epoch}/{epochs_phase2}] Loss: {train_res['loss']:.4f} Acc: {train_res['accuracy']*100:.2f}% | Val Loss: {val_res['val_loss']:.4f} Val Acc: {val_res['val_accuracy']*100:.2f}%")
        
        if val_res['val_loss'] < best_val_loss:
            best_val_loss = val_res['val_loss']
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best checkpoint to {save_path}")
            
    print(f"\nTraining Complete. Best Validation Loss: {best_val_loss:.4f}")
