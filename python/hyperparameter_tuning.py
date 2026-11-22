"""
Objective 3: Hyperparameter Optimization (HPO) for LungDeepNet using Optuna
Optimizes:
- Channel Attention Reduction Ratio (r in [8, 16, 32])
- Spatial Attention Kernel Size (3 vs 7)
- Multi-Task Severity Loss Weight (lambda in [0.1, 0.8])
- Learning Rate (1e-5 to 1e-3 log scale)
- Dropout Rate (0.1 to 0.4)
- Optimizer Selection (AdamW vs SGD with Momentum vs RMSprop)
"""

import optuna
import torch
import torch.nn as nn
from model import LungDeepNet
from losses import MultiTaskLoss
from train import train_epoch, validate


def objective(trial, train_loader, val_loader, device='cuda'):
    # 1. Sample Hyperparameters via Tree-structured Parzen Estimator (TPE)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    reduction_ratio = trial.suggest_categorical('reduction_ratio', [8, 16, 32])
    kernel_size = trial.suggest_categorical('spatial_kernel_size', [3, 7])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4, step=0.05)
    sev_loss_weight = trial.suggest_float('sev_loss_weight', 0.1, 0.7, step=0.1)
    optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'SGD'])

    # 2. Instantiate Model with trial parameters
    model = LungDeepNet(
        num_classes=4,
        reduction_ratio=reduction_ratio,
        dropout_rate=dropout_rate,
        pretrained=True
    ).to(device)
    
    # Configure spatial kernel size
    model.spatial_attention.conv = nn.Conv2d(
        2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False
    ).to(device)

    # 3. Optimizer & Criterion
    if optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    criterion = MultiTaskLoss(cls_weight=1.0, sev_weight=sev_loss_weight)
    scaler = torch.cuda.amp.GradScaler()

    # 4. Pruned Short Training Loop (3 epochs per trial)
    best_val_f1 = 0.0
    for epoch in range(3):
        train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_res = validate(model, val_loader, criterion, device)
        
        # Report intermediate validation accuracy to Optuna for Median Pruning
        trial.report(val_res['val_accuracy'], epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        best_val_f1 = max(best_val_f1, val_res['val_accuracy'])

    return best_val_f1


def run_optuna_study(train_loader, val_loader, n_trials=25, device='cuda'):
    """Executes Bayesian Hyperparameter Optimization with TPE Sampler."""
    print(f"[Optuna HPO] Starting {n_trials} trials of Bayesian Optimization...")
    
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    study.optimize(lambda trial: objective(trial, train_loader, val_loader, device), n_trials=n_trials)

    print("\n" + "="*50)
    print("HYPERPARAMETER OPTIMIZATION (OPTUNA) SUMMARY")
    print("="*50)
    print(f"Best Trial Value (Val Accuracy): {study.best_value:.4f}")
    print("Optimal Hyperparameter Configuration:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*50)
    
    return study.best_params
