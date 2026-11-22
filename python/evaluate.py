"""
Comprehensive Multi-Task Evaluation & Metrics Computation
Evaluates:
- Multi-class AUROC (One-vs-Rest)
- Macro Sensitivity, Specificity, F1-Score
- Multi-class Confusion Matrix
- Continuous Severity Mean Absolute Error (MAE), RMSE, and Pearson Correlation
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, mean_absolute_error,
    mean_squared_error
)
from scipy.stats import pearsonr, spearmanr


def evaluate_test_set(model, test_loader, device):
    """
    Evaluates model performance on completely held-out test cohort.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    all_pred_sev, all_true_sev, all_masks = [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['disease_label'].to(device)
            severities = batch['severity_score'].to(device)
            masks = batch['severity_mask'].to(device)
            
            outputs = model(images)
            probs = F.softmax(outputs['cls_logits'], dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_pred_sev.extend(outputs['severity'].squeeze().cpu().numpy())
            all_true_sev.extend(severities.cpu().numpy())
            all_masks.extend(masks.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_pred_sev = np.array(all_pred_sev)
    all_true_sev = np.array(all_true_sev)
    all_masks = np.array(all_masks)
    
    # 1. Classification Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    cm = confusion_matrix(all_labels, all_preds)
    
    # 2. Continuous Severity Metrics (evaluated only on pathological samples)
    valid_idx = (all_masks > 0.5)
    if np.sum(valid_idx) > 0:
        valid_true_sev = all_true_sev[valid_idx]
        valid_pred_sev = all_pred_sev[valid_idx]
        mae = mean_absolute_error(valid_true_sev, valid_pred_sev)
        rmse = float(np.sqrt(mean_squared_error(valid_true_sev, valid_pred_sev)))
        pearson_r, _ = pearsonr(valid_true_sev, valid_pred_sev)
        spearman_rho, _ = spearmanr(valid_true_sev, valid_pred_sev)
    else:
        mae, rmse, pearson_r, spearman_rho = 0.0, 0.0, 0.0, 0.0
        
    metrics = {
        'accuracy': float(acc),
        'macro_f1': float(f1),
        'macro_sensitivity': float(recall),
        'macro_precision': float(precision),
        'macro_auroc': float(auroc),
        'confusion_matrix': cm.tolist(),
        'severity_mae': float(mae),
        'severity_rmse': float(rmse),
        'severity_pearson_r': float(pearson_r),
        'severity_spearman_rho': float(spearman_rho)
    }
    
    print("\n" + "="*50)
    print("TEST COHORT BENCHMARK RESULTS")
    print("="*50)
    print(f"Accuracy:            {metrics['accuracy']*100:.2f}%")
    print(f"Macro F1-Score:      {metrics['macro_f1']:.4f}")
    print(f"Macro Sensitivity:   {metrics['macro_sensitivity']:.4f}")
    print(f"Macro AUROC:         {metrics['macro_auroc']:.4f}")
    print(f"Severity MAE:        {metrics['severity_mae']:.4f}")
    print(f"Severity RMSE:       {metrics['severity_rmse']:.4f}")
    print(f"Pearson Correlation: {metrics['severity_pearson_r']:.4f}")
    print("="*50)
    
    return metrics
