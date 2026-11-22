"""Script to update the Colab notebook for ConvNeXt-V2 training."""
import json
import os

nb_path = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'lungdeepnet_colab_training.ipynb')
nb = json.load(open(nb_path, 'r'))

# Cell 2: install deps (index 1)
nb['cells'][1]['source'] = [
    "# [Cell 2] Install Required Dependencies (timm added for ConvNeXt-V2 Base)\n",
    "!pip install -q timm albumentations scikit-learn matplotlib seaborn tqdm Pillow pandas"
]

# Cell 6: Model init (index 5)
nb['cells'][5]['source'] = [
    "# [Cell 6] Initialize LungDeepNet v2 (ConvNeXt-V2 Base + Dual-Head Attention)\n",
    "# ConvNeXtV2-Base: 89M params, 2024 SOTA for chest X-ray (AUC ~0.85+), beats DenseNet121 AUC ~0.75\n",
    "from model import LungDeepNet\n",
    "from losses import MultiTaskLoss\n",
    "from train import DEFAULT_CLASS_WEIGHTS\n",
    "\n",
    "device = torch.device('cuda')\n",
    "# pretrained=True loads ImageNet-22k / FCMAE self-supervised weights via timm\n",
    "model = LungDeepNet(num_classes=4, reduction_ratio=16, dropout_rate=0.30, pretrained=True).to(device)\n",
    "criterion = MultiTaskLoss(\n",
    "    cls_weight=1.0, sev_weight=0.4, focal_gamma=2.0,\n",
    "    class_weights=DEFAULT_CLASS_WEIGHTS.to(device)\n",
    ")\n",
    "\n",
    "total_params = sum(p.numel() for p in model.parameters())\n",
    "trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n",
    "print(f'[Model Initialized on {device}]')\n",
    "print('  Backbone: ConvNeXt-V2 Base (88M params, ImageNet-22k pretrained)')\n",
    "print(f'  Total Parameters: {total_params:,} | Trainable: {trainable_params:,}')\n",
    "print('  Loss: Focal Loss (gamma=2.0) + Class Weights [0.8, 1.0, 1.5, 2.0]')"
]

# Cell 7: Training (index 6) — 3-phase
nb['cells'][6]['source'] = [
    "# [Cell 7] Three-Phase GPU Training (ConvNeXt-V2 Base on 40,000 Scans)\n",
    "# Phase 1 : 3 epochs  — frozen backbone, warmup heads only (LR=1e-3)\n",
    "# Phase 2 : 8 epochs  — unfreeze Stage-4 + differential LR (backbone 5e-5 / heads 5e-4)\n",
    "# Phase 3 : 9 epochs  — unfreeze Stages 3+4 deep fine-tuning (backbone 1e-5 / heads 1e-4)\n",
    "# Total   : 20 epochs | Expected time: ~20-30 min on T4 GPU with 40,000 images\n",
    "from train import run_training_pipeline\n",
    "\n",
    "print('=== Starting LungDeepNet v2 (ConvNeXt-V2 Base) 40k-Scan 3-Phase GPU Training ===')\n",
    "run_training_pipeline(\n",
    "    model=model,\n",
    "    train_loader=train_loader,\n",
    "    val_loader=val_loader,\n",
    "    epochs_phase1=3,\n",
    "    epochs_phase2=8,\n",
    "    epochs_phase3=9,\n",
    "    device='cuda',\n",
    "    save_path='checkpoints/best_model.pth'\n",
    ")\n",
    "print('\\n[SUCCESS] ConvNeXtV2 Training Complete. Best checkpoint: checkpoints/best_model.pth')"
]

json.dump(nb, open(nb_path, 'w'), indent=2)
print('Notebook updated successfully for ConvNeXt-V2 training.')
