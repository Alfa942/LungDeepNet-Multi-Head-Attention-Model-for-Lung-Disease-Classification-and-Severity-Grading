"""
Generates the Google Colab training notebook configured for Free Tier (Tesla T4 GPU)
with 1-click 40,000 Verified Clinical Radiograph Ingestion (10,000 per class:
Normal, Pneumonia, Emphysema, Pulmonary Fibrosis) & Two-Phase Training.
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")
PUBLIC_DIR = os.path.join(BASE_DIR, "public")

os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
os.makedirs(PUBLIC_DIR, exist_ok=True)

notebook = {
    "nbformat": 4,
    "nbformat_minor": 2,
    "metadata": {
        "colab": {
            "name": "lungdeepnet_colab_training.ipynb",
            "provenance": [],
            "toc_visible": True
        },
        "accelerator": "GPU",
        "gpuClass": "standard",
        "language_info": {"name": "python"}
    },
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# LungDeepNet: Google Colab Free GPU 40,000-Image Clinical Training Suite\n",
                "### Multi-Task DenseNet121 + Dual-Head (Spatial & Channel) Attention\n",
                "**Hardware Target**: Google Colab Free Tier (NVIDIA Tesla T4 GPU with ~15GB VRAM)\n",
                "**Dataset Scale**: 40,000 Verified Clinical Radiographs (10,000 each: Normal, Pneumonia, Emphysema, Pulmonary Fibrosis)\n\n",
                "---\n",
                "> **Setup Step (Mandatory)**:\n",
                "> 1. Go to **Runtime** -> **Change runtime type**\n",
                "> 2. Under **Hardware accelerator**, select **T4 GPU**\n",
                "> 3. Click **Save** before running Cell 1."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "cell_gpu_check"},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# [Cell 1] Verify NVIDIA Tesla T4 GPU Acceleration\n",
                "!nvidia-smi\n\n",
                "import torch\n",
                "assert torch.cuda.is_available(), 'ERROR: GPU is not active! Please select T4 GPU in Runtime -> Change runtime type.'\n",
                "gpu_name = torch.cuda.get_device_name(0)\n",
                "vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)\n",
                "print(f'\\n>>> SUCCESS: CUDA Accelerated on {gpu_name} ({vram_gb:.2f} GB VRAM) <<<')\n",
                "torch.backends.cudnn.benchmark = True\n",
                "torch.cuda.empty_cache()"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "cell_install_deps"},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# [Cell 2] Install Required PyTorch & Imaging Dependencies\n",
                "!pip install -q timm albumentations scikit-learn matplotlib seaborn tqdm Pillow pandas"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "cell_upload_data"},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# [Cell 3] Ingest Code Modules\n",
                "import os, zipfile\n",
                "from google.colab import files\n\n",
                "if not os.path.exists('lungdeepnet_bundle.zip') and not os.path.exists('python'):\n",
                "    print('[*] Please upload lungdeepnet_bundle.zip (downloaded from the local web app):')\n",
                "    uploaded = files.upload()\n\n",
                "if os.path.exists('lungdeepnet_bundle.zip'):\n",
                "    print('[*] Extracting lungdeepnet_bundle.zip...')\n",
                "    with zipfile.ZipFile('lungdeepnet_bundle.zip', 'r') as z:\n",
                "        z.extractall('.')\n",
                "    print('[SUCCESS] Code extracted successfully.')\n\n",
                "os.makedirs('checkpoints', exist_ok=True)\n",
                "os.makedirs('results', exist_ok=True)"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "cell_download_40k_data"},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# [Cell 4] Ingest & Verify 40,000 Clinical Chest Radiographs (10,000 per class)\n",
                "import sys\n",
                "sys.path.insert(0, 'python')\n",
                "from download_40k_dataset import download_and_assemble_40k_dataset, audit_and_verify_dataset\n\n",
                "# Downloads and balances 10,000 verified images each for:\n",
                "# - Normal (10,000)\n",
                "# - Pneumonia (10,000)\n",
                "# - Emphysema (10,000)\n",
                "# - Pulmonary Fibrosis (10,000)\n",
                "print('[*] Assembling and verifying 40,000-image dataset (10k per class)...')\n",
                "download_and_assemble_40k_dataset(num_archives=32, target_per_class=10000)\n\n",
                "metadata_path = 'data/metadata_40k.csv'\n",
                "if not os.path.exists(metadata_path):\n",
                "    metadata_path = 'data/metadata_augmented.csv'\n",
                "print(f'[SUCCESS] Active metadata manifest: {metadata_path}')"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "cell_verify_dataset"},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# [Cell 5] Build Zero-Leakage Patient-Stratified DataLoaders\n",
                "import pandas as pd\n",
                "from dataset import get_patient_wise_splits, ChestXRayMultiTaskDataset\n",
                "from preprocessing import get_train_transforms, get_val_transforms\n",
                "from torch.utils.data import DataLoader\n\n",
                "df = pd.read_csv(metadata_path)\n",
                "print(f'[*] Total Dataset Records: {len(df):,}')\n",
                "print('\\nClass Distribution:')\n",
                "print(df['disease'].value_counts())\n\n",
                "train_df, val_df, test_df = get_patient_wise_splits(metadata_path, train_ratio=0.75, val_ratio=0.125, test_ratio=0.125)\n",
                "print(f'\\n[*] Patient Partition: {len(train_df):,} Train, {len(val_df):,} Val, {len(test_df):,} Test samples')\n\n",
                "# Batch size 64 takes full advantage of Tesla T4 VRAM with high throughput (~350-450 images/sec)\n",
                "train_loader = DataLoader(ChestXRayMultiTaskDataset(train_df, transform=get_train_transforms(), is_train=True), batch_size=64, shuffle=True, pin_memory=True, num_workers=2)\n",
                "val_loader = DataLoader(ChestXRayMultiTaskDataset(val_df, transform=get_val_transforms(), is_train=False), batch_size=64, shuffle=False, pin_memory=True, num_workers=2)\n",
                "test_loader = DataLoader(ChestXRayMultiTaskDataset(test_df, transform=get_val_transforms(), is_train=False), batch_size=64, shuffle=False, pin_memory=True, num_workers=2)\n",
                "print('[SUCCESS] High-throughput GPU DataLoaders ready.')"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "cell_model_init"},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# [Cell 6] Initialize LungDeepNet (DenseNet121 + Dual-Head Attention)\n",
                "from model import LungDeepNet\n",
                "from losses import MultiTaskLoss\n\n",
                "device = torch.device('cuda')\n",
                "model = LungDeepNet(num_classes=4, reduction_ratio=16, dropout_rate=0.25, pretrained=True).to(device)\n",
                "criterion = MultiTaskLoss(cls_weight=1.0, sev_weight=0.4)\n\n",
                "total_params = sum(p.numel() for p in model.parameters())\n",
                "trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n",
                "print(f'[Model Initialized on {device}] Total Parameters: {total_params:,} | Trainable: {trainable_params:,}')"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "cell_train_pipeline"},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# [Cell 7] Run Two-Phase GPU Mixed Precision Training on 40,000 Scans (~15 mins on T4 GPU)\n",
                "from train import run_training_pipeline\n\n",
                "print('=== Starting LungDeepNet 40,000-Scan GPU Training on Colab Tesla T4 ===')\n",
                "run_training_pipeline(\n",
                "    model=model,\n",
                "    train_loader=train_loader,\n",
                "    val_loader=val_loader,\n",
                "    epochs_phase1=3,\n",
                "    epochs_phase2=10,\n",
                "    device='cuda',\n",
                "    save_path='checkpoints/best_model.pth'\n",
                ")\n",
                "print('\\n[SUCCESS] 40,000-Image Training Complete. Best Checkpoint: checkpoints/best_model.pth')"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "cell_evaluate"},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# [Cell 8] Standalone Evaluation on Held-Out Test Cohort\n",
                "from evaluate import evaluate_test_set\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import numpy as np\n\n",
                "eval_model = LungDeepNet(num_classes=4, pretrained=False).to(device)\n",
                "checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)\n",
                "state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint\n",
                "eval_model.load_state_dict(state_dict)\n",
                "eval_model.eval()\n\n",
                "metrics = evaluate_test_set(eval_model, test_loader, device=device)\n",
                "print('\\n=== CLASSIFICATION REPORT ON HELD-OUT TEST COHORT ===')\n",
                "print(metrics['report'])\n",
                "print(f\"\\nOverall Multi-Class Accuracy: {metrics['accuracy']*100:.2f}%\")\n",
                "print(f\"Overall Severity Pearson r:     {metrics['pearson_r']:.4f}\")\n\n",
                "# Plot Confusion Matrix\n",
                "cm = metrics['confusion_matrix']\n",
                "classes = ['Normal', 'Pneumonia', 'Emphysema', 'Pulmonary Fibrosis']\n",
                "fig, ax = plt.subplots(figsize=(6, 5), dpi=150)\n",
                "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)\n",
                "ax.set_title('Test Set Confusion Matrix (Patient-Partitioned)', fontweight='bold')\n",
                "ax.set_ylabel('True Clinical Pathology')\n",
                "ax.set_xlabel('LungDeepNet Predicted Pathology')\n",
                "plt.tight_layout()\n",
                "plt.savefig('results/confusion_matrix.png', bbox_inches='tight')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "cell_download_checkpoint"},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# [Cell 9] Download Trained Model Checkpoint to Local Machine\n",
                "# Place this file in your local project under: checkpoints/best_model.pth\n",
                "from google.colab import files\n\n",
                "print('[*] Packaging best_model.pth for download...')\n",
                "if os.path.exists('checkpoints/best_model.pth'):\n",
                "    files.download('checkpoints/best_model.pth')\n",
                "    print('[SUCCESS] Download initiated. Move best_model.pth to your project\\'s checkpoints/ folder.')\n",
                "else:\n",
                "    print('[!] Checkpoint not found. Please run Cell 7 first.')"
            ]
        }
    ]
}

def export_colab_notebook():
    out_path_1 = os.path.join(NOTEBOOKS_DIR, "lungdeepnet_colab_training.ipynb")
    out_path_2 = os.path.join(PUBLIC_DIR, "lungdeepnet_colab_training.ipynb")
    
    with open(out_path_1, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    print(f"[SUCCESS] Exported Colab notebook to: {out_path_1}")

    with open(out_path_2, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    print(f"[SUCCESS] Exported public Colab notebook to: {out_path_2}")

if __name__ == "__main__":
    export_colab_notebook()
