"""
Generates the Google Colab training notebook configured for Free Tier (Tesla T4 GPU).
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
                "# LungDeepNet: Google Colab Free GPU Training & Evaluation Suite\n",
                "### Multi-Task DenseNet121 + Dual-Head (Spatial & Channel) Attention\n",
                "**Hardware Target**: Google Colab Free Tier (NVIDIA Tesla T4 GPU with ~15GB VRAM)\n\n",
                "---\n",
                "> **Important Setup Step**:\n",
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
                "!pip install -q timm albumentations scikit-learn matplotlib seaborn tqdm Pillow"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "cell_upload_data"},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# [Cell 3] Ingest Dataset & Python Code Modules\n",
                "import os, zipfile\n",
                "from google.colab import files\n\n",
                "# If lungdeepnet_bundle.zip is not present, prompt to upload it\n",
                "if not os.path.exists('lungdeepnet_bundle.zip') and not os.path.exists('data'):\n",
                "    print('[*] Please upload lungdeepnet_bundle.zip (downloaded from the local web app):')\n",
                "    uploaded = files.upload()\n\n",
                "if os.path.exists('lungdeepnet_bundle.zip'):\n",
                "    print('[*] Extracting lungdeepnet_bundle.zip...')\n",
                "    with zipfile.ZipFile('lungdeepnet_bundle.zip', 'r') as z:\n",
                "        z.extractall('.')\n",
                "    print('[SUCCESS] Code and dataset extracted successfully.')\n\n",
                "os.makedirs('checkpoints', exist_ok=True)\n",
                "os.makedirs('results', exist_ok=True)"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "cell_verify_dataset"},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# [Cell 4] Verify Dataset Harmonization & Patient-Wise Splits\n",
                "import sys\n",
                "sys.path.insert(0, 'python')\n",
                "import pandas as pd\n",
                "from dataset import get_patient_wise_splits, ChestXRayMultiTaskDataset\n",
                "from preprocessing import get_train_transforms, get_val_transforms\n",
                "from torch.utils.data import DataLoader\n\n",
                "metadata_path = 'data/metadata.csv'\n",
                "assert os.path.exists(metadata_path), f'Metadata CSV not found at {metadata_path}'\n",
                "df = pd.read_csv(metadata_path)\n",
                "print(f'[*] Total Dataset Records: {len(df)}')\n",
                "print(df['disease'].value_counts())\n\n",
                "train_df, val_df, test_df = get_patient_wise_splits(metadata_path, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)\n",
                "print(f'[*] Patient-Wise Partition: {len(train_df)} Train, {len(val_df)} Val, {len(test_df)} Test samples')\n\n",
                "train_loader = DataLoader(ChestXRayMultiTaskDataset(train_df, transform=get_train_transforms(), is_train=True), batch_size=16, shuffle=True, pin_memory=True, num_workers=2)\n",
                "val_loader = DataLoader(ChestXRayMultiTaskDataset(val_df, transform=get_val_transforms(), is_train=False), batch_size=16, shuffle=False, pin_memory=True, num_workers=2)\n",
                "test_loader = DataLoader(ChestXRayMultiTaskDataset(test_df, transform=get_val_transforms(), is_train=False), batch_size=16, shuffle=False, pin_memory=True, num_workers=2)\n",
                "print('[SUCCESS] DataLoaders prepared for GPU streaming with zero patient leakage.')"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "cell_model_init"},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# [Cell 5] Initialize LungDeepNet (DenseNet121 + Dual-Head Attention)\n",
                "from model import LungDeepNet\n",
                "from losses import MultiTaskLoss\n\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "model = LungDeepNet(num_classes=4, reduction_ratio=16, dropout_rate=0.25, pretrained=True).to(device)\n",
                "criterion = MultiTaskLoss(cls_weight=1.0, sev_weight=0.4)\n\n",
                "total_params = sum(p.numel() for p in model.parameters())\n",
                "trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n",
                "print(f'[Model Initialized on {device}] Total Parameters: {total_params:,} | Trainable: {trainable_params:,}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## PHASE 1: EXECUTE GPU TRAINING PIPELINE\n",
                "Runs Two-Phase Training using FP16 Automatic Mixed Precision (AMP):\n",
                "- **Phase 1 (5 epochs)**: Warmup attention and multi-task heads with frozen backbone.\n",
                "- **Phase 2 (20 epochs)**: Fine-tune DenseBlock 4 and attention with Cosine Annealing learning rate.\n",
                "- Saves the best performing model weights to `checkpoints/best_model.pth`."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "cell_train_pipeline"},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# [Cell 6] Run Two-Phase GPU Mixed Precision Training Pipeline\n",
                "from train import run_training_pipeline\n\n",
                "print('=== Starting LungDeepNet GPU Training Pipeline on Colab T4 ===')\n",
                "run_training_pipeline(\n",
                "    model=model,\n",
                "    train_loader=train_loader,\n",
                "    val_loader=val_loader,\n",
                "    epochs_phase1=5,\n",
                "    epochs_phase2=20,\n",
                "    device='cuda',\n",
                "    save_path='checkpoints/best_model.pth'\n",
                ")\n",
                "print('\\n[SUCCESS] Training completed. Best checkpoint saved to checkpoints/best_model.pth')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## PHASE 2: STANDALONE EVALUATION & BENCHMARKING\n",
                "Strictly decoupled from training. Loads `checkpoints/best_model.pth`, freezes weights, and tests on the held-out test cohort."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "cell_evaluate"},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# [Cell 7] Standalone Evaluation on Held-Out Test Cohort\n",
                "from evaluate import evaluate_test_set\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import numpy as np\n\n",
                "# Reload model strictly from saved checkpoint\n",
                "eval_model = LungDeepNet(num_classes=4, pretrained=False).to(device)\n",
                "eval_model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))\n",
                "eval_model.eval()\n\n",
                "metrics = evaluate_test_set(eval_model, test_loader, device=device)\n\n",
                "# Plot Confusion Matrix\n",
                "plt.figure(figsize=(6, 5))\n",
                "classes = ['Normal', 'Pneumonia', 'Emphysema', 'Pulmonary Fibrosis']\n",
                "cm = np.array(metrics['confusion_matrix'])\n",
                "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)\n",
                "plt.title('Held-Out Test Cohort Confusion Matrix')\n",
                "plt.xlabel('Predicted Class')\n",
                "plt.ylabel('True Class')\n",
                "plt.tight_layout()\n",
                "plt.savefig('results/confusion_matrix.png', dpi=300)\n",
                "plt.show()\n",
                "print('[SUCCESS] Evaluation complete. Results saved to results/confusion_matrix.png')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## PHASE 3: EXPORT TRAINED MODEL CHECKPOINT\n",
                "Download `best_model.pth` back to your local computer and place it inside your project's `checkpoints/` directory."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {"id": "cell_download_checkpoint"},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# [Cell 8] Download Checkpoint to Your Local Computer\n",
                "from google.colab import files\n",
                "import os\n\n",
                "checkpoint_file = 'checkpoints/best_model.pth'\n",
                "if os.path.exists(checkpoint_file):\n",
                "    size_mb = os.path.getsize(checkpoint_file) / (1024 * 1024)\n",
                "    print(f'[*] Triggering download for {checkpoint_file} ({size_mb:.2f} MB)...')\n",
                "    files.download(checkpoint_file)\n",
                "    print('>>> Instructions: Move \"best_model.pth\" into your local project \"checkpoints/\" folder! <<<')\n",
                "else:\n",
                "    print('[ERROR] Checkpoint file not found! Please run the training cell first.')"
            ]
        }
    ]
}

nb_path1 = os.path.join(NOTEBOOKS_DIR, "lungdeepnet_colab_training.ipynb")
nb_path2 = os.path.join(PUBLIC_DIR, "lungdeepnet_colab_training.ipynb")

with open(nb_path1, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

with open(nb_path2, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"[SUCCESS] Written notebook to {nb_path1} and {nb_path2}")
