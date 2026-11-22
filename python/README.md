# LungDeepNet: Python / PyTorch Deep Learning Pipeline

This folder contains the complete, reproducible Python & PyTorch codebase for **LungDeepNet**:
**DenseNet121 + Dual-Head Attention + Severity Grading + Explainable AI (Grad-CAM)**

---

## 📂 File Directory

| File | Purpose | Corresponding Project Objective |
| :--- | :--- | :--- |
| `model.py` | Full PyTorch architecture with DenseNet121 backbone, Spatial & Channel attention heads, and dual classification/regression outputs. | **Objective 2** (Model Architecture) |
| `attention.py` | Standalone modules for `SpatialAttention` (7x7 Conv + Sigmoid) and `ChannelAttention` (Squeeze-and-Excitation MLP). | **Objective 2** (Dual-Head Attention) |
| `losses.py` | `MultiTaskLoss` class computing $L_{\text{total}} = L_{\text{ce}} + \lambda L_{\text{sev}}$ with masking for Normal scans. | **Objective 2** (Multi-Task Optimization) |
| `preprocessing.py` | Albumentations image augmentations, Gaussian noise filtering, and ImageNet standardization. | **Objective 1** (Preprocessing & Denoising) |
| `dataset.py` | Patient-wise grouped train/val/test splits, DICOM/PNG reader, and PyTorch `Dataset`. | **Objective 1 & 2** (Data Pipeline) |
| `download_dataset.py` | Ingestion instructions for COVID-19/NIH datasets & instant demo sample generator. | **Data Ingestion** |
| `train.py` | Two-phase training pipeline: Phase 1 (warmup attention heads), Phase 2 (fine-tune DenseBlock 4). | **Objective 2** (Model Training) |
| `evaluate.py` | Comprehensive metrics: Multi-class AUROC, Sensitivity, Specificity, F1, Confusion Matrix, and Severity MAE. | **Objective 4** (Comparative Evaluation) |
| `hyperparameter_tuning.py` | Optuna Bayesian optimization study searching optimal learning rate, attention reduction ratio $r$, and loss weights. | **Objective 3** (Hyperparameter Optimization) |
| `gradcam.py` | Explainable AI engine extracting Grad-CAM, Grad-CAM++, and Spatial Attention lesion maps. | **Objective 2 & 4** (Interpretability) |
| `run_pipeline.py` | Command-line runner to verify and execute the pipeline. | Execution CLI |
| `requirements.txt` | Python dependencies. | Environment |

---

## 🚀 How to Run in Google Colab / Local Machine

1. Clone or download this `python/` folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Test a quick forward pass of the model:
   ```bash
   python run_pipeline.py --mode demo
   ```
4. Run Hyperparameter Optimization (Optuna):
   ```bash
   python run_pipeline.py --mode optuna --trials 20
   ```
5. Run full training:
   ```bash
   python run_pipeline.py --mode train --epochs 25
   ```
