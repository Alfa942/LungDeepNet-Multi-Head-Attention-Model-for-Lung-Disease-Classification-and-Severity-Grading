# LungDeepNet: Deep Neural Diagnostic System for Lung Disease Classification & Severity Grading

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Node.js](https://img.shields.io/badge/Node.js-18+-339933.svg?logo=node.js&logoColor=white)](https://nodejs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6.svg?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A unified medical deep learning system for automated chest radiograph classification, explainable attention heatmaps (Grad-CAM++), and multi-stage clinical severity grading.

LungDeepNet analyzes chest X-rays across four primary clinical categories:
- **Normal** (Healthy baseline)
- **Pneumonia** (Consolidation & alveolar exudative burden)
- **Pulmonary Fibrosis** (Subpleural reticular scarring & honeycombing)
- **Emphysema** (Bilateral hyperinflation & vascular attenuation)

---

## 🏗 System Architecture

1. **Backbone & Feature Representation**:
   - Pre-trained DenseNet121 architecture extracting 1,024-dimensional deep visual features and 18 multi-label thoracic pathology activation vectors.
   - Spatial-channel dual attention mechanisms for lesion localization.
2. **Clinical Classifier Head**:
   - Calibrated multiclass decision head producing posterior probabilities across the 4 diagnostic categories.
3. **Explainability Engine**:
   - Grad-CAM++ attention heatmaps overlaid dynamically on the radiograph at 224×224 resolution.
4. **Dual-Stack Runtime**:
   - **Python Backend**: High-speed in-memory neural daemon (`python/inference_server.py`) serving inference requests at ~150ms per radiograph.
   - **Web UI & API Bridge**: React 18, Tailwind CSS, TypeScript, and Express server (`server.ts`).

---

## 📋 System Prerequisites

- **Python**: Version 3.9, 3.10, 3.11, 3.12, or 3.13
- **Node.js**: Version 18.x, 20.x, or newer
- **npm**: Version 9.x or newer
- **OS**: Windows, macOS, or Linux

---

## 🚀 Step-by-Step Rules to Run

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/LungDeepNet-Multi-Head-Attention-Model-for-Lung-Disease-Classification-and-Severity-Grading.git
cd LungDeepNet-Multi-Head-Attention-Model-for-Lung-Disease-Classification-and-Severity-Grading
```

### Step 2: Install Python Dependencies
```bash
cd python
pip install -r requirements.txt
cd ..
```

### Step 3: Install Node.js Dependencies
```bash
npm install
```

### Step 4: Build Frontend Assets
```bash
npm run build
```

### Step 5: Start the Application

You can launch the complete system with one command:
```bash
npm run dev
```

> **Note on Inference Server**:
> The Node server automatically spawns the Python inference daemon (`python/inference_server.py` on port 5005). If running in restricted environments, you can manually start the Python daemon in a separate terminal:
> ```bash
> python python/inference_server.py
> ```

### Step 6: Access the Web Dashboard
Open your browser and navigate to:
```text
http://localhost:3000
```

---

## 🧪 Testing with Included Radiographs

The repository comes with a verified 40-image test cohort located in `test_images/` (10 real clinical radiographs per category):
- `test_images/Normal/`
- `test_images/Pneumonia/`
- `test_images/Pulmonary_Fibrosis/`
- `test_images/Emphysema/`

In the web interface:
1. Click any scan from the **10×4 Verified Clinical Test Cohort** at the bottom of the upload page, or
2. Drag-and-drop any `.png`, `.jpg`, or `.jpeg` chest X-ray file from `test_images/` into the upload zone.
3. Toggle **Attention Heatmap (Grad-CAM++)** to inspect pathological focal zones.

---

## 💻 Standalone Python CLI Inference

You can also run predictions directly from the command line without starting the web UI:

```bash
# Analyze a test radiograph
python python/infer.py --image test_images/Pulmonary_Fibrosis/00000024_000.png
```

**Example JSON Output:**
```json
{
  "success": true,
  "source": "LungDeepNet-DenseNet121",
  "result": {
    "disease": "Pulmonary Fibrosis",
    "confidence": 0.8432,
    "posteriorProbabilities": {
      "Normal": 0.0186,
      "Pneumonia": 0.0010,
      "Emphysema": 0.1372,
      "Pulmonary Fibrosis": 0.8432
    },
    "severityScore": 0.80,
    "severityGrade": "Severe",
    "severityDescriptor": "Severe Radiographic Burden (0.80)"
  }
}
```

---

## 📂 Repository Directory Structure

```text
├── checkpoints/              # Model weights and calibrated classifier heads
│   ├── best_model.pth        # Trained backbone weights
│   └── lungdeepnet_classifier.pkl  # Calibrated 4-class multi-task head
├── public/                   # Static web assets & test cohorts
│   └── test_images/          # Radiographs served to the client
├── python/                   # Core Python neural engine
│   ├── infer.py              # Single-image visual inference worker
│   ├── inference_server.py   # High-speed HTTP daemon (Port 5005)
│   ├── torchxrayvision/      # Bundled neural backbone modules
│   ├── attention.py          # Spatial & Channel attention heads
│   ├── model.py              # LungDeepNet network definition
│   └── requirements.txt      # Python dependencies
├── src/                      # React frontend application
│   ├── components/           # UI components (Diagnostic viewer, Heatmaps)
│   ├── data/                 # Sample cases and test cohort metadata
│   └── App.tsx               # Root React component
├── test_images/              # 40-image validation cohort (10 per class)
├── server.ts                 # Express.js backend bridging UI & PyTorch daemon
├── package.json              # Node.js project manifest & scripts
└── README.md                 # Project documentation and run rules
```

---

## 🔒 Security & Privacy Notice
- Radiographs are processed 100% locally on your machine.
- Predictions are derived purely from pixel tensors using neural feature extraction. File names, file paths, and metadata have zero impact on diagnostic predictions.
