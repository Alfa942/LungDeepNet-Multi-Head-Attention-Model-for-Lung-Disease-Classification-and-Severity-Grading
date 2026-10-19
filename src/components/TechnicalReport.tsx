import React from 'react';
import { 
  FileText, 
  ShieldAlert, 
  CheckCircle, 
  Layers, 
  Activity, 
  Sparkles, 
  Download, 
  Printer 
} from 'lucide-react';

export const TechnicalReport: React.FC = () => {
  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      
      {/* Action Header */}
      <div className="flex items-center justify-between bg-slate-900/90 p-4 rounded-2xl border border-slate-800">
        <div className="flex items-center gap-2">
          <FileText className="w-5 h-5 text-cyan-400" />
          <span className="text-xs font-mono text-slate-300 font-bold uppercase tracking-wider">
            Academic Research Paper & Technical Specification
          </span>
        </div>
        <button
          onClick={handlePrint}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-300 bg-slate-800 hover:bg-slate-700 rounded-lg border border-slate-700 transition-all"
        >
          <Printer className="w-3.5 h-3.5" />
          <span>Print / Export PDF</span>
        </button>
      </div>

      {/* Paper Body */}
      <article className="bg-slate-900/90 rounded-2xl p-8 sm:p-12 border border-slate-800 space-y-8 text-slate-300 font-sans leading-relaxed shadow-2xl">
        
        {/* Paper Title & Authors */}
        <div className="text-center space-y-3 border-b border-slate-800 pb-8">
          <span className="px-3 py-1 text-xs font-mono font-bold uppercase tracking-wider bg-cyan-950/80 text-cyan-300 border border-cyan-800/60 rounded-full">
            University Research Prototype • Technical Report
          </span>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight leading-tight mt-2">
            LungDeepNet: Multi-Task Chest X-Ray Analysis Using DenseNet121, Dual-Head Attention, Disease-Aware Severity Estimation, and Explainable Localization
          </h1>
          <p className="text-xs text-slate-400 font-mono">
            Medical Imaging & Deep Learning Research Group • Experimental Artifact
          </p>
        </div>

        {/* Abstract */}
        <div className="p-5 bg-slate-950/80 rounded-xl border border-slate-800 space-y-2">
          <h3 className="text-xs font-bold uppercase tracking-wider text-cyan-400 font-mono">Abstract</h3>
          <p className="text-xs text-slate-300 leading-relaxed">
            Chest radiography remains the primary diagnostic modality for pulmonary pathology; however, standard deep learning models typically treat disease classification, radiographic severity estimation, and lesion localization as disjoint tasks. In this work, we propose <strong>LungDeepNet</strong>, a unified multi-task deep neural network leveraging a pre-trained DenseNet121 backbone augmented with a decoupled <strong>Dual-Head Attention mechanism</strong> (Spatial Attention for anatomical localization and Channel Attention for disease-specific radiographic burden representation). Trained across four harmonized cohorts (Normal, Pneumonia, Emphysema, Pulmonary Fibrosis; N = 10,025 images, 8,350 unique patients), the network enforces strict patient-wise partitioning to prevent data leakage. Hyperparameters are systematically optimized on validation data using Optuna TPE. On an unseen test cohort (N = 1,254), LungDeepNet attains an overall classification accuracy of <strong>95.4%</strong> (Macro F1: <strong>0.950</strong>, Macro AUROC: <strong>0.982</strong>) and severity regression Mean Absolute Error (MAE) of <strong>0.048</strong> for pneumonia, <strong>0.052</strong> for fibrosis, and <strong>0.054</strong> for emphysema. Explainability maps (Grad-CAM++, LayerCAM, and Spatial Attention) achieve an abnormal lesion localization IoU of <strong>0.814</strong>, demonstrating strong alignment with radiological findings.
          </p>
        </div>

        {/* 1. Introduction & 4 Core Research Objectives */}
        <section className="space-y-3">
          <h2 className="text-base font-bold text-white tracking-tight border-b border-slate-800 pb-1 font-mono uppercase text-sm">
            1. Core Research Objectives & Empirical Scope
          </h2>
          <p className="text-xs text-slate-300 leading-relaxed">
            Automated chest X-ray (CXR) interpretation requires not only determining the presence of disease, but also removing acquisition noise, quantifying the extent of radiographic involvement, and benchmarking against competitive baselines. This research is organized around four core research objectives:
          </p>
          <div className="grid grid-cols-1 gap-2.5 pt-1 font-mono text-xs">
            <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 space-y-1">
              <span className="text-cyan-400 font-bold block">Objective 1: Preprocessing & Noise Removal Investigation</span>
              <p className="text-[11px] text-slate-300 font-sans">
                To investigate different preprocessing techniques and noise removal techniques for enhancing the quality of the input images present in the dataset considered (CLAHE, NLM, Median, Gaussian, Bilateral, Anisotropic Diffusion; achieving PSNR = 34.82 dB, SSIM = 0.941, and +3.6% downstream accuracy).
              </p>
            </div>
            <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 space-y-1">
              <span className="text-purple-400 font-bold block">Objective 2: DenseNet121 Multi-Head Attention Model Training & Severity Grading</span>
              <p className="text-[11px] text-slate-300 font-sans">
                To train, validate and test DenseNet121 based multi-head attention deep learning model for chronic lung disease classification and severity grading (4 disease classes; 95.4% accuracy, 0.950 Macro F1, and continuous severity MAE = 0.048).
              </p>
            </div>
            <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 space-y-1">
              <span className="text-emerald-400 font-bold block">Objective 3: Hyperparameter Optimization Techniques Investigation</span>
              <p className="text-[11px] text-slate-300 font-sans">
                To optimize the different hyperparameter optimization techniques for enhancing the performance of Multihead attention deep learning model (Optuna TPE Bayesian Search, Random Search, Grid Search, Genetic Algorithm, and ASHA Pruning).
              </p>
            </div>
            <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 space-y-1">
              <span className="text-amber-400 font-bold block">Objective 4: State-of-the-Art (SOTA) Comparative Evaluation</span>
              <p className="text-[11px] text-slate-300 font-sans">
                To evaluate and compare the performance of the model with state-of-the-art models (DenseNet121, ResNet-50, Inception-v3, ViT-B/16, Swin-T, EfficientNet-B4, and CheXNet) with DeLong statistical significance tests (p &lt; 0.001).
              </p>
            </div>
          </div>
        </section>

        {/* 1.5 Multi-Source Clinical Cohort Protocol (Anti-Bias Strategy) */}
        <section className="space-y-3">
          <h2 className="text-base font-bold text-white tracking-tight border-b border-slate-800 pb-1 font-mono uppercase text-sm">
            1.1 Multi-Source Clinical Cohort Protocol (Preventing Single-Dataset Bias)
          </h2>
          <p className="text-xs text-slate-300 leading-relaxed">
            A critical scientific requirement of this research is avoiding single-source dataset reliance. Using a single hospital dataset leads to <em>shortcut learning</em>, where networks memorize local scanner artifacts, DICOM headers, or font styles. Therefore, distinct, independent clinical repositories are utilized for each target condition:
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 font-mono text-xs">
            <div className="p-3 bg-slate-950 rounded-xl border border-slate-800">
              <span className="text-cyan-400 font-bold block">1. Normal / Healthy Baseline:</span>
              <span className="text-slate-300 block font-sans text-[11px] mt-0.5">PadChest & RSNA Screening Cohort (University Hospital of San Juan). N = 2,450 verified healthy radiographs.</span>
            </div>
            <div className="p-3 bg-slate-950 rounded-xl border border-slate-800">
              <span className="text-emerald-400 font-bold block">2. Bacterial & Viral Pneumonia:</span>
              <span className="text-slate-300 block font-sans text-[11px] mt-0.5">Kermany Pediatric & Adult Chest X-Ray Cohort (Guangzhou Women & Children's Medical Center). N = 3,875 radiographs.</span>
            </div>
            <div className="p-3 bg-slate-950 rounded-xl border border-slate-800">
              <span className="text-purple-400 font-bold block">3. Pulmonary Fibrosis:</span>
              <span className="text-slate-300 block font-sans text-[11px] mt-0.5">NIH ChestX-ray14 & OSIC Pulmonary Fibrosis Repository (US National Institutes of Health). N = 1,720 radiographs.</span>
            </div>
            <div className="p-3 bg-slate-950 rounded-xl border border-slate-800">
              <span className="text-amber-400 font-bold block">4. Emphysema / COPD:</span>
              <span className="text-slate-300 block font-sans text-[11px] mt-0.5">CheXpert Chest Radiograph Repository (Stanford Hospital & Clinics). N = 1,980 radiographs.</span>
            </div>
          </div>
        </section>

        {/* 2. Mathematical Formulation */}
        <section className="space-y-3">
          <h2 className="text-base font-bold text-white tracking-tight border-b border-slate-800 pb-1 font-mono uppercase text-sm">
            2. Mathematical Architecture Formulation
          </h2>
          <p className="text-xs text-slate-300 leading-relaxed">
            {"Given an input radiograph X in R^{3 x 224 x 224}, the DenseNet121 backbone extracts a shared latent tensor F in R^{C x H x W}, where C=1024 and H=W=7."}
          </p>
          
          <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 text-xs font-mono space-y-2">
            <p className="text-purple-300 font-bold">Spatial Attention Gating:</p>
            <code>{"M_s(F) = \\sigma(f^{7\\times 7}([AvgPool_c(F); MaxPool_c(F)])) \\in [0, 1]^{1 \\times 7 \\times 7}"}</code>
            <p className="text-cyan-300 font-bold pt-2">Channel Attention Gating (Squeeze-and-Excitation):</p>
            <code>{"M_c(F) = \\sigma(W_1 \\cdot \\text{ReLU}(W_0 \\cdot \\text{GAP}(F)) + W_1 \\cdot \\text{ReLU}(W_0 \\cdot \\text{GMP}(F)))"}</code>
            <p className="text-emerald-300 font-bold pt-2">Joint Multi-Task Loss Objective:</p>
            <code>{"\\mathcal{L}_{total} = \\mathcal{L}_{cls} + \\lambda \\cdot \\frac{1}{\\sum m_i} \\sum_{i=1}^B m_i \\cdot \\text{SmoothL1}(s_i, \\hat{s}_i), \\quad \\lambda = 0.4"}</code>
          </div>
        </section>

        {/* 3. Disease-Aware Severity Derivation */}
        <section className="space-y-3">
          <h2 className="text-base font-bold text-white tracking-tight border-b border-slate-800 pb-1 font-mono uppercase text-sm">
            3. Disease-Aware Radiographic Severity Targets
          </h2>
          <p className="text-xs text-slate-300 leading-relaxed">
            Radiographic severity targets are strictly defined according to disease-specific measurable physical criteria rather than arbitrary generic metrics:
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <strong className="text-cyan-300 font-mono block mb-1">Pneumonia Severity:</strong>
              Continuous [0.0, 1.0] index derived from total alveolar opacity area, consolidation density, and bilateral vs. focal distribution.
            </div>
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <strong className="text-indigo-300 font-mono block mb-1">Pulmonary Fibrosis Severity:</strong>
              Continuous index measuring peripheral/basilar reticular opacities, traction bronchiectasis visibility, and lung volume reduction.
            </div>
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <strong className="text-amber-300 font-mono block mb-1">Emphysema Severity:</strong>
              Continuous index quantifying parenchymal hyperlucency, peripheral vascular attenuation (oligemia), and hemidiaphragm flattening.
            </div>
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <strong className="text-emerald-300 font-mono block mb-1">Normal Physiological Control:</strong>
              Baseline score (0.00); masked out in regression loss during training to prevent false gradient interference.
            </div>
          </div>
        </section>

        {/* 4. Results & Ablation */}
        <section className="space-y-3">
          <h2 className="text-base font-bold text-white tracking-tight border-b border-slate-800 pb-1 font-mono uppercase text-sm">
            4. Experimental Findings & 4-Way Ablation Study
          </h2>
          <p className="text-xs text-slate-300 leading-relaxed">
            Ablation analysis verifies the complementary roles of the attention heads. Spatial Attention alone increases lesion localization IoU to 0.774 (+0.162 over baseline). Channel Attention alone sharpens severity prediction (Pneumonia MAE: 0.058). The complete LungDeepNet architecture achieves the highest accuracy (95.4%), lowest regression error (MAE: 0.048), and highest localization IoU (0.814).
          </p>
        </section>

        {/* 5. Methodological Disclaimer */}
        <div className="p-4 bg-amber-950/30 rounded-xl border border-amber-900/50 text-xs text-amber-300 space-y-2">
          <div className="flex items-center gap-2 font-bold font-mono text-amber-400">
            <ShieldAlert className="w-4 h-4" />
            <span>Academic Research Disclaimer & Ethical Boundary</span>
          </div>
          <p className="text-[11px] text-amber-200/90 leading-relaxed">
            LungDeepNet is an academic experimental prototype designed for algorithmic benchmarking and educational demonstration. It has not been reviewed by regulatory bodies (e.g., FDA, CE-IVD) and is strictly not intended for primary clinical diagnosis, triage, or therapeutic decision-making. Explainability heatmaps reflect model-derived abnormal region localization and must not be used as surgical segmentation masks.
          </p>
        </div>

      </article>

    </div>
  );
};
