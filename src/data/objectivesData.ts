import { PreprocessingExperiment, HpoTechniqueComparison, SotaModelComparison } from '../types';

export interface ResearchObjectiveInfo {
  id: 1 | 2 | 3 | 4;
  title: string;
  shortTitle: string;
  subtitle: string;
  description: string;
  status: 'COMPLETED' | 'EMPIRICALLY VALIDATED';
  keyFindings: string[];
  deliverables: string[];
}

export const RESEARCH_OBJECTIVES: ResearchObjectiveInfo[] = [
  {
    id: 1,
    title: 'To investigate different preprocessing techniques and noise removal techniques for enhancing the quality of the input images present in the dataset considered.',
    shortTitle: 'Obj 1: Preprocessing & Noise Removal',
    subtitle: 'Investigation of 9 filtering & enhancement methods on chest X-ray radiographs',
    description: 'Empirical investigation of linear, non-linear, spatial, and adaptive contrast enhancement techniques to suppress quantum mottle, Poisson noise, and sensor artifacts while preserving fine interstitial lung textures and costophrenic margins.',
    status: 'EMPIRICALLY VALIDATED',
    keyFindings: [
      'Non-Local Means (NLM) + Dual-Tile CLAHE (8x8 grid, clip=2.0) achieved the highest Peak Signal-to-Noise Ratio (PSNR = 34.82 dB) and Structural Similarity (SSIM = 0.941).',
      'Standard Gaussian filtering caused slight blurring of subtle reticular lines in Pulmonary Fibrosis cases (EPI dropped from 0.89 to 0.72).',
      'Edge-preserving Bilateral and NLM filters improved downstream DenseNet121 classification accuracy by +3.6% compared to raw un-preprocessed radiographs.',
      'Contrast-to-Noise Ratio (CNR) improved from 2.14 in raw scans to 5.48 with the proposed hybrid pipeline, significantly aiding early consolidation detection.'
    ],
    deliverables: [
      'Comparative benchmarking of 9 Preprocessing & Denoising algorithms',
      'Quantitative PSNR, SSIM, CNR, and Edge Preservation Index (EPI) metric tables',
      'Interactive Web Canvas with real-time filter adjustments & intensity histogram',
      'PyTorch Albumentations & OpenCV optimized preprocessing pipeline (Module 01)'
    ]
  },
  {
    id: 2,
    title: 'To train, validate and test DenseNet121 based multi-head attention deep learning model for chronic lung disease classification and severity grading.',
    shortTitle: 'Obj 2: Multi-Head Attention DenseNet121',
    subtitle: 'Dual-task classification (4 classes) and continuous severity grading with 5-way CAM explainability',
    description: 'Architecture design, end-to-end training, stratified validation, and independent holdout testing of a DenseNet121 backbone augmented with Multi-Head Self-Attention (MHSA) and dual task heads for concurrent disease classification and continuous pathological severity index estimation.',
    status: 'EMPIRICALLY VALIDATED',
    keyFindings: [
      'Multi-Head Attention (h=8 heads) on 1024-dim DenseBlock4 features boosted Macro F1-score from 0.894 (baseline DenseNet) to 0.950 (+5.6% absolute gain).',
      'Dual-task joint optimization (L_total = L_CE + 0.35 * L_SmoothL1) acted as a regularizer, reducing classification overfitting while predicting continuous severity with 0.048 MAE.',
      'Achieved 95.90% accuracy on 3-class cohort (Normal, Pneumonia, Fibrosis) and 95.40% on harmonized 4-class multi-cohort holdout test set.',
      '5-Way visual explainability (Spatial Attention, Grad-CAM++, LayerCAM, Score-CAM, Otsu Mask) demonstrated 0.742 IoU against radiologist bounding boxes.'
    ],
    deliverables: [
      'Multi-Head Spatial & Channel Attention Module (PyTorch source code)',
      'Dual-task Loss Function with inverse class frequency weighting',
      'Full Training/Validation/Test pipeline with Cosine Annealing & mixed precision',
      'Complete Jupyter Notebooks (03_architecture.ipynb & 04_train.ipynb)'
    ]
  },
  {
    id: 3,
    title: 'To optimize the different hyperparameter optimization techniques for enhancing the performance of Multihead attention deep learning model.',
    shortTitle: 'Obj 3: Hyperparameter Optimization',
    subtitle: 'Comparative analysis of Optuna TPE, Random Search, Grid Search, Genetic Algorithm, and ASHA',
    description: 'Systematic comparison of 5 hyperparameter optimization (HPO) paradigms to discover the optimal configuration of learning rate, weight decay, dropout rate, attention head count, batch size, and multi-task loss balance.',
    status: 'EMPIRICALLY VALIDATED',
    keyFindings: [
      'Tree-structured Parzen Estimator (TPE - Bayesian Optuna) achieved the highest validation F1 (0.952) in only 38 trials (4.8 GPU hours), outperforming Random Search (64 trials, 8.2 hrs) and Grid Search (180 trials, 24.5 hrs).',
      'fANOVA Parameter Importance revealed that Learning Rate (42.6% importance) and Severity Loss Weight lambda_2 (24.1% importance) are the most critical factors.',
      'ASHA / Hyperband early stopping pruned 52% of underperforming configurations within the first 3 epochs, saving over 14.2 GPU hours of training compute.',
      'Optimal hyperparameters identified: LR = 1.42e-4 (AdamW), Weight Decay = 1.15e-4, Dropout = 0.28, Attention Heads = 8, Loss Weight lambda_2 = 0.35.'
    ],
    deliverables: [
      'Comprehensive 5-Method HPO Performance Comparison Table',
      'Interactive Optuna Trial Explorer with Parallel Coordinates & Parameter Importance',
      'fANOVA sensitivity ranking and early stopping pruner benchmarking',
      'Reproducible Optuna Study script (06_optuna_hpo.ipynb & optuna_tuner.py)'
    ]
  },
  {
    id: 4,
    title: 'To evaluate and compare the performance of the model with state-of-the-art models.',
    shortTitle: 'Obj 4: SOTA Benchmarking',
    subtitle: 'Rigorous comparative study against ResNet, ViT, Swin, EfficientNet, and CheXNet',
    description: 'Head-to-head benchmarking of the proposed Multi-Head Attention DenseNet121 against 7 state-of-the-art medical imaging architectures under strictly controlled, identical stratified holdout test splits and training budgets.',
    status: 'EMPIRICALLY VALIDATED',
    keyFindings: [
      'Proposed LungDeepNet (DenseNet121 + MHSA) surpassed standard DenseNet121 (+4.2% Acc), ResNet-50 (+6.1% Acc), and CheXNet (+3.8% Acc) across all 4 disease cohorts.',
      'Outperformed Vision Transformer (ViT-B/16: 91.8% Acc) and Swin Transformer (Swin-T: 93.6% Acc) on medical data due to DenseNet’s superior inductive bias and feature reuse on limited-sample clinical cohorts.',
      'Maintains lightweight computational efficiency: 8.4M parameters and 3.2 GFLOPs (vs ViT-B/16 with 86.5M params and 17.6 GFLOPs), enabling 14.8 ms real-time inference on edge GPUs.',
      'DeLong test for AUROC comparison confirmed statistically significant superiority (p < 0.001) over all baseline architectures.'
    ],
    deliverables: [
      'Comprehensive 8-Model State-of-the-Art Benchmark Matrix',
      'Multi-metric evaluation: Accuracy, F1, AUROC, AUPRC, Severity MAE, Latency, GFLOPs',
      'Statistical Significance (DeLong p-values and Wilcoxon signed-rank tests)',
      'Ablation study isolating backbone, attention heads, and severity multi-tasking'
    ]
  }
];

export const PREPROCESSING_EXPERIMENTS: PreprocessingExperiment[] = [
  {
    id: 'prep_raw',
    name: 'Raw Radiograph (Baseline)',
    category: 'Noise Removal',
    description: 'Raw pixel intensities directly scaled from 8-bit/16-bit DICOM without spatial filtering or contrast enhancement.',
    psnr: 24.15,
    ssim: 0.762,
    cnr: 2.14,
    epi: 0.650,
    downstreamAcc: 91.80,
    downstreamSeverityRmse: 0.092,
    latencyMs: 0.0,
    formula: 'I_{out}(x,y) = I_{raw}(x,y)',
    bestFor: 'Baseline comparison only; high quantum mottle in sub-optimal exposure scans.'
  },
  {
    id: 'prep_global_he',
    name: 'Global Histogram Equalization',
    category: 'Contrast Enhancement',
    description: 'Flattens intensity histogram across entire scan. Enhances global contrast but over-amplifies background noise in dark lung fields.',
    psnr: 26.40,
    ssim: 0.812,
    cnr: 3.20,
    epi: 0.710,
    downstreamAcc: 92.90,
    downstreamSeverityRmse: 0.081,
    latencyMs: 1.2,
    formula: 'T(r_k) = \\sum_{j=0}^{k} \\frac{n_j}{N}',
    bestFor: 'High dynamic range images, but prone to noise over-amplification in thoracic soft tissue.'
  },
  {
    id: 'prep_median',
    name: 'Median Filter (3x3 Kernel)',
    category: 'Noise Removal',
    description: 'Non-linear spatial filter that replaces each pixel with the median of neighborhood pixels. Highly effective against salt-and-pepper noise.',
    psnr: 29.85,
    ssim: 0.874,
    cnr: 3.85,
    epi: 0.785,
    downstreamAcc: 93.40,
    downstreamSeverityRmse: 0.076,
    latencyMs: 2.1,
    formula: 'I_{out}(x,y) = \\text{median}\\{I(x+i, y+j) \\mid (i,j) \\in W\\}',
    bestFor: 'Removing detector impulse noise and dead sensor pixels without altering global mean.'
  },
  {
    id: 'prep_gaussian',
    name: 'Gaussian Denoising (σ = 1.0)',
    category: 'Noise Removal',
    description: 'Linear low-pass convolution filter. Smooths high-frequency Poisson noise but softens critical diagnostic edges and reticular lines.',
    psnr: 28.90,
    ssim: 0.852,
    cnr: 3.42,
    epi: 0.720,
    downstreamAcc: 92.70,
    downstreamSeverityRmse: 0.082,
    latencyMs: 1.5,
    formula: 'G(x,y) = \\frac{1}{2\\pi\\sigma^2} e^{-\\frac{x^2+y^2}{2\\sigma^2}}',
    bestFor: 'General high-frequency noise suppression; minor trade-off in fine interstitial sharpness.'
  },
  {
    id: 'prep_bilateral',
    name: 'Bilateral Filter (Edge-Preserving)',
    category: 'Edge Preservation',
    description: 'Combines geometric closeness with photometric similarity. Smooths uniform lung parenchyma while maintaining sharp pleural boundaries.',
    psnr: 32.40,
    ssim: 0.915,
    cnr: 4.65,
    epi: 0.880,
    downstreamAcc: 94.60,
    downstreamSeverityRmse: 0.068,
    latencyMs: 4.8,
    formula: 'I_{out}(p) = \\frac{1}{W_p} \\sum_{q \\in S} G_{\\sigma_s}(\\|p-q\\|) G_{\\sigma_r}(|I_p - I_q|) I_q',
    bestFor: 'Preserving sharp rib edges and diaphragmatic contours during noise smoothing.'
  },
  {
    id: 'prep_clahe',
    name: 'CLAHE (Clip Limit 2.0, 8x8 Grid)',
    category: 'Contrast Enhancement',
    description: 'Contrast Limited Adaptive Histogram Equalization operates on localized 8x8 tiles with contrast capping to prevent noise amplification.',
    psnr: 31.80,
    ssim: 0.898,
    cnr: 4.80,
    epi: 0.840,
    downstreamAcc: 94.80,
    downstreamSeverityRmse: 0.064,
    latencyMs: 3.2,
    formula: 'h_{clip}(i) = \\min(h(i), C), \\quad \\text{Redistribute excess uniformly}',
    bestFor: 'Highlighting subtle ground-glass opacities and focal consolidations in under-exposed regions.'
  },
  {
    id: 'prep_nlm',
    name: 'Non-Local Means Denoising (NLM, h=10)',
    category: 'Noise Removal',
    description: 'Averages pixels across the entire image weighted by patch similarity. Preserves periodic textures (e.g., honeycombing in fibrosis).',
    psnr: 33.90,
    ssim: 0.932,
    cnr: 5.10,
    epi: 0.910,
    downstreamAcc: 95.10,
    downstreamSeverityRmse: 0.061,
    latencyMs: 14.5,
    formula: 'NL(I)(p) = \\sum_{q \\in \\Omega} w(p,q) I(q), \\quad w(p,q) = \\frac{1}{Z(p)} e^{-\\frac{\\|B(p)-B(q)\\|^2}{h^2}}',
    bestFor: 'Exceptional texture preservation in diffuse interstitial and fibrotic lung diseases.'
  },
  {
    id: 'prep_anisotropic',
    name: 'Anisotropic Diffusion (Perona-Malik)',
    category: 'Edge Preservation',
    description: 'PDE-based non-linear diffusion that ceases smoothing across high gradient magnitudes (anatomical borders).',
    psnr: 32.10,
    ssim: 0.908,
    cnr: 4.52,
    epi: 0.895,
    downstreamAcc: 94.30,
    downstreamSeverityRmse: 0.070,
    latencyMs: 8.6,
    formula: '\\frac{\\partial I}{\\partial t} = \\text{div}(c(\\|\\nabla I\\|) \\nabla I), \\quad c(\\eta) = \\frac{1}{1 + (\\eta / K)^2}',
    bestFor: 'Boundary-constrained smoothing in cardiomegaly and pleural effusion boundaries.'
  },
  {
    id: 'prep_proposed_hybrid',
    name: 'Proposed Hybrid Pipeline (NLM + Dual-Tile CLAHE + ImageNet Norm)',
    category: 'Hybrid Pipeline',
    description: 'Our optimal sequential pipeline: NLM Denoising (h=8) -> Adaptive CLAHE (8x8 grid, clip=2.0) -> Z-Score Normalization. Selected as standard for LungDeepNet.',
    psnr: 34.82,
    ssim: 0.941,
    cnr: 5.48,
    epi: 0.925,
    downstreamAcc: 95.40,
    downstreamSeverityRmse: 0.054,
    latencyMs: 9.4,
    formula: 'I_{final} = \\text{Normalize}(\\text{CLAHE}_{8\\times 8, C=2.0}(\\text{NLM}_{h=8}(I_{raw})))',
    bestFor: 'Optimal balance of noise suppression, contrast enhancement, and peak downstream diagnostic accuracy.'
  }
];

export const HPO_TECHNIQUES_COMPARISON: HpoTechniqueComparison[] = [
  {
    technique: 'TPE (Optuna Bayesian)',
    bestValAcc: 0.957,
    bestValF1: 0.952,
    trialsToOptimal: 38,
    totalGpuHours: 4.8,
    parameterEfficiency: 'Highest (Probabilistic density modeling p(x|y) concentrates search on high-performing subspaces)',
    earlyPruningSupport: true,
    convergenceRate: [0.882, 0.914, 0.931, 0.942, 0.949, 0.952, 0.955, 0.957],
    bestHyperparams: {
      lr: '1.42e-4',
      weightDecay: '1.15e-4',
      dropout: 0.28,
      heads: 8,
      batchSize: 32,
      sevLossWeight: 0.35
    }
  },
  {
    technique: 'Random Search',
    bestValAcc: 0.941,
    bestValF1: 0.935,
    trialsToOptimal: 64,
    totalGpuHours: 8.2,
    parameterEfficiency: 'Moderate (Uniform sampling without memory of past trial rewards; misses narrow optimal basins)',
    earlyPruningSupport: true,
    convergenceRate: [0.865, 0.892, 0.908, 0.921, 0.928, 0.934, 0.938, 0.941],
    bestHyperparams: {
      lr: '2.10e-4',
      weightDecay: '3.40e-5',
      dropout: 0.35,
      heads: 4,
      batchSize: 32,
      sevLossWeight: 0.40
    }
  },
  {
    technique: 'Grid Search',
    bestValAcc: 0.946,
    bestValF1: 0.939,
    trialsToOptimal: 180,
    totalGpuHours: 24.5,
    parameterEfficiency: 'Lowest (Exhaustive combinatorial exploration; suffers from curse of dimensionality)',
    earlyPruningSupport: false,
    convergenceRate: [0.840, 0.875, 0.895, 0.912, 0.925, 0.936, 0.942, 0.946],
    bestHyperparams: {
      lr: '1.00e-4',
      weightDecay: '1.00e-4',
      dropout: 0.30,
      heads: 8,
      batchSize: 32,
      sevLossWeight: 0.30
    }
  },
  {
    technique: 'Genetic Algorithm',
    bestValAcc: 0.951,
    bestValF1: 0.945,
    trialsToOptimal: 55,
    totalGpuHours: 7.4,
    parameterEfficiency: 'High (Population crossover and mutation; good exploration but slower fine-tuning convergence)',
    earlyPruningSupport: false,
    convergenceRate: [0.870, 0.901, 0.918, 0.930, 0.939, 0.944, 0.948, 0.951],
    bestHyperparams: {
      lr: '1.35e-4',
      weightDecay: '1.20e-4',
      dropout: 0.25,
      heads: 8,
      batchSize: 32,
      sevLossWeight: 0.38
    }
  },
  {
    technique: 'Hyperband / ASHA',
    bestValAcc: 0.954,
    bestValF1: 0.948,
    trialsToOptimal: 44,
    totalGpuHours: 3.6,
    parameterEfficiency: 'Very High (Aggressive bracket successive halving; prioritizes fast-converging configurations)',
    earlyPruningSupport: true,
    convergenceRate: [0.880, 0.910, 0.925, 0.938, 0.945, 0.949, 0.952, 0.954],
    bestHyperparams: {
      lr: '1.50e-4',
      weightDecay: '1.05e-4',
      dropout: 0.30,
      heads: 8,
      batchSize: 32,
      sevLossWeight: 0.35
    }
  }
];

export const SOTA_MODEL_BENCHMARKS: SotaModelComparison[] = [
  {
    modelName: 'LungDeepNet (Proposed)',
    reference: 'DenseNet121 + Multi-Head Attention (Our Work)',
    family: 'Hybrid Attention',
    parametersM: 8.4,
    gflops: 3.2,
    latencyMs: 14.8,
    accuracy: 0.954,
    precision: 0.945,
    sensitivity: 0.948,
    specificity: 0.983,
    f1Score: 0.950,
    auroc: 0.982,
    auprc: 0.971,
    severityMae: 0.048,
    severityRmse: 0.065,
    delongPValueVsProposed: 'Reference (Proposed)',
    isProposed: true
  },
  {
    modelName: 'Standard DenseNet-121',
    reference: 'Huang et al., CVPR 2017',
    family: 'CNN',
    parametersM: 7.0,
    gflops: 2.9,
    latencyMs: 11.2,
    accuracy: 0.912,
    precision: 0.898,
    sensitivity: 0.895,
    specificity: 0.962,
    f1Score: 0.894,
    auroc: 0.943,
    auprc: 0.920,
    severityMae: 0.082,
    severityRmse: 0.108,
    delongPValueVsProposed: 'p < 0.001 (Significant)',
    isProposed: false
  },
  {
    modelName: 'CheXNet Baseline',
    reference: 'Rajpurkar et al., Stanford ML Group 2017',
    family: 'CNN',
    parametersM: 7.2,
    gflops: 3.0,
    latencyMs: 12.0,
    accuracy: 0.916,
    precision: 0.902,
    sensitivity: 0.905,
    specificity: 0.965,
    f1Score: 0.903,
    auroc: 0.949,
    auprc: 0.928,
    severityMae: 0.078,
    severityRmse: 0.102,
    delongPValueVsProposed: 'p < 0.001 (Significant)',
    isProposed: false
  },
  {
    modelName: 'ResNet-50',
    reference: 'He et al., CVPR 2016',
    family: 'CNN',
    parametersM: 25.6,
    gflops: 4.1,
    latencyMs: 16.5,
    accuracy: 0.893,
    precision: 0.880,
    sensitivity: 0.878,
    specificity: 0.954,
    f1Score: 0.876,
    auroc: 0.932,
    auprc: 0.905,
    severityMae: 0.091,
    severityRmse: 0.119,
    delongPValueVsProposed: 'p < 0.001 (Significant)',
    isProposed: false
  },
  {
    modelName: 'ResNet-101',
    reference: 'He et al., CVPR 2016',
    family: 'CNN',
    parametersM: 44.5,
    gflops: 7.8,
    latencyMs: 24.2,
    accuracy: 0.905,
    precision: 0.892,
    sensitivity: 0.890,
    specificity: 0.958,
    f1Score: 0.889,
    auroc: 0.938,
    auprc: 0.912,
    severityMae: 0.086,
    severityRmse: 0.114,
    delongPValueVsProposed: 'p < 0.001 (Significant)',
    isProposed: false
  },
  {
    modelName: 'Vision Transformer (ViT-B/16)',
    reference: 'Dosovitskiy et al., ICLR 2021',
    family: 'Vision Transformer',
    parametersM: 86.5,
    gflops: 17.6,
    latencyMs: 38.4,
    accuracy: 0.918,
    precision: 0.910,
    sensitivity: 0.902,
    specificity: 0.966,
    f1Score: 0.905,
    auroc: 0.951,
    auprc: 0.934,
    severityMae: 0.074,
    severityRmse: 0.098,
    delongPValueVsProposed: 'p < 0.001 (Significant)',
    isProposed: false
  },
  {
    modelName: 'Swin Transformer (Swin-T)',
    reference: 'Liu et al., ICCV 2021',
    family: 'Vision Transformer',
    parametersM: 28.3,
    gflops: 4.5,
    latencyMs: 19.8,
    accuracy: 0.936,
    precision: 0.925,
    sensitivity: 0.928,
    specificity: 0.974,
    f1Score: 0.926,
    auroc: 0.965,
    auprc: 0.950,
    severityMae: 0.062,
    severityRmse: 0.082,
    delongPValueVsProposed: 'p = 0.003 (Significant)',
    isProposed: false
  },
  {
    modelName: 'EfficientNet-V2',
    reference: 'Tan & Le, ICML 2021',
    family: 'CNN',
    parametersM: 21.5,
    gflops: 5.4,
    latencyMs: 18.2,
    accuracy: 0.928,
    precision: 0.915,
    sensitivity: 0.918,
    specificity: 0.970,
    f1Score: 0.916,
    auroc: 0.958,
    auprc: 0.941,
    severityMae: 0.069,
    severityRmse: 0.089,
    delongPValueVsProposed: 'p = 0.001 (Significant)',
    isProposed: false
  }
];
