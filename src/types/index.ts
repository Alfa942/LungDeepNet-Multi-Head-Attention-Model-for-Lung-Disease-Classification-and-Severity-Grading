export type DiseaseClass = 'Normal' | 'Pneumonia' | 'Emphysema' | 'Pulmonary Fibrosis';

export interface DiseaseFeatureBreakdown {
  label: string;
  value: number; // 0.0 to 1.0
  description: string;
  clinicalSignificance: string;
}

export interface PatientCase {
  id: string;
  patientId: string;
  groundTruth: DiseaseClass;
  predictedClass: DiseaseClass;
  confidence: number;
  posteriorProbabilities: Record<DiseaseClass, number>;
  severityScore: number | null; // 0.0 to 1.0, null for normal
  severityDescriptor?: string;
  diseaseFeatures: DiseaseFeatureBreakdown[];
  affectedLungZones: string[];
  findingsDescription: string;
  dicomMetadata: {
    modality: string;
    viewPosition: string; // 'PA' or 'AP'
    patientAge: number;
    patientSex: 'M' | 'F';
    kVp: number;
    exposureTimeMs: number;
    pixelSpacing: string;
    matrixSize: string;
    photometricInterpretation: string;
  };
  imageSrc: string;
  // Synthetic / pre-calculated attention maps for live visualization
  spatialAttentionGrid: number[][]; // 7x7 or 14x14 normalized values
  gradCamGrid: number[][];
  gradCamPlusPlusGrid: number[][];
  layerCamGrid: number[][];
  scoreCamGrid?: number[][];
  isLowConfidence?: boolean;
  hasGroundTruthBBox?: boolean;
  groundTruthBBox?: { x: number; y: number; width: number; height: number; label: string };
  iouScore?: number;
  diceScore?: number;
}

export interface DatasetSummary {
  disease: DiseaseClass;
  sourceCohort?: string;
  institution?: string;
  clinicalRationale?: string;
  path: string;
  totalImages: number;
  uniquePatients: number;
  trainCount: number;
  valCount: number;
  testCount: number;
  severityAvailable: boolean;
  severityFormat: string;
  boundingBoxesCount: number;
  formatTypes: string[];
}

export interface OptunaTrial {
  trialNumber: number;
  learningRate: number;
  batchSize: number;
  dropout: number;
  weightDecay: number;
  denseNetFineTuneDepth: string; // 'none', 'denseblock4', 'denseblock3+4', 'full'
  attentionReductionRatio: number;
  clsLossWeight: number;
  sevLossWeight: number;
  valLoss: number;
  valF1: number;
  valSeverityMae: number;
  status: 'COMPLETED' | 'PRUNED';
}

export interface AblationConfig {
  name: string;
  key: 'baseline' | 'spatial_only' | 'channel_only' | 'dual_head';
  description: string;
  accuracy: number;
  macroF1: number;
  auroc: number;
  pneumoniaMae: number;
  fibrosisMae: number;
  emphysemaMae: number;
  localizationIoU: number;
  inferenceTimeMs: number;
  parametersM: number;
}

export interface JupyterNotebook {
  id: string;
  fileName: string;
  title: string;
  objective: string;
  summary: string;
  cells: {
    cell_type: 'markdown' | 'code';
    source: string[];
    execution_count?: number | null;
    outputs?: Array<{
      output_type: string;
      text?: string[];
      data?: Record<string, any>;
    }>;
  }[];
}

export interface PreprocessingExperiment {
  id: string;
  name: string;
  category: 'Noise Removal' | 'Contrast Enhancement' | 'Edge Preservation' | 'Hybrid Pipeline';
  description: string;
  psnr: number;
  ssim: number;
  cnr: number;
  epi: number;
  downstreamAcc: number;
  downstreamSeverityRmse: number;
  latencyMs: number;
  formula: string;
  bestFor: string;
}

export interface HpoTechniqueComparison {
  technique: string;
  bestValAcc: number;
  bestValF1: number;
  trialsToOptimal: number;
  totalGpuHours: number;
  parameterEfficiency: string;
  earlyPruningSupport: boolean;
  convergenceRate: number[];
  bestHyperparams: {
    lr: string;
    weightDecay: string;
    dropout: number;
    heads: number;
    batchSize: number;
    sevLossWeight: number;
  };
}

export interface SotaModelComparison {
  modelName: string;
  reference: string;
  family: 'CNN' | 'Vision Transformer' | 'Hybrid Attention';
  parametersM: number;
  gflops: number;
  latencyMs: number;
  accuracy: number;
  precision: number;
  sensitivity: number;
  specificity: number;
  f1Score: number;
  auroc: number;
  auprc: number;
  severityMae: number;
  severityRmse: number;
  delongPValueVsProposed: string;
  isProposed: boolean;
}

export interface PythonSourceFile {
  name: string;
  path: string;
  description: string;
  code: string;
}
