import { PatientCase, DiseaseClass } from '../types';

export type SeverityStage = 'none' | 'mild' | 'moderate' | 'severe';

// Helper to generate realistic high-resolution SVG Chest Radiographs with distinct severity stages
export function createChestXraySvg(type: 'normal' | 'pneumonia' | 'emphysema' | 'fibrosis', severity: SeverityStage = 'moderate'): string {
  let pathologyOverlay = '';

  if (type === 'pneumonia') {
    if (severity === 'mild') {
      // Mild: Focal small right lower lobe subsegmental opacity
      pathologyOverlay = `
        <!-- Mild Pneumonia Focal Opacity -->
        <circle cx="150" cy="290" r="32" fill="rgba(240, 245, 255, 0.38)" filter="url(#blur-med)" />
        <circle cx="150" cy="290" r="18" fill="rgba(255, 255, 255, 0.45)" filter="url(#blur-subtle)" />
        <path d="M 145 280 L 155 300" stroke="rgba(40,45,55,0.6)" stroke-width="1.5" stroke-linecap="round" />
      `;
    } else if (severity === 'severe') {
      // Severe: Dense multi-lobar confluent consolidation with air bronchograms
      pathologyOverlay = `
        <!-- Severe Pneumonia Dense Confluent Bilateral Opacities -->
        <path d="M 100 200 Q 150 220 185 270 Q 170 350 95 340 Q 75 260 100 200 Z" fill="rgba(255, 255, 255, 0.72)" filter="url(#blur-heavy)" />
        <path d="M 115 220 Q 155 240 175 290 Q 150 330 110 310 Z" fill="rgba(255, 255, 255, 0.85)" filter="url(#blur-med)" />
        <!-- Left lower zone dense consolidation -->
        <path d="M 290 230 Q 350 250 380 310 Q 340 360 280 340 Z" fill="rgba(240, 245, 255, 0.62)" filter="url(#blur-heavy)" />
        <circle cx="330" cy="290" r="35" fill="rgba(255, 255, 255, 0.7)" filter="url(#blur-med)" />
        <!-- Extensive air bronchograms -->
        <path d="M 130 235 L 150 275 M 150 275 L 165 315 M 150 275 L 135 305" stroke="rgba(20,25,35,0.85)" stroke-width="2.5" stroke-linecap="round" />
        <path d="M 320 255 L 340 285 M 340 285 L 350 315" stroke="rgba(20,25,35,0.85)" stroke-width="2.5" stroke-linecap="round" />
      `;
    } else {
      // Moderate (Default): Classic unilateral lobar consolidation
      pathologyOverlay = `
        <!-- Moderate Pneumonia Lobar Consolidation -->
        <path d="M 120 220 Q 150 240 180 290 Q 160 340 110 320 Q 90 270 120 220 Z" fill="rgba(240, 245, 255, 0.52)" filter="url(#blur-heavy)" />
        <path d="M 130 240 Q 160 260 170 300 Q 140 320 120 290 Z" fill="rgba(255, 255, 255, 0.68)" filter="url(#blur-med)" />
        <path d="M 310 270 Q 340 290 350 320 Q 320 340 300 320 Z" fill="rgba(240, 245, 255, 0.28)" filter="url(#blur-med)" />
        <!-- Air bronchogram lines -->
        <path d="M 140 250 L 155 285 M 155 285 L 165 310" stroke="rgba(40,45,55,0.7)" stroke-width="2" stroke-linecap="round" />
      `;
    }
  } else if (type === 'emphysema') {
    if (severity === 'mild') {
      // Mild: Apical mild hyperlucency, normal diaphragm height
      pathologyOverlay = `
        <rect x="90" y="110" width="90" height="120" fill="rgba(10, 12, 16, 0.28)" filter="url(#blur-heavy)" />
        <rect x="300" y="110" width="90" height="120" fill="rgba(10, 12, 16, 0.28)" filter="url(#blur-heavy)" />
      `;
    } else if (severity === 'severe') {
      // Severe: Profound diffuse hyperlucency, low flat diaphragms, narrow elongated heart
      pathologyOverlay = `
        <!-- Extreme Hyperlucency across all lung zones -->
        <rect x="65" y="90" width="140" height="300" fill="rgba(5, 7, 10, 0.58)" filter="url(#blur-heavy)" />
        <rect x="275" y="90" width="140" height="300" fill="rgba(5, 7, 10, 0.58)" filter="url(#blur-heavy)" />
        <!-- Very low, completely flattened diaphragmatic domes below 11th rib -->
        <path d="M 50 405 Q 140 412 215 398" stroke="rgba(230,235,245,0.8)" stroke-width="4.5" fill="none" filter="url(#blur-subtle)" />
        <path d="M 265 398 Q 340 412 430 405" stroke="rgba(230,235,245,0.8)" stroke-width="4.5" fill="none" filter="url(#blur-subtle)" />
        <!-- Prominent bullous lucencies -->
        <circle cx="140" cy="160" r="55" fill="none" stroke="rgba(56, 189, 248, 0.35)" stroke-width="2" stroke-dasharray="4,4" />
        <circle cx="340" cy="160" r="55" fill="none" stroke="rgba(56, 189, 248, 0.35)" stroke-width="2" stroke-dasharray="4,4" />
      `;
    } else {
      // Moderate: Classic hyperlucent lung fields with flattened diaphragms
      pathologyOverlay = `
        <rect x="75" y="110" width="125" height="260" fill="rgba(10, 12, 16, 0.38)" filter="url(#blur-heavy)" />
        <rect x="280" y="110" width="125" height="260" fill="rgba(10, 12, 16, 0.38)" filter="url(#blur-heavy)" />
        <path d="M 60 385 Q 140 395 210 380" stroke="rgba(230,235,245,0.7)" stroke-width="4" fill="none" filter="url(#blur-subtle)" />
        <path d="M 270 380 Q 340 395 420 385" stroke="rgba(230,235,245,0.7)" stroke-width="4" fill="none" filter="url(#blur-subtle)" />
        <circle cx="140" cy="180" r="42" fill="none" stroke="rgba(56, 189, 248, 0.25)" stroke-width="2" stroke-dasharray="4,4" />
        <circle cx="340" cy="180" r="42" fill="none" stroke="rgba(56, 189, 248, 0.25)" stroke-width="2" stroke-dasharray="4,4" />
      `;
    }
  } else if (type === 'fibrosis') {
    if (severity === 'mild') {
      // Mild: Fine peripheral subpleural reticular opacities at bases
      pathologyOverlay = `
        <g stroke="rgba(240, 245, 255, 0.42)" stroke-width="1.5" fill="none" filter="url(#blur-subtle)">
          <path d="M 90 320 L 115 335 L 95 355 L 125 365" />
          <path d="M 370 315 L 345 335 L 365 355 L 335 365" />
        </g>
        <ellipse cx="125" cy="350" rx="30" ry="20" fill="rgba(235, 240, 255, 0.2)" filter="url(#blur-med)" />
        <ellipse cx="355" cy="350" rx="30" ry="20" fill="rgba(235, 240, 255, 0.2)" filter="url(#blur-med)" />
      `;
    } else if (severity === 'severe') {
      // Severe: Coarse honeycombing, severe traction bronchiectasis, marked volume reduction
      pathologyOverlay = `
        <!-- Coarse Honeycombing Lattice & Architecture Distortion -->
        <g stroke="rgba(255, 255, 255, 0.75)" stroke-width="2.5" fill="none" filter="url(#blur-subtle)">
          <path d="M 70 260 L 110 280 L 85 315 L 135 330 L 105 365 L 165 355 M 85 285 L 130 290 L 145 335 M 65 320 L 95 360" />
          <path d="M 115 270 L 155 305 L 130 345 M 95 305 L 145 320" stroke-width="3" />
          <path d="M 400 250 L 355 280 L 385 315 L 335 330 L 365 365 L 315 355 M 385 285 L 335 295 L 325 335" />
          <path d="M 355 270 L 315 305 L 345 345 M 375 305 L 325 320" stroke-width="3" />
        </g>
        <!-- Diffuse ground glass and basal contraction -->
        <ellipse cx="125" cy="335" rx="55" ry="42" fill="rgba(255, 255, 255, 0.45)" filter="url(#blur-heavy)" />
        <ellipse cx="350" cy="330" rx="55" ry="42" fill="rgba(255, 255, 255, 0.45)" filter="url(#blur-heavy)" />
      `;
    } else {
      // Moderate: Established bilateral basilar reticular network
      pathologyOverlay = `
        <g stroke="rgba(240, 245, 255, 0.55)" stroke-width="2" fill="none" filter="url(#blur-subtle)">
          <path d="M 80 290 L 110 310 L 90 340 L 130 350 L 110 380 L 160 370 M 90 310 L 130 315 L 140 350 M 70 340 L 100 370" />
          <path d="M 120 300 L 150 330 L 130 360 M 100 330 L 140 340" stroke-width="2.5" />
          <path d="M 390 280 L 360 310 L 380 340 L 340 350 L 360 380 L 320 370 M 380 300 L 340 315 L 330 350" />
          <path d="M 350 300 L 320 330 L 340 360 M 370 330 L 330 340" stroke-width="2.5" />
        </g>
        <ellipse cx="125" cy="340" rx="45" ry="35" fill="rgba(235, 240, 255, 0.35)" filter="url(#blur-med)" />
        <ellipse cx="350" cy="335" rx="45" ry="35" fill="rgba(235, 240, 255, 0.35)" filter="url(#blur-med)" />
      `;
    }
  }

  const svgString = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 480" width="100%" height="100%" class="w-full h-full bg-black rounded-lg">
      <defs>
        <radialGradient id="lung-grad" cx="50%" cy="40%" r="60%">
          <stop offset="0%" stop-color="#12161f" />
          <stop offset="70%" stop-color="#0a0c10" />
          <stop offset="100%" stop-color="#020305" />
        </radialGradient>
        <filter id="blur-heavy" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="16" />
        </filter>
        <filter id="blur-med" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="8" />
        </filter>
        <filter id="blur-subtle" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="2.5" />
        </filter>
      </defs>

      <!-- Background Black Radiograph Base -->
      <rect width="480" height="480" fill="#030712" />

      <!-- Thoracic Cavity Silhouette -->
      <path d="M 120 40 Q 240 30 360 40 Q 430 100 440 260 Q 445 420 380 470 L 100 470 Q 35 420 40 260 Q 50 100 120 40 Z" fill="#111827" filter="url(#blur-heavy)" opacity="0.9" />

      <!-- Right Lung Field -->
      <path d="M 180 80 Q 210 130 200 240 Q 190 320 180 370 Q 140 360 70 330 Q 60 220 90 120 Q 130 65 180 80 Z" fill="url(#lung-grad)" stroke="#334155" stroke-width="1.5" filter="url(#blur-subtle)" />

      <!-- Left Lung Field -->
      <path d="M 300 80 Q 270 130 280 220 Q 260 290 280 370 Q 340 360 410 330 Q 420 220 390 120 Q 350 65 300 80 Z" fill="url(#lung-grad)" stroke="#334155" stroke-width="1.5" filter="url(#blur-subtle)" />

      <!-- Clavicles -->
      <path d="M 110 95 Q 170 115 230 110" stroke="#94a3b8" stroke-width="5" stroke-linecap="round" fill="none" opacity="0.65" filter="url(#blur-subtle)" />
      <path d="M 370 95 Q 310 115 250 110" stroke="#94a3b8" stroke-width="5" stroke-linecap="round" fill="none" opacity="0.65" filter="url(#blur-subtle)" />

      <!-- Rib Arcs -->
      <g stroke="#64748b" stroke-width="4" stroke-linecap="round" fill="none" opacity="0.4" filter="url(#blur-subtle)">
        <path d="M 190 120 Q 130 140 85 160" />
        <path d="M 195 155 Q 135 180 80 205" />
        <path d="M 200 195 Q 135 220 75 250" />
        <path d="M 200 240 Q 135 265 75 300" />
        <path d="M 195 285 Q 135 310 80 345" />
        
        <path d="M 290 120 Q 350 140 395 160" />
        <path d="M 285 155 Q 345 180 400 205" />
        <path d="M 280 195 Q 345 220 405 250" />
        <path d="M 280 240 Q 345 265 405 300" />
        <path d="M 285 285 Q 345 310 400 345" />
      </g>

      <!-- Spine & Carina -->
      <rect x="233" y="40" width="14" height="380" fill="#cbd5e1" opacity="0.45" filter="url(#blur-med)" />
      <path d="M 237 40 L 237 150 M 243 40 L 243 150" stroke="#020617" stroke-width="4" opacity="0.8" />
      <path d="M 237 150 Q 215 170 195 190 M 243 150 Q 265 170 285 190" stroke="#020617" stroke-width="3" opacity="0.75" />

      <!-- Mediastinum & Cardiac Silhouette -->
      <path d="M 230 110 Q 240 180 220 250 Q 210 320 230 370 L 250 370 Q 270 320 260 250 Q 240 180 250 110 Z" fill="#e2e8f0" opacity="0.65" filter="url(#blur-med)" />
      <path d="M 230 230 Q 275 250 315 310 Q 330 350 280 365 Q 230 370 215 340 Q 210 280 230 230 Z" fill="#f1f5f9" opacity="0.75" filter="url(#blur-med)" />

      <!-- Bronchovascular Markings -->
      <g stroke="#cbd5e1" stroke-width="2.5" stroke-linecap="round" fill="none" opacity="0.5" filter="url(#blur-subtle)">
        <path d="M 205 180 Q 170 195 140 220 M 195 195 Q 165 230 145 265 M 190 210 Q 155 245 130 280" />
        <path d="M 275 180 Q 310 195 340 220 M 285 195 Q 315 230 335 265 M 290 210 Q 325 245 350 280" />
      </g>

      <!-- Diaphragm Domes -->
      ${type !== 'emphysema' ? `
        <path d="M 60 345 Q 130 325 210 355" stroke="#f8fafc" stroke-width="5" fill="none" opacity="0.8" filter="url(#blur-subtle)" />
        <path d="M 270 365 Q 340 335 420 345" stroke="#f8fafc" stroke-width="5" fill="none" opacity="0.8" filter="url(#blur-subtle)" />
      ` : ''}

      <!-- Disease Pathology Layer -->
      ${pathologyOverlay}

      <text x="18" y="32" fill="#38bdf8" font-size="14" font-family="monospace" font-weight="bold" opacity="0.85">R</text>
      <text x="445" y="32" fill="#38bdf8" font-size="14" font-family="monospace" font-weight="bold" opacity="0.85">L</text>
      <text x="18" y="460" fill="#64748b" font-size="10" font-family="monospace">PA ERECT • 224x224</text>
      <text x="340" y="460" fill="#64748b" font-size="10" font-family="monospace">DenseNet121 + DualHead</text>
    </svg>
  `;

  return `data:image/svg+xml;utf8,${encodeURIComponent(svgString)}`;
}

// 10 Distinct Clinical Benchmark Cases covering all diseases and severity tiers
export const SAMPLE_CASES: PatientCase[] = [
  // 1. NORMAL - Baseline (0.00 / None)
  {
    id: 'case_norm_001',
    patientId: 'PT-NRM-1092',
    groundTruth: 'Normal',
    predictedClass: 'Normal',
    confidence: 0.985,
    posteriorProbabilities: {
      Normal: 0.985,
      Pneumonia: 0.007,
      'Pulmonary Fibrosis': 0.004,
      Emphysema: 0.004,
    },
    severityScore: null,
    severityDescriptor: 'Normal Baseline (0.00 Radiographic Burden)',
    affectedLungZones: ['Clear Bilateral Parenchyma', 'Sharp Costophrenic Angles'],
    findingsDescription:
      'Clear bilateral lung fields without focal consolidation, abnormal interstitial reticulation, effusion, or pneumothorax. Normal cardiothoracic ratio (< 0.50) and sharp costophrenic angles bilaterally.',
    diseaseFeatures: [
      { label: 'Parenchymal Clarity Ratio', value: 0.99, description: 'Clear parenchyma without opacities', clinicalSignificance: 'Normal baseline' },
      { label: 'Cardiothoracic Ratio', value: 0.44, description: 'Normal heart size', clinicalSignificance: 'Normal cardiac contour' },
      { label: 'Diaphragm Sharpness', value: 0.96, description: 'Sharp acute angles', clinicalSignificance: 'Excludes effusion' },
      { label: 'Pathological Burden Metric', value: 0.00, description: 'Zero abnormality detected', clinicalSignificance: 'Masked severity regression branch' },
    ],
    dicomMetadata: {
      modality: 'DX',
      viewPosition: 'PA',
      patientAge: 32,
      patientSex: 'F',
      kVp: 115,
      exposureTimeMs: 8,
      pixelSpacing: '0.139\\0.139 mm',
      matrixSize: '2048 x 2048 -> 224 x 224',
      photometricInterpretation: 'MONOCHROME2',
    },
    imageSrc: createChestXraySvg('normal', 'none'),
    hasGroundTruthBBox: false,
    spatialAttentionGrid: [
      [0.01, 0.02, 0.02, 0.03, 0.02, 0.02, 0.01],
      [0.02, 0.04, 0.05, 0.06, 0.05, 0.04, 0.02],
      [0.03, 0.06, 0.08, 0.09, 0.08, 0.06, 0.03],
      [0.04, 0.08, 0.10, 0.12, 0.10, 0.08, 0.04],
      [0.03, 0.07, 0.09, 0.11, 0.09, 0.07, 0.03],
      [0.02, 0.05, 0.06, 0.07, 0.06, 0.05, 0.02],
      [0.01, 0.02, 0.02, 0.03, 0.02, 0.02, 0.01],
    ],
    gradCamGrid: Array(7).fill(Array(7).fill(0.04)),
    gradCamPlusPlusGrid: Array(7).fill(Array(7).fill(0.03)),
    layerCamGrid: Array(7).fill(Array(7).fill(0.04)),
  },

  // 2. PNEUMONIA - MILD (0.24 / Grade 1 Mild)
  {
    id: 'case_pnm_mild',
    patientId: 'PT-PNM-1102',
    groundTruth: 'Pneumonia',
    predictedClass: 'Pneumonia',
    confidence: 0.932,
    posteriorProbabilities: {
      Normal: 0.042,
      Pneumonia: 0.932,
      'Pulmonary Fibrosis': 0.018,
      Emphysema: 0.008,
    },
    severityScore: 0.24,
    severityDescriptor: 'Mild Focal Subsegmental Opacity (0.24 Burden)',
    affectedLungZones: ['Right Lower Lobe (Subsegmental)'],
    findingsDescription:
      'Early mild focal airspace opacity localized to the right lower lobe posterior basilar segment. No confluent consolidation or air bronchograms. Left lung clear with normal costophrenic angles.',
    diseaseFeatures: [
      { label: 'Airspace Opacity Extent', value: 0.24, description: 'Focal involvement < 15% of right lung', clinicalSignificance: 'Grade 1 Mild localized infection' },
      { label: 'Consolidation Density', value: 0.32, description: 'Low density hazy infiltrate', clinicalSignificance: 'Early stage alveolar exudate' },
      { label: 'Bilateral Involvement', value: 0.00, description: 'Unilateral right lower lobe only', clinicalSignificance: 'Isolated focal process' },
    ],
    dicomMetadata: {
      modality: 'CR',
      viewPosition: 'PA',
      patientAge: 44,
      patientSex: 'M',
      kVp: 120,
      exposureTimeMs: 12,
      pixelSpacing: '0.143\\0.143 mm',
      matrixSize: '2048 x 2048 -> 224 x 224',
      photometricInterpretation: 'MONOCHROME2',
    },
    imageSrc: createChestXraySvg('pneumonia', 'mild'),
    hasGroundTruthBBox: true,
    groundTruthBBox: { x: 120, y: 260, width: 60, height: 60, label: 'Focal Opacity' },
    iouScore: 0.842,
    diceScore: 0.912,
    spatialAttentionGrid: [
      [0.01, 0.02, 0.02, 0.03, 0.02, 0.02, 0.01],
      [0.02, 0.05, 0.08, 0.06, 0.04, 0.03, 0.01],
      [0.04, 0.18, 0.25, 0.12, 0.06, 0.04, 0.02],
      [0.06, 0.42, 0.58, 0.22, 0.08, 0.05, 0.02],
      [0.05, 0.48, 0.62, 0.25, 0.09, 0.06, 0.02],
      [0.03, 0.22, 0.32, 0.14, 0.06, 0.04, 0.01],
      [0.01, 0.04, 0.06, 0.04, 0.03, 0.02, 0.01],
    ],
    gradCamGrid: Array(7).fill(Array(7).fill(0.08)),
    gradCamPlusPlusGrid: [
      [0.01, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01],
      [0.02, 0.08, 0.14, 0.08, 0.03, 0.02, 0.01],
      [0.04, 0.35, 0.52, 0.20, 0.06, 0.03, 0.01],
      [0.08, 0.68, 0.88, 0.34, 0.09, 0.04, 0.02],
      [0.06, 0.72, 0.94, 0.38, 0.10, 0.05, 0.02],
      [0.03, 0.30, 0.42, 0.18, 0.06, 0.03, 0.01],
      [0.01, 0.04, 0.06, 0.04, 0.02, 0.01, 0.01],
    ],
    layerCamGrid: Array(7).fill(Array(7).fill(0.1)),
  },

  // 3. PNEUMONIA - MODERATE (0.54 / Grade 2 Moderate)
  {
    id: 'case_pnm_mod',
    patientId: 'PT-PNM-3419',
    groundTruth: 'Pneumonia',
    predictedClass: 'Pneumonia',
    confidence: 0.954,
    posteriorProbabilities: {
      Normal: 0.014,
      Pneumonia: 0.954,
      'Pulmonary Fibrosis': 0.022,
      Emphysema: 0.010,
    },
    severityScore: 0.54,
    severityDescriptor: 'Moderate Unilateral Lobar Consolidation (0.54 Burden)',
    affectedLungZones: ['Right Lower Lobe', 'Right Mid Zone'],
    findingsDescription:
      'Moderate confluent alveolar airspace consolidation occupying the right lower and mid lung zones with visible air bronchograms. Mild reactive blunting of the right lateral costophrenic angle.',
    diseaseFeatures: [
      { label: 'Airspace Opacity Extent', value: 0.54, description: 'Lobar consolidation ~32% total lung area', clinicalSignificance: 'Grade 2 Moderate lobar pneumonia' },
      { label: 'Consolidation Density', value: 0.68, description: 'Dense opacity with air bronchograms', clinicalSignificance: 'Active dense exudative filling' },
      { label: 'Bilateral Involvement', value: 0.18, description: 'Predominantly unilateral right lung', clinicalSignificance: 'Unilateral lobar pattern' },
    ],
    dicomMetadata: {
      modality: 'CR / DX',
      viewPosition: 'PA',
      patientAge: 56,
      patientSex: 'M',
      kVp: 120,
      exposureTimeMs: 14,
      pixelSpacing: '0.139\\0.139 mm',
      matrixSize: '2048 x 2048 -> 224 x 224',
      photometricInterpretation: 'MONOCHROME2',
    },
    imageSrc: createChestXraySvg('pneumonia', 'moderate'),
    hasGroundTruthBBox: true,
    groundTruthBBox: { x: 90, y: 220, width: 110, height: 130, label: 'Lobar Consolidation' },
    iouScore: 0.814,
    diceScore: 0.887,
    spatialAttentionGrid: [
      [0.02, 0.04, 0.05, 0.06, 0.05, 0.04, 0.02],
      [0.05, 0.14, 0.22, 0.16, 0.12, 0.08, 0.03],
      [0.08, 0.42, 0.65, 0.32, 0.18, 0.12, 0.04],
      [0.12, 0.72, 0.95, 0.68, 0.28, 0.18, 0.06],
      [0.10, 0.78, 0.98, 0.72, 0.35, 0.25, 0.07],
      [0.06, 0.48, 0.68, 0.45, 0.22, 0.18, 0.04],
      [0.02, 0.08, 0.12, 0.10, 0.08, 0.06, 0.02],
    ],
    gradCamGrid: Array(7).fill(Array(7).fill(0.12)),
    gradCamPlusPlusGrid: [
      [0.01, 0.02, 0.03, 0.03, 0.02, 0.01, 0.01],
      [0.03, 0.15, 0.25, 0.16, 0.08, 0.04, 0.02],
      [0.06, 0.48, 0.74, 0.38, 0.14, 0.08, 0.03],
      [0.10, 0.82, 0.98, 0.76, 0.24, 0.14, 0.05],
      [0.08, 0.88, 1.00, 0.82, 0.32, 0.20, 0.06],
      [0.04, 0.55, 0.74, 0.50, 0.18, 0.12, 0.03],
      [0.01, 0.06, 0.10, 0.08, 0.05, 0.03, 0.01],
    ],
    layerCamGrid: Array(7).fill(Array(7).fill(0.15)),
  },

  // 4. PNEUMONIA - SEVERE (0.86 / Grade 3 Severe)
  {
    id: 'case_pnm_sev',
    patientId: 'PT-PNM-8930',
    groundTruth: 'Pneumonia',
    predictedClass: 'Pneumonia',
    confidence: 0.978,
    posteriorProbabilities: {
      Normal: 0.003,
      Pneumonia: 0.978,
      'Pulmonary Fibrosis': 0.014,
      Emphysema: 0.005,
    },
    severityScore: 0.86,
    severityDescriptor: 'Severe Bilateral Multi-Lobar Consolidation (0.86 Burden)',
    affectedLungZones: ['Bilateral Lower Zones', 'Right Mid Zone', 'Retrocardiac Region'],
    findingsDescription:
      'Extensive dense bilateral confluent airspace consolidation with widespread air bronchograms involving the right lower/mid zones and left basilar retrocardiac parenchyma. High acute respiratory risk.',
    diseaseFeatures: [
      { label: 'Airspace Opacity Extent', value: 0.88, description: 'Bilateral consolidation > 55% lung fields', clinicalSignificance: 'Grade 3 Severe acute multi-lobar pneumonia' },
      { label: 'Consolidation Density', value: 0.94, description: 'Dense opacification with air bronchograms', clinicalSignificance: 'Near total alveolar volume filling' },
      { label: 'Bilateral Involvement', value: 0.92, description: 'Diffuse bilateral lower & mid lobe involvement', clinicalSignificance: 'Critical respiratory compromise' },
    ],
    dicomMetadata: {
      modality: 'DX',
      viewPosition: 'AP Portable',
      patientAge: 68,
      patientSex: 'M',
      kVp: 125,
      exposureTimeMs: 18,
      pixelSpacing: '0.143\\0.143 mm',
      matrixSize: '2048 x 2048 -> 224 x 224',
      photometricInterpretation: 'MONOCHROME2',
    },
    imageSrc: createChestXraySvg('pneumonia', 'severe'),
    hasGroundTruthBBox: true,
    groundTruthBBox: { x: 75, y: 190, width: 330, height: 170, label: 'Bilateral Multi-Lobar Opacity' },
    iouScore: 0.875,
    diceScore: 0.933,
    spatialAttentionGrid: [
      [0.03, 0.06, 0.08, 0.08, 0.08, 0.06, 0.03],
      [0.08, 0.28, 0.42, 0.32, 0.38, 0.28, 0.08],
      [0.15, 0.65, 0.88, 0.55, 0.72, 0.62, 0.16],
      [0.22, 0.92, 1.00, 0.85, 0.95, 0.88, 0.24],
      [0.20, 0.95, 0.98, 0.88, 0.96, 0.92, 0.22],
      [0.12, 0.68, 0.82, 0.65, 0.75, 0.68, 0.14],
      [0.04, 0.14, 0.20, 0.16, 0.18, 0.15, 0.04],
    ],
    gradCamGrid: Array(7).fill(Array(7).fill(0.2)),
    gradCamPlusPlusGrid: [
      [0.02, 0.04, 0.06, 0.06, 0.06, 0.04, 0.02],
      [0.06, 0.32, 0.50, 0.36, 0.44, 0.32, 0.06],
      [0.12, 0.74, 0.94, 0.62, 0.82, 0.70, 0.14],
      [0.20, 0.96, 1.00, 0.90, 0.98, 0.94, 0.22],
      [0.18, 0.98, 0.99, 0.92, 0.98, 0.96, 0.20],
      [0.10, 0.72, 0.86, 0.70, 0.80, 0.74, 0.12],
      [0.03, 0.12, 0.18, 0.14, 0.16, 0.12, 0.03],
    ],
    layerCamGrid: Array(7).fill(Array(7).fill(0.25)),
  },

  // 5. PULMONARY FIBROSIS - MILD (0.28 / Grade 1 Mild)
  {
    id: 'case_fib_mild',
    patientId: 'PT-FIB-1940',
    groundTruth: 'Pulmonary Fibrosis',
    predictedClass: 'Pulmonary Fibrosis',
    confidence: 0.924,
    posteriorProbabilities: {
      Normal: 0.045,
      Pneumonia: 0.021,
      'Pulmonary Fibrosis': 0.924,
      Emphysema: 0.010,
    },
    severityScore: 0.28,
    severityDescriptor: 'Mild Early Subpleural Basilar Reticulation (0.28 Burden)',
    affectedLungZones: ['Subpleural Periphery', 'Bilateral Basilar Margins'],
    findingsDescription:
      'Early mild fine reticular interstitial markings localized strictly to the subpleural basilar periphery. No honeycombing cysts or traction bronchiectasis. Preserved lung volumes.',
    diseaseFeatures: [
      { label: 'Reticular Pattern Extent', value: 0.28, description: 'Fine peripheral basilar reticulation', clinicalSignificance: 'Grade 1 Early Interstitial Lung Disease' },
      { label: 'Honeycombing Index', value: 0.05, description: 'No macrocystic honeycombing detected', clinicalSignificance: 'Absence of end-stage destruction' },
      { label: 'Volume Loss Ratio', value: 0.15, description: 'Normal diaphragmatic excursion and height', clinicalSignificance: 'No restrictive volume contraction' },
    ],
    dicomMetadata: {
      modality: 'DX',
      viewPosition: 'PA',
      patientAge: 61,
      patientSex: 'F',
      kVp: 120,
      exposureTimeMs: 14,
      pixelSpacing: '0.143\\0.143 mm',
      matrixSize: '2048 x 2048 -> 224 x 224',
      photometricInterpretation: 'MONOCHROME2',
    },
    imageSrc: createChestXraySvg('fibrosis', 'mild'),
    hasGroundTruthBBox: true,
    groundTruthBBox: { x: 80, y: 310, width: 310, height: 70, label: 'Basilar Peripheral Reticulation' },
    iouScore: 0.795,
    diceScore: 0.874,
    spatialAttentionGrid: [
      [0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01],
      [0.02, 0.04, 0.06, 0.05, 0.06, 0.04, 0.02],
      [0.04, 0.08, 0.10, 0.07, 0.11, 0.09, 0.04],
      [0.08, 0.22, 0.20, 0.12, 0.22, 0.26, 0.08],
      [0.18, 0.52, 0.44, 0.20, 0.45, 0.55, 0.18],
      [0.22, 0.65, 0.58, 0.28, 0.60, 0.68, 0.22],
      [0.06, 0.20, 0.22, 0.10, 0.21, 0.24, 0.06],
    ],
    gradCamGrid: Array(7).fill(Array(7).fill(0.09)),
    gradCamPlusPlusGrid: [
      [0.01, 0.01, 0.02, 0.02, 0.02, 0.01, 0.01],
      [0.02, 0.04, 0.05, 0.04, 0.05, 0.04, 0.02],
      [0.03, 0.10, 0.12, 0.06, 0.14, 0.12, 0.03],
      [0.09, 0.28, 0.25, 0.10, 0.26, 0.32, 0.08],
      [0.22, 0.62, 0.50, 0.16, 0.52, 0.65, 0.20],
      [0.28, 0.75, 0.68, 0.22, 0.70, 0.78, 0.25],
      [0.05, 0.18, 0.20, 0.08, 0.18, 0.20, 0.05],
    ],
    layerCamGrid: Array(7).fill(Array(7).fill(0.12)),
  },

  // 6. PULMONARY FIBROSIS - MODERATE (0.58 / Grade 2 Moderate)
  {
    id: 'case_fib_mod',
    patientId: 'PT-FIB-4902',
    groundTruth: 'Pulmonary Fibrosis',
    predictedClass: 'Pulmonary Fibrosis',
    confidence: 0.946,
    posteriorProbabilities: {
      Normal: 0.010,
      Pneumonia: 0.026,
      'Pulmonary Fibrosis': 0.946,
      Emphysema: 0.018,
    },
    severityScore: 0.58,
    severityDescriptor: 'Moderate Bilateral Basilar Reticular Network (0.58 Burden)',
    affectedLungZones: ['Bilateral Lower Zones', 'Subpleural Periphery', 'Basilar Parenchyma'],
    findingsDescription:
      'Moderate bilateral subpleural and basilar reticular opacities with early architectural distortion and subtle traction markers. Mild reduction in lung volumes with elevated diaphragmatic domes.',
    diseaseFeatures: [
      { label: 'Reticular Pattern Extent', value: 0.62, description: 'Bilateral lower lobe reticular network', clinicalSignificance: 'Grade 2 Established Interstitial Disease' },
      { label: 'Honeycombing Index', value: 0.42, description: 'Focal microcystic subpleural lucencies', clinicalSignificance: 'Structural parenchymal remodeling' },
      { label: 'Volume Loss Ratio', value: 0.50, description: 'Mild-to-moderate volume restriction', clinicalSignificance: 'Restrictive pulmonary impairment' },
    ],
    dicomMetadata: {
      modality: 'DX',
      viewPosition: 'PA',
      patientAge: 68,
      patientSex: 'M',
      kVp: 125,
      exposureTimeMs: 16,
      pixelSpacing: '0.143\\0.143 mm',
      matrixSize: '2048 x 2048 -> 224 x 224',
      photometricInterpretation: 'MONOCHROME2',
    },
    imageSrc: createChestXraySvg('fibrosis', 'moderate'),
    hasGroundTruthBBox: true,
    groundTruthBBox: { x: 70, y: 280, width: 340, height: 110, label: 'Bilateral Basilar Infiltrate' },
    iouScore: 0.782,
    diceScore: 0.864,
    spatialAttentionGrid: [
      [0.01, 0.02, 0.03, 0.03, 0.03, 0.02, 0.01],
      [0.03, 0.06, 0.08, 0.07, 0.08, 0.06, 0.03],
      [0.05, 0.12, 0.15, 0.10, 0.16, 0.14, 0.05],
      [0.12, 0.38, 0.35, 0.18, 0.36, 0.42, 0.12],
      [0.28, 0.78, 0.62, 0.30, 0.65, 0.82, 0.26],
      [0.35, 0.92, 0.88, 0.45, 0.85, 0.94, 0.32],
      [0.10, 0.35, 0.38, 0.18, 0.34, 0.38, 0.09],
    ],
    gradCamGrid: Array(7).fill(Array(7).fill(0.14)),
    gradCamPlusPlusGrid: [
      [0.01, 0.01, 0.02, 0.02, 0.02, 0.01, 0.01],
      [0.02, 0.04, 0.06, 0.05, 0.06, 0.04, 0.02],
      [0.03, 0.16, 0.18, 0.08, 0.20, 0.18, 0.03],
      [0.15, 0.48, 0.42, 0.16, 0.44, 0.52, 0.14],
      [0.36, 0.88, 0.72, 0.25, 0.74, 0.91, 0.33],
      [0.45, 0.99, 0.96, 0.39, 0.92, 1.00, 0.40],
      [0.08, 0.28, 0.30, 0.12, 0.28, 0.30, 0.07],
    ],
    layerCamGrid: Array(7).fill(Array(7).fill(0.18)),
  },

  // 7. PULMONARY FIBROSIS - SEVERE (0.90 / Grade 3 Severe)
  {
    id: 'case_fib_sev',
    patientId: 'PT-FIB-7104',
    groundTruth: 'Pulmonary Fibrosis',
    predictedClass: 'Pulmonary Fibrosis',
    confidence: 0.968,
    posteriorProbabilities: {
      Normal: 0.004,
      Pneumonia: 0.018,
      'Pulmonary Fibrosis': 0.968,
      Emphysema: 0.010,
    },
    severityScore: 0.90,
    severityDescriptor: 'Severe Honeycombing & End-Stage UIP Fibrosis (0.90 Burden)',
    affectedLungZones: ['Bilateral Lower Zones', 'Bilateral Mid Zones', 'Subpleural Mantle'],
    findingsDescription:
      'Severe end-stage Usual Interstitial Pneumonia (UIP) pattern with coarse honeycombing cysts, extensive traction bronchiectasis, profound parenchymal distortion, and marked restrictive lung volume loss.',
    diseaseFeatures: [
      { label: 'Reticular Pattern Extent', value: 0.92, description: 'Coarse reticulation over bilateral lower 2/3', clinicalSignificance: 'Grade 3 Severe advanced fibrosis' },
      { label: 'Honeycombing Index', value: 0.94, description: 'Stacked subpleural microcystic clusters', clinicalSignificance: 'Irreversible alveolar destruction' },
      { label: 'Volume Loss Ratio', value: 0.88, description: 'Marked diaphragmatic elevation & shrunken thorax', clinicalSignificance: 'Severe restrictive physiological defect' },
    ],
    dicomMetadata: {
      modality: 'DX',
      viewPosition: 'PA',
      patientAge: 74,
      patientSex: 'M',
      kVp: 125,
      exposureTimeMs: 18,
      pixelSpacing: '0.143\\0.143 mm',
      matrixSize: '2048 x 2048 -> 224 x 224',
      photometricInterpretation: 'MONOCHROME2',
    },
    imageSrc: createChestXraySvg('fibrosis', 'severe'),
    hasGroundTruthBBox: true,
    groundTruthBBox: { x: 65, y: 240, width: 350, height: 160, label: 'Severe Honeycomb Architecture' },
    iouScore: 0.842,
    diceScore: 0.910,
    spatialAttentionGrid: [
      [0.02, 0.03, 0.04, 0.04, 0.04, 0.03, 0.02],
      [0.05, 0.12, 0.16, 0.14, 0.16, 0.12, 0.05],
      [0.12, 0.32, 0.38, 0.25, 0.38, 0.35, 0.12],
      [0.25, 0.68, 0.65, 0.40, 0.68, 0.72, 0.25],
      [0.45, 0.94, 0.88, 0.55, 0.90, 0.98, 0.44],
      [0.55, 1.00, 0.98, 0.65, 0.96, 1.00, 0.52],
      [0.18, 0.45, 0.50, 0.28, 0.48, 0.52, 0.16],
    ],
    gradCamGrid: Array(7).fill(Array(7).fill(0.2)),
    gradCamPlusPlusGrid: [
      [0.01, 0.02, 0.03, 0.03, 0.03, 0.02, 0.01],
      [0.04, 0.10, 0.14, 0.10, 0.14, 0.10, 0.04],
      [0.10, 0.35, 0.42, 0.22, 0.42, 0.38, 0.10],
      [0.28, 0.78, 0.75, 0.38, 0.76, 0.84, 0.28],
      [0.52, 0.98, 0.92, 0.52, 0.94, 1.00, 0.50],
      [0.60, 1.00, 0.99, 0.60, 0.98, 1.00, 0.58],
      [0.15, 0.42, 0.46, 0.22, 0.44, 0.48, 0.14],
    ],
    layerCamGrid: Array(7).fill(Array(7).fill(0.24)),
  },

  // 8. EMPHYSEMA - MILD (0.22 / Grade 1 Mild)
  {
    id: 'case_emp_mild',
    patientId: 'PT-EMP-2041',
    groundTruth: 'Emphysema',
    predictedClass: 'Emphysema',
    confidence: 0.918,
    posteriorProbabilities: {
      Normal: 0.052,
      Pneumonia: 0.014,
      'Pulmonary Fibrosis': 0.016,
      Emphysema: 0.918,
    },
    severityScore: 0.22,
    severityDescriptor: 'Mild Apical Hyperlucency & Oligemia (0.22 Burden)',
    affectedLungZones: ['Bilateral Apical Regions'],
    findingsDescription:
      'Early mild bilateral apical hyperlucency with slight attenuation of peripheral bronchovascular markings. Normal diaphragmatic contour and cardiothoracic diameter.',
    diseaseFeatures: [
      { label: 'Apical Hyperlucency', value: 0.26, description: 'Mild decreased attenuation in apices', clinicalSignificance: 'Grade 1 Early Centrilobular Emphysema' },
      { label: 'Vascular Attenuation', value: 0.22, description: 'Mild distal pruning in upper zones', clinicalSignificance: 'Minimal capillary loss' },
      { label: 'Diaphragm Flattening', value: 0.10, description: 'Normal dome curvature preserved', clinicalSignificance: 'No significant air trapping' },
    ],
    dicomMetadata: {
      modality: 'CR',
      viewPosition: 'PA',
      patientAge: 59,
      patientSex: 'M',
      kVp: 120,
      exposureTimeMs: 10,
      pixelSpacing: '0.140\\0.140 mm',
      matrixSize: '2048 x 2048 -> 224 x 224',
      photometricInterpretation: 'MONOCHROME2',
    },
    imageSrc: createChestXraySvg('emphysema', 'mild'),
    hasGroundTruthBBox: true,
    groundTruthBBox: { x: 80, y: 110, width: 320, height: 130, label: 'Apical Lucency' },
    iouScore: 0.772,
    diceScore: 0.852,
    spatialAttentionGrid: [
      [0.03, 0.12, 0.24, 0.08, 0.22, 0.14, 0.03],
      [0.06, 0.38, 0.55, 0.14, 0.52, 0.40, 0.06],
      [0.07, 0.45, 0.58, 0.16, 0.56, 0.48, 0.07],
      [0.05, 0.28, 0.35, 0.12, 0.34, 0.30, 0.05],
      [0.03, 0.14, 0.18, 0.08, 0.16, 0.15, 0.03],
      [0.02, 0.08, 0.10, 0.05, 0.09, 0.08, 0.02],
      [0.01, 0.03, 0.04, 0.02, 0.04, 0.03, 0.01],
    ],
    gradCamGrid: Array(7).fill(Array(7).fill(0.08)),
    gradCamPlusPlusGrid: [
      [0.02, 0.15, 0.30, 0.10, 0.28, 0.16, 0.02],
      [0.07, 0.48, 0.68, 0.16, 0.65, 0.50, 0.07],
      [0.09, 0.55, 0.74, 0.20, 0.72, 0.58, 0.09],
      [0.06, 0.32, 0.42, 0.14, 0.40, 0.35, 0.06],
      [0.03, 0.15, 0.18, 0.08, 0.16, 0.14, 0.03],
      [0.02, 0.08, 0.10, 0.05, 0.08, 0.07, 0.02],
      [0.01, 0.02, 0.03, 0.02, 0.03, 0.02, 0.01],
    ],
    layerCamGrid: Array(7).fill(Array(7).fill(0.1)),
  },

  // 9. EMPHYSEMA - MODERATE (0.52 / Grade 2 Moderate)
  {
    id: 'case_emp_mod',
    patientId: 'PT-EMP-5891',
    groundTruth: 'Emphysema',
    predictedClass: 'Emphysema',
    confidence: 0.941,
    posteriorProbabilities: {
      Normal: 0.018,
      Pneumonia: 0.016,
      'Pulmonary Fibrosis': 0.025,
      Emphysema: 0.941,
    },
    severityScore: 0.52,
    severityDescriptor: 'Moderate Bilateral Hyperinflation (0.52 Burden)',
    affectedLungZones: ['Bilateral Upper Zones', 'Bilateral Mid Zones', 'Diaphragm Bases'],
    findingsDescription:
      'Moderate bilateral lung hyperlucency with vascular attenuation in the upper lobes. Hyperinflated lungs with flattened hemidiaphragmatic domes and slight widening of intercostal rib spaces.',
    diseaseFeatures: [
      { label: 'Parenchymal Hyperlucency', value: 0.58, description: 'Decreased attenuation over bilateral upper halves', clinicalSignificance: 'Grade 2 Moderate alveolar destruction' },
      { label: 'Vascular Attenuation', value: 0.55, description: 'Distal vascular pruning with central hilar prominence', clinicalSignificance: 'Capillary bed destruction' },
      { label: 'Diaphragm Flattening', value: 0.52, description: 'Low flattened diaphragms at 10th posterior rib', clinicalSignificance: 'Moderate chronic air trapping' },
    ],
    dicomMetadata: {
      modality: 'CR',
      viewPosition: 'PA',
      patientAge: 69,
      patientSex: 'M',
      kVp: 120,
      exposureTimeMs: 12,
      pixelSpacing: '0.140\\0.140 mm',
      matrixSize: '2048 x 2048 -> 224 x 224',
      photometricInterpretation: 'MONOCHROME2',
    },
    imageSrc: createChestXraySvg('emphysema', 'moderate'),
    hasGroundTruthBBox: true,
    groundTruthBBox: { x: 75, y: 110, width: 330, height: 260, label: 'Hyperlucent Lung Fields' },
    iouScore: 0.765,
    diceScore: 0.842,
    spatialAttentionGrid: [
      [0.04, 0.15, 0.32, 0.10, 0.30, 0.18, 0.04],
      [0.08, 0.54, 0.78, 0.18, 0.75, 0.58, 0.09],
      [0.10, 0.65, 0.84, 0.22, 0.82, 0.69, 0.11],
      [0.09, 0.48, 0.62, 0.20, 0.59, 0.52, 0.10],
      [0.07, 0.32, 0.38, 0.16, 0.35, 0.34, 0.08],
      [0.08, 0.45, 0.52, 0.18, 0.48, 0.46, 0.09],
      [0.03, 0.12, 0.15, 0.08, 0.14, 0.12, 0.03],
    ],
    gradCamGrid: Array(7).fill(Array(7).fill(0.12)),
    gradCamPlusPlusGrid: [
      [0.03, 0.22, 0.42, 0.14, 0.40, 0.24, 0.03],
      [0.10, 0.65, 0.91, 0.22, 0.89, 0.68, 0.11],
      [0.14, 0.78, 0.98, 0.28, 0.95, 0.81, 0.15],
      [0.11, 0.58, 0.74, 0.24, 0.70, 0.62, 0.12],
      [0.05, 0.28, 0.32, 0.12, 0.30, 0.28, 0.06],
      [0.10, 0.52, 0.61, 0.22, 0.58, 0.54, 0.11],
      [0.02, 0.08, 0.10, 0.05, 0.09, 0.07, 0.02],
    ],
    layerCamGrid: Array(7).fill(Array(7).fill(0.15)),
  },

  // 10. EMPHYSEMA - SEVERE (0.82 / Grade 3 Severe)
  {
    id: 'case_emp_sev',
    patientId: 'PT-EMP-9351',
    groundTruth: 'Emphysema',
    predictedClass: 'Emphysema',
    confidence: 0.965,
    posteriorProbabilities: {
      Normal: 0.006,
      Pneumonia: 0.012,
      'Pulmonary Fibrosis': 0.017,
      Emphysema: 0.965,
    },
    severityScore: 0.82,
    severityDescriptor: 'Severe Bullous Hyperinflation & Vascular Depletion (0.82 Burden)',
    affectedLungZones: ['Bilateral Whole Lung Fields', 'Bilateral Diaphragmatic Domes', 'Retrosternal Space'],
    findingsDescription:
      'Severe end-stage pulmonary emphysema with profound diffuse hyperlucency, massive vascular pruning, low completely flattened diaphragmatic domes below the 11th rib, and barrel chest thoracic deformity.',
    diseaseFeatures: [
      { label: 'Parenchymal Hyperlucency', value: 0.88, description: 'Profound diffuse radiolucency throughout thorax', clinicalSignificance: 'Grade 3 Severe panlobular destruction' },
      { label: 'Vascular Attenuation', value: 0.92, description: 'Marked capillary depletion with string-like vessels', clinicalSignificance: 'Severe chronic pulmonary vascular loss' },
      { label: 'Diaphragm Flattening', value: 0.85, description: 'Inversion/flat domes below 11th rib', clinicalSignificance: 'Massive chronic hyperinflation' },
    ],
    dicomMetadata: {
      modality: 'CR',
      viewPosition: 'PA',
      patientAge: 76,
      patientSex: 'M',
      kVp: 120,
      exposureTimeMs: 14,
      pixelSpacing: '0.140\\0.140 mm',
      matrixSize: '2048 x 2048 -> 224 x 224',
      photometricInterpretation: 'MONOCHROME2',
    },
    imageSrc: createChestXraySvg('emphysema', 'severe'),
    hasGroundTruthBBox: true,
    groundTruthBBox: { x: 65, y: 90, width: 350, height: 300, label: 'Severe Hyperinflation' },
    iouScore: 0.825,
    diceScore: 0.895,
    spatialAttentionGrid: [
      [0.06, 0.25, 0.45, 0.18, 0.42, 0.28, 0.06],
      [0.12, 0.72, 0.92, 0.28, 0.88, 0.75, 0.12],
      [0.16, 0.85, 0.98, 0.35, 0.95, 0.88, 0.16],
      [0.14, 0.68, 0.82, 0.30, 0.78, 0.72, 0.14],
      [0.10, 0.45, 0.55, 0.22, 0.52, 0.48, 0.10],
      [0.14, 0.65, 0.74, 0.28, 0.70, 0.68, 0.14],
      [0.05, 0.20, 0.25, 0.12, 0.22, 0.18, 0.05],
    ],
    gradCamGrid: Array(7).fill(Array(7).fill(0.18)),
    gradCamPlusPlusGrid: [
      [0.05, 0.30, 0.55, 0.22, 0.52, 0.35, 0.05],
      [0.15, 0.80, 0.98, 0.32, 0.95, 0.84, 0.15],
      [0.20, 0.92, 1.00, 0.40, 0.98, 0.94, 0.20],
      [0.16, 0.75, 0.88, 0.34, 0.85, 0.78, 0.16],
      [0.08, 0.40, 0.48, 0.18, 0.45, 0.42, 0.08],
      [0.15, 0.70, 0.82, 0.30, 0.78, 0.74, 0.15],
      [0.04, 0.15, 0.18, 0.08, 0.16, 0.12, 0.04],
    ],
    layerCamGrid: Array(7).fill(Array(7).fill(0.22)),
  },
];
