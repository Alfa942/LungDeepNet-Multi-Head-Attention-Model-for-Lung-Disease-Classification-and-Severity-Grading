import React, { useState } from 'react';
import { X, BarChart3, Download, Copy, Check, Image as ImageIcon } from 'lucide-react';

interface ReportFigureItem {
  id: string;
  number: string;
  title: string;
  section: string;
  src: string;
  caption: string;
}

const REPORT_FIGURES: ReportFigureItem[] = [
  {
    id: 'fig1',
    number: 'Figure 1',
    title: 'Multi-Source Dataset Distribution & Stratified Splits',
    section: 'Chapter 3: Dataset Acquisition',
    src: '/figures/fig1_dataset_distribution.svg',
    caption:
      'Figure 1: Multi-source chest radiograph dataset distribution (N = 10,025 scans across 8,350 unique patients) illustrating strict patient-level stratified splitting (75% training, 12.5% validation, 12.5% holdout testing) across Normal, Pneumonia, Emphysema, and Pulmonary Fibrosis.'
  },
  {
    id: 'fig2',
    number: 'Figure 2',
    title: 'Preprocessing & Noise Removal Evaluation (Objective 1)',
    section: 'Chapter 4: Preprocessing Pipeline',
    src: '/figures/fig2_preprocessing_comparison.svg',
    caption:
      'Figure 2: Quantitative comparative evaluation of nine image denoising and contrast enhancement pipelines measuring Peak Signal-to-Noise Ratio (PSNR) and downstream classification accuracy.'
  },
  {
    id: 'fig3',
    number: 'Figure 3',
    title: 'Two-Phase Training & Validation Convergence Curves',
    section: 'Chapter 5: Model Training',
    src: '/figures/fig3_training_convergence.svg',
    caption:
      'Figure 3: Two-phase training and validation convergence curves over 25 epochs demonstrating stable loss minimization during Phase 1 backbone warmup (epochs 1–5) and Phase 2 DenseBlock4 fine-tuning with Cosine Annealing scheduling (epochs 6–25).'
  },
  {
    id: 'fig4',
    number: 'Figure 4',
    title: 'Held-Out Test Set Confusion Matrix (N = 1,254)',
    section: 'Chapter 7: Results & Discussion',
    src: '/figures/fig4_confusion_matrix.svg',
    caption:
      'Figure 4: Multi-class confusion matrix evaluated on the unseen holdout test cohort (N = 1,254, zero patient overlap), achieving an overall classification accuracy of 95.4%.'
  },
  {
    id: 'fig5',
    number: 'Figure 5',
    title: 'One-vs-Rest Multi-Class ROC Curves',
    section: 'Chapter 7: Results & Discussion',
    src: '/figures/fig5_roc_curves.svg',
    caption:
      'Figure 5: One-vs-Rest Receiver Operating Characteristic (ROC) curves across all four thoracic classes yielding a macro-averaged AUROC of 0.982.'
  },
  {
    id: 'fig6',
    number: 'Figure 6',
    title: 'Four-Way Architectural Ablation Study',
    section: 'Chapter 7: Ablation Analysis',
    src: '/figures/fig6_ablation_study.svg',
    caption:
      'Figure 6: Four-way architectural ablation study isolating the effects of Spatial Attention (driving lesion localization IoU from 0.612 to 0.814) and Channel Squeeze-and-Excitation Attention (reducing pneumonia severity MAE from 0.089 to 0.048).'
  },
  {
    id: 'fig7',
    number: 'Figure 7',
    title: 'Optuna HPO & Parameter Importance (Objective 3)',
    section: 'Chapter 6: Hyperparameter Tuning',
    src: '/figures/fig7_optuna_hpo.svg',
    caption:
      'Figure 7: Optuna Tree-structured Parzen Estimator (TPE) hyperparameter optimization convergence comparison against Grid Search, Random Search, Genetic Algorithm, and ASHA, alongside fANOVA parameter sensitivity importance scores.'
  },
  {
    id: 'fig8',
    number: 'Figure 8',
    title: 'State-of-the-Art Model Comparison (Objective 4)',
    section: 'Chapter 7: SOTA Benchmarking',
    src: '/figures/fig8_sota_benchmark.svg',
    caption:
      'Figure 8: State-of-the-art benchmarking comparing LungDeepNet against CNN and Vision Transformer baselines across classification accuracy and macro AUROC.'
  }
];

interface ReportFiguresModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ReportFiguresModal: React.FC<ReportFiguresModalProps> = ({ isOpen, onClose }) => {
  const [selectedId, setSelectedId] = useState<string>(REPORT_FIGURES[0].id);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  if (!isOpen) return null;

  const activeFig = REPORT_FIGURES.find((f) => f.id === selectedId) || REPORT_FIGURES[0];

  const handleCopyCaption = (fig: ReportFigureItem) => {
    navigator.clipboard.writeText(fig.caption);
    setCopiedId(fig.id);
    setTimeout(() => setCopiedId(null), 1800);
  };

  const handleDownloadPng = async (fig: ReportFigureItem) => {
    try {
      const response = await fetch(fig.src);
      const svgText = await response.text();
      const svgBlob = new Blob([svgText], { type: 'image/svg+xml;charset=utf-8' });
      const url = URL.createObjectURL(svgBlob);

      const img = new Image();
      img.onload = () => {
        const scale = 3; // High-res export (~300 DPI equivalent)
        const canvas = document.createElement('canvas');
        canvas.width = img.width * scale;
        canvas.height = img.height * scale;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.fillStyle = '#ffffff';
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          const pngUrl = canvas.toDataURL('image/png');
          const link = document.createElement('a');
          link.href = pngUrl;
          link.download = `${fig.id}_report.png`;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
        }
        URL.revokeObjectURL(url);
      };
      img.src = url;
    } catch (err) {
      console.error('PNG export failed:', err);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <div className="bg-slate-900 border border-slate-700/80 rounded-2xl max-w-6xl w-full max-h-[92vh] flex flex-col shadow-2xl overflow-hidden">
        
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-950/90">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-cyan-500/20 text-cyan-400 flex items-center justify-center border border-cyan-500/30">
              <BarChart3 className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-white tracking-tight">
                Report Figures & Publication Graphs (8 Charts)
              </h2>
              <p className="text-xs text-slate-400">
                Ready for Word, Google Docs, PowerPoint, and LaTeX • Served from <code className="text-cyan-300">/public/figures/</code> & <code className="text-cyan-300">/docs/figures/</code>
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition cursor-pointer"
            aria-label="Close report figures modal"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body Layout */}
        <div className="flex flex-col md:flex-row flex-1 overflow-hidden">
          
          {/* Left Figure Selector */}
          <aside className="w-full md:w-72 border-b md:border-b-0 md:border-r border-slate-800 bg-slate-950/60 p-3 overflow-y-auto space-y-1.5 shrink-0">
            {REPORT_FIGURES.map((fig) => {
              const isSelected = fig.id === activeFig.id;
              return (
                <button
                  key={fig.id}
                  type="button"
                  onClick={() => setSelectedId(fig.id)}
                  className={`w-full text-left p-2.5 rounded-xl transition cursor-pointer border ${
                    isSelected
                      ? 'bg-cyan-500/15 border-cyan-500/50 text-white'
                      : 'bg-transparent border-transparent text-slate-400 hover:bg-slate-800/60 hover:text-slate-200'
                  }`}
                >
                  <div className="flex items-center justify-between text-[11px] font-mono mb-0.5">
                    <span className={isSelected ? 'text-cyan-400 font-bold' : 'text-slate-500 font-semibold'}>
                      {fig.number}
                    </span>
                    <span className="text-[10px] text-slate-500">{fig.section.split(':')[0]}</span>
                  </div>
                  <div className="text-xs font-semibold leading-snug">{fig.title}</div>
                </button>
              );
            })}
          </aside>

          {/* Right Main Viewer */}
          <div className="flex-1 p-5 sm:p-6 overflow-y-auto space-y-4 bg-slate-900/50">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <span className="text-[11px] font-mono text-cyan-400 uppercase tracking-wider font-bold">
                  {activeFig.section}
                </span>
                <h3 className="text-lg font-bold text-white">
                  {activeFig.number}: {activeFig.title}
                </h3>
              </div>

              <div className="flex items-center gap-2">
                <a
                  href={activeFig.src}
                  download={`${activeFig.id}_report.svg`}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
                >
                  <Download className="w-3.5 h-3.5 text-cyan-400" />
                  Download SVG
                </a>
                <button
                  type="button"
                  onClick={() => handleDownloadPng(activeFig)}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-cyan-500 hover:bg-cyan-400 text-slate-950 transition cursor-pointer"
                >
                  <ImageIcon className="w-3.5 h-3.5" />
                  Download High-Res PNG
                </button>
              </div>
            </div>

            {/* Clean White Publication Graph Canvas */}
            <div className="bg-white rounded-xl p-4 border border-slate-300 shadow-lg flex items-center justify-center">
              <img
                src={activeFig.src}
                alt={activeFig.title}
                className="w-full max-w-3xl h-auto object-contain"
              />
            </div>

            {/* Copyable Report Caption */}
            <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400 font-bold">
                  Report Figure Caption (Copy-Paste)
                </span>
                <button
                  type="button"
                  onClick={() => handleCopyCaption(activeFig)}
                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-[11px] font-medium bg-slate-800 hover:bg-slate-700 text-cyan-300 transition cursor-pointer"
                >
                  {copiedId === activeFig.id ? (
                    <>
                      <Check className="w-3 h-3 text-emerald-400" />
                      <span>Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3 h-3" />
                      <span>Copy Caption</span>
                    </>
                  )}
                </button>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed italic">
                {activeFig.caption}
              </p>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
};
