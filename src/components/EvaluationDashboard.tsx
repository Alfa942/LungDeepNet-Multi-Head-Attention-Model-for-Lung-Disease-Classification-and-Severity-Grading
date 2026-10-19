import React, { useState } from 'react';
import { 
  CONFUSION_MATRIX, 
  CLASSIFICATION_METRICS, 
  MACRO_METRICS, 
  SEVERITY_EVALUATION, 
  ABLATION_STUDY 
} from '../data/benchmarkData';
import { 
  BarChart3, 
  CheckCircle, 
  Layers, 
  TrendingUp, 
  Target, 
  Activity, 
  Sparkles, 
  Zap,
  Info
} from 'lucide-react';

export const EvaluationDashboard: React.FC = () => {
  const [selectedDiseaseAblation, setSelectedDiseaseAblation] = useState<'pnm' | 'fib' | 'emp'>('pnm');

  return (
    <div className="space-y-6">
      
      {/* Header Banner */}
      <div className="bg-slate-900/90 rounded-2xl p-6 border border-slate-800 space-y-4 shadow-xl">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-emerald-500/20 text-emerald-400 rounded-xl">
            <BarChart3 className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white tracking-tight">
              Comprehensive Evaluation on Held-Out Test Cohort & Ablation Study
            </h2>
            <p className="text-xs text-slate-400">
              Evaluated on 1,254 unseen patient radiographs (0% patient overlap) across Classification, Severity Regression, and Localization.
            </p>
          </div>
        </div>

        {/* Macro Summary Metrics */}
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 pt-2 font-mono">
          <div className="p-3 bg-slate-950/70 rounded-xl border border-slate-800/80">
            <span className="text-[10px] text-slate-500 block uppercase">Test Accuracy</span>
            <span className="text-xl font-bold text-white">{(MACRO_METRICS.overallAccuracy * 100).toFixed(1)}%</span>
          </div>
          <div className="p-3 bg-slate-950/70 rounded-xl border border-slate-800/80">
            <span className="text-[10px] text-slate-500 block uppercase">Macro F1-Score</span>
            <span className="text-xl font-bold text-cyan-400">{MACRO_METRICS.macroF1.toFixed(3)}</span>
          </div>
          <div className="p-3 bg-slate-950/70 rounded-xl border border-slate-800/80">
            <span className="text-[10px] text-slate-500 block uppercase">Macro AUROC</span>
            <span className="text-xl font-bold text-purple-400">{MACRO_METRICS.macroAuroc.toFixed(3)}</span>
          </div>
          <div className="p-3 bg-slate-950/70 rounded-xl border border-slate-800/80">
            <span className="text-[10px] text-slate-500 block uppercase">Macro Sensitivity</span>
            <span className="text-xl font-bold text-emerald-400">{(MACRO_METRICS.macroSensitivity * 100).toFixed(1)}%</span>
          </div>
          <div className="p-3 bg-slate-950/70 rounded-xl border border-slate-800/80">
            <span className="text-[10px] text-slate-500 block uppercase">Macro Specificity</span>
            <span className="text-xl font-bold text-blue-400">{(MACRO_METRICS.macroSpecificity * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>

      {/* 4-Way Ablation Study Section (Critical Requirement) */}
      <div className="bg-slate-900/90 rounded-2xl p-6 border border-purple-900/40 space-y-4 shadow-xl">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-purple-400" />
            <h3 className="text-sm font-bold text-white font-mono uppercase tracking-wider">
              Ablation Study: Empirical Validation of Dual-Head Attention
            </h3>
          </div>
          <span className="text-xs font-mono text-purple-300">
            A vs. B vs. C vs. D (LungDeepNet)
          </span>
        </div>

        <div className="overflow-x-auto rounded-xl border border-slate-800/80">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 bg-slate-950">
                <th className="p-3">Architecture Variant</th>
                <th className="p-3 text-right">Accuracy</th>
                <th className="p-3 text-right">Macro F1</th>
                <th className="p-3 text-right">AUROC</th>
                <th className="p-3 text-right">Pneumonia MAE</th>
                <th className="p-3 text-right">Fibrosis MAE</th>
                <th className="p-3 text-right">Emphysema MAE</th>
                <th className="p-3 text-right">Localization IoU</th>
                <th className="p-3 text-right">Inference</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300 bg-slate-950/40">
              {ABLATION_STUDY.map((a) => {
                const isProposed = a.key === 'dual_head';
                return (
                  <tr
                    key={a.key}
                    className={`transition-colors ${
                      isProposed ? 'bg-purple-950/30 hover:bg-purple-950/40 font-semibold' : 'hover:bg-slate-800/30'
                    }`}
                  >
                    <td className="p-3">
                      <div className="font-bold text-white flex items-center gap-2">
                        <span>{a.name}</span>
                        {isProposed && (
                          <span className="px-1.5 py-0.5 text-[9px] bg-purple-500 text-slate-950 font-bold rounded">
                            PROPOSED
                          </span>
                        )}
                      </div>
                      <div className="text-[10px] text-slate-400 font-sans mt-0.5">{a.description}</div>
                    </td>
                    <td className="p-3 text-right text-emerald-400 font-bold">{(a.accuracy * 100).toFixed(1)}%</td>
                    <td className="p-3 text-right text-cyan-300 font-bold">{a.macroF1.toFixed(3)}</td>
                    <td className="p-3 text-right text-purple-300 font-bold">{a.auroc.toFixed(3)}</td>
                    <td className="p-3 text-right text-amber-300">{a.pneumoniaMae.toFixed(3)}</td>
                    <td className="p-3 text-right text-amber-300">{a.fibrosisMae.toFixed(3)}</td>
                    <td className="p-3 text-right text-amber-300">{a.emphysemaMae.toFixed(3)}</td>
                    <td className="p-3 text-right text-cyan-400 font-bold">{a.localizationIoU.toFixed(3)}</td>
                    <td className="p-3 text-right text-slate-400">{a.inferenceTimeMs}ms</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Ablation Key Takeaways Banner */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2 text-xs font-sans">
          <div className="p-3 bg-slate-950/80 rounded-xl border border-slate-800/80 text-slate-300 leading-relaxed">
            <strong className="text-cyan-300 block mb-1 font-mono">Spatial Attention Contribution (Head A):</strong>
            Spatial attention improves disease classification accuracy (+2.6%) and increases lesion localization IoU by +0.162 over the DenseNet121 baseline, guiding the network to specific alveolar and interstitial foci.
          </div>
          <div className="p-3 bg-slate-950/80 rounded-xl border border-slate-800/80 text-slate-300 leading-relaxed">
            <strong className="text-purple-300 block mb-1 font-mono">Dual-Head Synergy (Proposed LungDeepNet):</strong>
            Combining Spatial and Channel Attention simultaneously minimizes severity regression MAE (0.048 for pneumonia) while achieving the highest classification F1 (0.950) with only a modest +4.2ms inference overhead.
          </div>
        </div>
      </div>

      {/* Multiclass Confusion Matrix & Per-Class Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Confusion Matrix (6 Cols) */}
        <div className="lg:col-span-6 bg-slate-900/90 rounded-2xl p-6 border border-slate-800 space-y-4">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-400 font-mono">
            Held-Out Test Set Confusion Matrix (N = 1,254)
          </span>

          <div className="overflow-x-auto">
            <table className="w-full text-center text-xs font-mono">
              <thead>
                <tr className="text-slate-400 bg-slate-950/80">
                  <th className="p-2.5 text-left">Actual \ Pred</th>
                  <th className="p-2.5">Normal</th>
                  <th className="p-2.5">Pneumonia</th>
                  <th className="p-2.5">Emphysema</th>
                  <th className="p-2.5">Fibrosis</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {CONFUSION_MATRIX.map((row) => (
                  <tr key={row.actual} className="hover:bg-slate-800/30">
                    <td className="p-2.5 text-left font-bold text-white bg-slate-950/50">{row.actual}</td>
                    
                    {/* Normal */}
                    <td className={`p-2.5 font-bold ${row.actual === 'Normal' ? 'bg-cyan-950 text-cyan-300 font-extrabold' : 'text-slate-500'}`}>
                      {row.normal}
                    </td>

                    {/* Pneumonia */}
                    <td className={`p-2.5 font-bold ${row.actual === 'Pneumonia' ? 'bg-cyan-950 text-cyan-300 font-extrabold' : 'text-slate-500'}`}>
                      {row.pneumonia}
                    </td>

                    {/* Emphysema */}
                    <td className={`p-2.5 font-bold ${row.actual === 'Emphysema' ? 'bg-cyan-950 text-cyan-300 font-extrabold' : 'text-slate-500'}`}>
                      {row.emphysema}
                    </td>

                    {/* Fibrosis */}
                    <td className={`p-2.5 font-bold ${row.actual === 'Pulmonary Fibrosis' ? 'bg-cyan-950 text-cyan-300 font-extrabold' : 'text-slate-500'}`}>
                      {row.fibrosis}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Class-wise Performance Metrics (6 Cols) */}
        <div className="lg:col-span-6 bg-slate-900/90 rounded-2xl p-6 border border-slate-800 space-y-4">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-400 font-mono">
            Per-Class Diagnostic Metrics
          </span>

          <div className="space-y-3">
            {CLASSIFICATION_METRICS.map((m) => (
              <div key={m.disease} className="p-3 bg-slate-950/60 rounded-xl border border-slate-800 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-bold text-white font-mono">{m.disease}</span>
                  <span className="font-mono text-cyan-400 font-semibold">AUROC: {m.auroc.toFixed(3)}</span>
                </div>
                <div className="grid grid-cols-4 gap-2 text-[11px] font-mono text-center">
                  <div className="p-1.5 bg-slate-900 rounded">
                    <span className="text-slate-500 block text-[9px]">Sensitivity</span>
                    <span className="text-emerald-400 font-bold">{(m.sensitivity * 100).toFixed(1)}%</span>
                  </div>
                  <div className="p-1.5 bg-slate-900 rounded">
                    <span className="text-slate-500 block text-[9px]">Specificity</span>
                    <span className="text-blue-400 font-bold">{(m.specificity * 100).toFixed(1)}%</span>
                  </div>
                  <div className="p-1.5 bg-slate-900 rounded">
                    <span className="text-slate-500 block text-[9px]">Precision</span>
                    <span className="text-purple-400 font-bold">{(m.precision * 100).toFixed(1)}%</span>
                  </div>
                  <div className="p-1.5 bg-slate-900 rounded">
                    <span className="text-slate-500 block text-[9px]">F1-Score</span>
                    <span className="text-cyan-300 font-bold">{m.f1.toFixed(3)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

      </div>

      {/* Disease-Aware Severity Regression Evaluation Table */}
      <div className="bg-slate-900/90 rounded-2xl p-6 border border-slate-800 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-purple-400" />
            <h3 className="text-sm font-bold text-white font-mono uppercase tracking-wider">
              Disease-Aware Severity / Radiographic Burden Regression Evaluation
            </h3>
          </div>
          <span className="text-xs font-mono text-purple-300">
            Evaluated separately per pathology
          </span>
        </div>

        <div className="overflow-x-auto rounded-xl border border-slate-800/80">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 bg-slate-950">
                <th className="p-3">Disease Cohort</th>
                <th className="p-3 text-right">Samples (N)</th>
                <th className="p-3 text-right">MAE</th>
                <th className="p-3 text-right">RMSE</th>
                <th className="p-3 text-right">Pearson r</th>
                <th className="p-3 text-right">Spearman ρ</th>
                <th className="p-3 text-right">R²</th>
                <th className="p-3">Target Burden Source</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300 bg-slate-950/40">
              {SEVERITY_EVALUATION.map((s) => (
                <tr key={s.disease} className="hover:bg-slate-800/30">
                  <td className="p-3 font-bold text-white">{s.disease}</td>
                  <td className="p-3 text-right text-slate-400">{s.cohortSize}</td>
                  <td className="p-3 text-right text-cyan-300 font-bold">{s.mae.toFixed(3)}</td>
                  <td className="p-3 text-right text-emerald-400 font-bold">{s.rmse.toFixed(3)}</td>
                  <td className="p-3 text-right text-purple-300 font-bold">{s.pearsonR !== null ? s.pearsonR.toFixed(3) : '—'}</td>
                  <td className="p-3 text-right text-purple-300">{s.spearmanRho !== null ? s.spearmanRho.toFixed(3) : '—'}</td>
                  <td className="p-3 text-right text-yellow-300">{s.rSquared !== null ? s.rSquared.toFixed(3) : '—'}</td>
                  <td className="p-3 text-[11px] text-slate-400 font-sans">{s.targetSource}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
};
