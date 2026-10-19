import React, { useState } from 'react';
import { 
  Layers, 
  Eye, 
  Activity, 
  Cpu, 
  Zap, 
  CheckCircle, 
  ArrowRight, 
  Sliders, 
  Maximize2,
  Box
} from 'lucide-react';

export const ArchitectureViewer: React.FC = () => {
  const [selectedBlock, setSelectedBlock] = useState<string>('dual_head');

  return (
    <div className="space-y-6">
      
      {/* Overview Header */}
      <div className="bg-slate-900/90 rounded-2xl p-6 border border-slate-800 space-y-3 shadow-xl">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-500/20 text-purple-400 rounded-xl">
            <Layers className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white tracking-tight">
              LungDeepNet Multi-Task Architecture Workbench
            </h2>
            <p className="text-xs text-slate-400">
              DenseNet121 Convolutional Backbone + Dual-Head (Spatial & Channel) Attention + Multi-Task Decoupled Heads
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-2 font-mono text-xs">
          <div className="p-3 bg-slate-950/70 rounded-xl border border-slate-800/80">
            <span className="text-[10px] text-slate-500 block uppercase">Objective 1: Shared Backbone</span>
            <span className="text-cyan-300 font-bold">DenseNet121 (DenseBlocks 1–4)</span>
            <p className="text-[11px] text-slate-400 font-sans mt-1">High-dimensional shared feature extractor (7×7×1024 tensor).</p>
          </div>
          <div className="p-3 bg-slate-950/70 rounded-xl border border-slate-800/80">
            <span className="text-[10px] text-slate-500 block uppercase">Objective 2: Dual-Head Attention</span>
            <span className="text-purple-300 font-bold">Spatial (Ms) + Channel (Mc)</span>
            <p className="text-[11px] text-slate-400 font-sans mt-1">Decouples 'Where' (localization) from 'Which' (severity features).</p>
          </div>
          <div className="p-3 bg-slate-950/70 rounded-xl border border-slate-800/80">
            <span className="text-[10px] text-slate-500 block uppercase">Multi-Task Learning</span>
            <span className="text-emerald-300 font-bold">Classification + Masked Severity</span>
            <p className="text-[11px] text-slate-400 font-sans mt-1">Joint optimization with hyperparameter λ = 0.4.</p>
          </div>
        </div>
      </div>

      {/* Interactive Layer Flow Diagram */}
      <div className="bg-slate-900/90 rounded-2xl p-6 border border-slate-800 space-y-6">
        <div className="flex items-center justify-between">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-400 font-mono">
            Interactive Network Dataflow (Click any block for technical specification)
          </span>
          <span className="text-xs font-mono text-cyan-400">Total Params: ~7.35M • FP16 AMP Compatible</span>
        </div>

        <div className="space-y-4">
          
          {/* Stage 1: Input & Backbone */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
            
            {/* Input Node */}
            <button
              onClick={() => setSelectedBlock('input')}
              className={`p-4 rounded-xl text-left border transition-all ${
                selectedBlock === 'input'
                  ? 'bg-cyan-950/80 border-cyan-500 ring-1 ring-cyan-500'
                  : 'bg-slate-950/70 border-slate-800 hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between text-xs font-mono text-slate-400 mb-1">
                <span>Input Radiograph</span>
                <Box className="w-3.5 h-3.5 text-cyan-400" />
              </div>
              <div className="text-sm font-bold text-white">Chest X-Ray</div>
              <div className="text-[11px] font-mono text-cyan-400 mt-1">Shape: (B, 3, 224, 224)</div>
              <p className="text-[10px] text-slate-400 mt-1 font-sans">PA View, ImageNet normalized</p>
            </button>

            {/* DenseNet Conv + DB1-2 */}
            <button
              onClick={() => setSelectedBlock('densenet_early')}
              className={`p-4 rounded-xl text-left border transition-all ${
                selectedBlock === 'densenet_early'
                  ? 'bg-cyan-950/80 border-cyan-500 ring-1 ring-cyan-500'
                  : 'bg-slate-950/70 border-slate-800 hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between text-xs font-mono text-slate-400 mb-1">
                <span>Backbone Early</span>
                <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              </div>
              <div className="text-sm font-bold text-white">7x7 Conv + DB 1 & 2</div>
              <div className="text-[11px] font-mono text-cyan-400 mt-1">Shape: (B, 256, 28, 28)</div>
              <p className="text-[10px] text-slate-400 mt-1 font-sans">Low-level edge & texture filters (Frozen)</p>
            </button>

            {/* DenseNet DB3-4 */}
            <button
              onClick={() => setSelectedBlock('densenet_late')}
              className={`p-4 rounded-xl text-left border transition-all ${
                selectedBlock === 'densenet_late'
                  ? 'bg-cyan-950/80 border-cyan-500 ring-1 ring-cyan-500'
                  : 'bg-slate-950/70 border-slate-800 hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between text-xs font-mono text-slate-400 mb-1">
                <span>Backbone Late</span>
                <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              </div>
              <div className="text-sm font-bold text-white">DenseBlock 3 & 4</div>
              <div className="text-[11px] font-mono text-cyan-400 mt-1">Shape: (B, 1024, 7, 7)</div>
              <p className="text-[10px] text-slate-400 mt-1 font-sans">High-dimensional shared latent representation</p>
            </button>

            {/* Shared Representation Hub */}
            <button
              onClick={() => setSelectedBlock('shared_hub')}
              className={`p-4 rounded-xl text-left border transition-all ${
                selectedBlock === 'shared_hub'
                  ? 'bg-cyan-950/80 border-cyan-500 ring-1 ring-cyan-500'
                  : 'bg-slate-950/70 border-slate-800 hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between text-xs font-mono text-slate-400 mb-1">
                <span>Shared Latent Hub</span>
                <Zap className="w-3.5 h-3.5 text-yellow-400" />
              </div>
              <div className="text-sm font-bold text-yellow-300">Latent Tensor F</div>
              <div className="text-[11px] font-mono text-yellow-400 mt-1">Shape: (B, 1024, 7, 7)</div>
              <p className="text-[10px] text-slate-400 mt-1 font-sans">Bifurcates to Spatial & Channel heads</p>
            </button>

          </div>

          {/* Branching: Dual-Head Attention Stage */}
          <div className="p-4 bg-slate-950/80 rounded-2xl border border-purple-900/50 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold font-mono text-purple-300 flex items-center gap-2">
                <Layers className="w-4 h-4 text-purple-400" />
                <span>Dual-Head Attention Decoupling Stage</span>
              </span>
              <span className="text-[11px] font-mono text-purple-400">Parallel Gating Pathways</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              
              {/* Head A: Spatial Attention */}
              <button
                onClick={() => setSelectedBlock('spatial_head')}
                className={`p-4 rounded-xl text-left border transition-all ${
                  selectedBlock === 'spatial_head'
                    ? 'bg-purple-950/80 border-purple-400 ring-1 ring-purple-400'
                    : 'bg-slate-900/90 border-slate-800 hover:border-purple-800'
                }`}
              >
                <div className="flex items-center justify-between text-xs font-mono text-purple-300 mb-1">
                  <span className="font-bold">Branch A: Spatial Attention Head (Ms)</span>
                  <Eye className="w-4 h-4 text-purple-400" />
                </div>
                <div className="text-sm font-bold text-white">WHERE to Look (Pathology Localization)</div>
                <div className="text-xs font-mono text-purple-400 mt-1">
                  MaxPool_c + AvgPool_c → 7×7 Conv → Sigmoid → (B, 1, 7, 7)
                </div>
                <p className="text-xs text-slate-300 mt-2 leading-relaxed font-sans">
                  Produces spatial gating matrix F_spatial = F ⊙ Ms(F). Emphasizes abnormal parenchymal consolidation, infiltrates, and hyperlucency zones.
                </p>
              </button>

              {/* Head B: Channel Attention */}
              <button
                onClick={() => setSelectedBlock('channel_head')}
                className={`p-4 rounded-xl text-left border transition-all ${
                  selectedBlock === 'channel_head'
                    ? 'bg-purple-950/80 border-purple-400 ring-1 ring-purple-400'
                    : 'bg-slate-900/90 border-slate-800 hover:border-purple-800'
                }`}
              >
                <div className="flex items-center justify-between text-xs font-mono text-purple-300 mb-1">
                  <span className="font-bold">Branch B: Channel Attention Head (Mc)</span>
                  <Sliders className="w-4 h-4 text-purple-400" />
                </div>
                <div className="text-sm font-bold text-white">WHICH Channels (Severity Representation)</div>
                <div className="text-xs font-mono text-purple-400 mt-1">
                  GAP + GMP → MLP (r=16) → Sigmoid → (B, 1024, 1, 1)
                </div>
                <p className="text-xs text-slate-300 mt-2 leading-relaxed font-sans">
                  Produces channel gating vector F_channel = F ⊙ Mc(F). Selectively amplifies disease-specific burden features (opacity density vs. reticular lattice).
                </p>
              </button>

            </div>
          </div>

          {/* Multi-Task Prediction Heads */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            
            {/* Classification Head */}
            <button
              onClick={() => setSelectedBlock('cls_head')}
              className={`p-4 rounded-xl text-left border transition-all ${
                selectedBlock === 'cls_head'
                  ? 'bg-emerald-950/80 border-emerald-400 ring-1 ring-emerald-400'
                  : 'bg-slate-950/70 border-slate-800 hover:border-emerald-800'
              }`}
            >
              <div className="flex items-center justify-between text-xs font-mono text-emerald-300 mb-1">
                <span className="font-bold">Task 1: Disease Classification Head</span>
                <CheckCircle className="w-4 h-4 text-emerald-400" />
              </div>
              <div className="text-sm font-bold text-white">4 Target Disease Classes</div>
              <div className="text-xs font-mono text-emerald-400 mt-1">
                GAP → FC(1024, 512) → BN → ReLU → Drop(0.25) → FC(512, 4)
              </div>
              <p className="text-xs text-slate-300 mt-2 font-sans">
                Loss: L_cls = CrossEntropy(y, y_hat) for Normal, Pneumonia, Emphysema, Fibrosis.
              </p>
            </button>

            {/* Severity Regression Head */}
            <button
              onClick={() => setSelectedBlock('sev_head')}
              className={`p-4 rounded-xl text-left border transition-all ${
                selectedBlock === 'sev_head'
                  ? 'bg-blue-950/80 border-blue-400 ring-1 ring-blue-400'
                  : 'bg-slate-950/70 border-slate-800 hover:border-blue-800'
              }`}
            >
              <div className="flex items-center justify-between text-xs font-mono text-blue-300 mb-1">
                <span className="font-bold">Task 2: Radiographic Severity Head</span>
                <Activity className="w-4 h-4 text-blue-400" />
              </div>
              <div className="text-sm font-bold text-white">Continuous Burden Score [0.0, 1.0]</div>
              <div className="text-xs font-mono text-blue-400 mt-1">
                GAP → FC(1024, 256) → BN → ReLU → Drop(0.25) → FC(256, 1) → Sigmoid
              </div>
              <p className="text-xs text-slate-300 mt-2 font-sans">
                Loss: L_sev = MaskedSmoothL1(s, s_hat) (Masked for Normal class).
              </p>
            </button>

          </div>

        </div>

        {/* Selected Block Deep-Dive Panel */}
        <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-2">
          <div className="flex items-center gap-2 text-xs font-mono text-cyan-400 font-bold">
            <span>Mathematical Formulation & Module Details:</span>
            <span className="text-slate-300 uppercase font-sans">[{selectedBlock}]</span>
          </div>

          <div className="text-xs text-slate-300 leading-relaxed font-mono bg-slate-900/90 p-3 rounded-lg border border-slate-800/80 overflow-x-auto">
            {selectedBlock === 'spatial_head' && (
              <div>
                <p className="text-purple-300 font-bold mb-1">Spatial Attention Map Formulation:</p>
                <code>
                  {"M_s(F) = \\sigma(f^{7\\times7}([AvgPool_c(F); MaxPool_c(F)]))"}
                </code>
                <p className="text-slate-400 mt-2 font-sans">
                  The spatial attention head aggregates channel information by applying average and max pooling operations along the channel axis, concatenating them into a 2-channel spatial descriptor, and filtering through a 7×7 convolutional layer. This produces an activation mask highlighting anatomical coordinates of pathological interest.
                </p>
              </div>
            )}

            {selectedBlock === 'channel_head' && (
              <div>
                <p className="text-purple-300 font-bold mb-1">Channel Attention Formulation (Squeeze-and-Excitation):</p>
                <code>
                  {"M_c(F) = \\sigma(W_1 \\cdot \\text{ReLU}(W_0 \\cdot \\text{AvgPool}_s(F)) + W_1 \\cdot \\text{ReLU}(W_0 \\cdot \\text{MaxPool}_s(F)))"}
                </code>
                <p className="text-slate-400 mt-2 font-sans">
                  {"With reduction ratio r=16, W_0 in R^{64 x 1024} and W_1 in R^{1024 x 64}. It captures non-linear inter-channel interactions, dynamically weighting feature descriptors that quantify lesion density and structural destruction."}
                </p>
              </div>
            )}

            {selectedBlock === 'cls_head' && (
              <div>
                <p className="text-emerald-300 font-bold mb-1">Classification Objective:</p>
                <code>
                  {"\\mathcal{L}_{cls} = -\\sum_{c=1}^4 y_c \\log (e^{z_c} / \\sum_{j=1}^4 e^{z_j})"}
                </code>
                <p className="text-slate-400 mt-2 font-sans">
                  Evaluates multiclass posterior probabilities for Normal (0), Pneumonia (1), Emphysema (2), and Pulmonary Fibrosis (3).
                </p>
              </div>
            )}

            {selectedBlock === 'sev_head' && (
              <div>
                <p className="text-blue-300 font-bold mb-1">Masked Severity Regression Objective:</p>
                <code>
                  {"\\mathcal{L}_{sev} = \\frac{1}{\\sum m_i} \\sum_{i=1}^B m_i \\cdot \\text{SmoothL1}(s_i, \\hat{s}_i, \\beta=0.05)"}
                </code>
                <p className="text-slate-400 mt-2 font-sans">
                  {"Where m_i in {0, 1} is a binary mask that is 0 for Normal physiological samples or images without verified ground-truth severity, preventing false gradient bias."}
                </p>
              </div>
            )}

            {(selectedBlock === 'input' || selectedBlock === 'densenet_early' || selectedBlock === 'densenet_late' || selectedBlock === 'shared_hub' || selectedBlock === 'dual_head') && (
              <div>
                <p className="text-cyan-300 font-bold mb-1">Multi-Task Joint Loss Formulation:</p>
                <code>
                  {"\\mathcal{L}_{total} = \\mathcal{L}_{cls} + \\lambda \\cdot \\mathcal{L}_{sev}, \\quad \\text{where } \\lambda = 0.4 \\text{ (optimized via Optuna)}"}
                </code>
                <p className="text-slate-400 mt-2 font-sans">
                  DenseNet121 features are shared across both task heads, encouraging the shared representation to learn robust pulmonary representations that simultaneously isolate pathological regions and quantify their radiographic burden.
                </p>
              </div>
            )}
          </div>
        </div>

      </div>

    </div>
  );
};
