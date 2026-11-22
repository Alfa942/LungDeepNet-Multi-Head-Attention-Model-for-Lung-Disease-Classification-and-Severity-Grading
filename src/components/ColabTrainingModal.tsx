import React, { useState, useEffect } from 'react';
import { 
  X, 
  Cpu, 
  Download, 
  ExternalLink, 
  CheckCircle2, 
  AlertCircle, 
  Terminal, 
  Zap, 
  Layers, 
  Copy, 
  Check, 
  HardDrive,
  RefreshCw
} from 'lucide-react';

interface ColabTrainingModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ColabTrainingModal: React.FC<ColabTrainingModalProps> = ({ isOpen, onClose }) => {
  const [hasCheckpoint, setHasCheckpoint] = useState<boolean>(false);
  const [checkpointStatus, setCheckpointStatus] = useState<string>('Checking local checkpoint...');
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  const checkLocalStatus = async () => {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        setHasCheckpoint(Boolean(data.hasLocalCheckpoint));
        setCheckpointStatus(data.checkpointStatus || (data.hasLocalCheckpoint ? 'Custom Trained Weights Active' : 'PyTorch Base Model'));
      }
    } catch {
      setCheckpointStatus('Local status check offline');
    }
  };

  useEffect(() => {
    if (isOpen) {
      checkLocalStatus();
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleCopy = (text: string, key: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md animate-in fade-in duration-200">
      <div className="bg-slate-900 border border-slate-800 w-full max-w-3xl rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-950/50">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-cyan-500/15 border border-cyan-500/30 text-cyan-400">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                Google Colab Free GPU Training Suite
                <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                  Tesla T4 GPU (Free)
                </span>
              </h3>
              <p className="text-xs text-slate-400">
                Train on your <code className="text-cyan-300 font-mono">data/</code> folder in ~2-3 minutes with zero local hardware strain
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-6 scrollbar-thin scrollbar-thumb-slate-800">
          
          {/* Quick Action Download Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <a
              href="/public/lungdeepnet_colab_training.ipynb"
              download="lungdeepnet_colab_training.ipynb"
              className="flex flex-col justify-between p-3.5 rounded-xl bg-slate-950/80 border border-cyan-500/40 hover:border-cyan-400 transition hover:bg-cyan-950/20 group"
            >
              <div className="space-y-1">
                <span className="text-[10px] font-mono text-cyan-400 font-bold uppercase tracking-wider block">Step 1</span>
                <span className="text-xs font-bold text-white block group-hover:text-cyan-300">Colab Notebook</span>
                <p className="text-[11px] text-slate-400 leading-snug">Pre-configured .ipynb with T4 mixed-precision pipeline</p>
              </div>
              <div className="mt-3 flex items-center gap-1.5 text-xs font-semibold text-cyan-300">
                <Download className="w-3.5 h-3.5" />
                <span>Download .ipynb</span>
              </div>
            </a>

            <a
              href="/public/lungdeepnet_bundle.zip"
              download="lungdeepnet_bundle.zip"
              className="flex flex-col justify-between p-3.5 rounded-xl bg-slate-950/80 border border-purple-500/40 hover:border-purple-400 transition hover:bg-purple-950/20 group"
            >
              <div className="space-y-1">
                <span className="text-[10px] font-mono text-purple-400 font-bold uppercase tracking-wider block">Step 2</span>
                <span className="text-xs font-bold text-white block group-hover:text-purple-300">Dataset & Code Zip</span>
                <p className="text-[11px] text-slate-400 leading-snug">Bundles 60 images in data/ and all python/ modules</p>
              </div>
              <div className="mt-3 flex items-center gap-1.5 text-xs font-semibold text-purple-300">
                <Download className="w-3.5 h-3.5" />
                <span>Download .zip</span>
              </div>
            </a>

            <a
              href="https://colab.research.google.com/"
              target="_blank"
              rel="noopener noreferrer"
              className="flex flex-col justify-between p-3.5 rounded-xl bg-slate-950/80 border border-amber-500/40 hover:border-amber-400 transition hover:bg-amber-950/20 group"
            >
              <div className="space-y-1">
                <span className="text-[10px] font-mono text-amber-400 font-bold uppercase tracking-wider block">Step 3</span>
                <span className="text-xs font-bold text-white block group-hover:text-amber-300">Open Google Colab</span>
                <p className="text-[11px] text-slate-400 leading-snug">Upload the notebook and set hardware accelerator to T4 GPU</p>
              </div>
              <div className="mt-3 flex items-center gap-1.5 text-xs font-semibold text-amber-300">
                <ExternalLink className="w-3.5 h-3.5" />
                <span>Open Colab in Browser</span>
              </div>
            </a>
          </div>

          {/* Local Checkpoint Status Banner */}
          <div className={`p-4 rounded-xl border flex items-center justify-between ${
            hasCheckpoint 
              ? 'bg-emerald-950/30 border-emerald-500/40 text-emerald-300' 
              : 'bg-amber-950/30 border-amber-500/40 text-amber-300'
          }`}>
            <div className="flex items-center gap-3">
              {hasCheckpoint ? (
                <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0" />
              ) : (
                <AlertCircle className="w-5 h-5 text-amber-400 shrink-0" />
              )}
              <div>
                <span className="text-xs font-bold block text-white">
                  Local Weight Status: {hasCheckpoint ? 'Custom Trained Checkpoint Loaded' : 'Awaiting Colab Training Checkpoint'}
                </span>
                <span className="text-[11px] text-slate-400">
                  {hasCheckpoint 
                    ? 'Found checkpoints/best_model.pth! Local CPU inference is using your custom trained weights.'
                    : 'Target file: checkpoints/best_model.pth (Train in Colab and place the downloaded file here).'}
                </span>
              </div>
            </div>
            <button
              onClick={checkLocalStatus}
              className="p-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 hover:text-white transition"
              title="Refresh checkpoint status"
            >
              <RefreshCw className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Step-by-Step Free Colab Walkthrough */}
          <div className="space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 font-mono flex items-center gap-1.5">
              <Terminal className="w-3.5 h-3.5 text-cyan-400" />
              Free Colab Execution Walkthrough
            </h4>

            <div className="space-y-2 text-xs">
              <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800 space-y-1">
                <span className="font-bold text-slate-200">1. Select T4 GPU in Google Colab (Free Tier)</span>
                <p className="text-slate-400 leading-relaxed text-[11px]">
                  In Colab menu: <span className="font-mono text-cyan-300">Runtime &gt; Change runtime type</span>. Under <strong>Hardware accelerator</strong>, select <strong>T4 GPU</strong>, then click <strong>Save</strong>.
                </p>
              </div>

              <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800 space-y-1">
                <span className="font-bold text-slate-200">2. Ingest Dataset & Python Code</span>
                <p className="text-slate-400 leading-relaxed text-[11px]">
                  Run Cell 3 to upload <span className="font-mono text-purple-300">lungdeepnet_bundle.zip</span>. The notebook automatically unzips all 60 radiographs in <span className="font-mono text-slate-300">data/</span> and prepares the stratified patient-wise DataLoaders.
                </p>
              </div>

              <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800 space-y-1">
                <span className="font-bold text-slate-200">3. Run Two-Phase GPU Mixed-Precision Training</span>
                <p className="text-slate-400 leading-relaxed text-[11px]">
                  Run Cell 6. It trains Phase 1 (5 warmup epochs) and Phase 2 (20 fine-tuning epochs) using <span className="font-mono text-cyan-300">torch.cuda.amp</span> FP16 mixed precision. It completes in <strong>~2-3 minutes</strong> and saves <span className="font-mono text-emerald-300">checkpoints/best_model.pth</span>.
                </p>
              </div>

              <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800 space-y-1">
                <span className="font-bold text-slate-200">4. Download Checkpoint to Local Machine</span>
                <p className="text-slate-400 leading-relaxed text-[11px]">
                  Run Cell 8. Colab triggers a browser download for <span className="font-mono text-emerald-300">best_model.pth</span> (~30 MB). Move this file into your project folder at:
                </p>
                <div className="flex items-center justify-between p-2 bg-slate-900 rounded-lg font-mono text-[11px] text-cyan-300 mt-1">
                  <span>LungDeepNet-Multi-Head-Attention.../checkpoints/best_model.pth</span>
                  <button
                    onClick={() => handleCopy('checkpoints/best_model.pth', 'chk')}
                    className="flex items-center gap-1 text-[10px] text-slate-400 hover:text-white"
                  >
                    {copiedKey === 'chk' ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                    <span>{copiedKey === 'chk' ? 'Copied' : 'Copy'}</span>
                  </button>
                </div>
              </div>
            </div>
          </div>

        </div>

        {/* Footer */}
        <div className="px-6 py-3.5 border-t border-slate-800 bg-slate-950/50 flex items-center justify-between text-xs">
          <div className="flex items-center gap-1.5 text-slate-400">
            <Zap className="w-3.5 h-3.5 text-cyan-400" />
            <span>100% Offline Local Model Architecture • Zero API Keys</span>
          </div>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold transition"
          >
            Done
          </button>
        </div>

      </div>
    </div>
  );
};
