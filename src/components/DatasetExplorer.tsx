import React, { useState } from 'react';
import { 
  DATASET_SUMMARIES, 
  TOTAL_DATASET_STATS, 
  DATASET_40K_STATS,
  DATASET_40K_SUMMARIES,
  LEAKAGE_AUDIT_CHECKLIST 
} from '../data/benchmarkData';
import { 
  Database, 
  ShieldCheck, 
  Users, 
  FileSpreadsheet, 
  Search, 
  Filter, 
  CheckCircle2, 
  AlertCircle, 
  Download,
  UploadCloud,
  Terminal,
  FolderTree,
  Cpu,
  Layers,
  ArrowRight,
  BookOpen
} from 'lucide-react';

export const DatasetExplorer: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedCohort, setSelectedCohort] = useState<string>('ALL');
  const [cohortScale, setCohortScale] = useState<'10k' | '40k'>('40k');

  const currentStats = cohortScale === '40k' ? DATASET_40K_STATS : TOTAL_DATASET_STATS;
  const currentSummaries = cohortScale === '40k' ? DATASET_40K_SUMMARIES : DATASET_SUMMARIES;

  // Sample rows from unified_metadata.csv for interactive inspection
  const sampleMetadataRows = [
    { image_path: 'data/pneumonia/PT_PNM_4829_01.dcm', patient_id: 'PT-PNM-4829', disease: 'Pneumonia', dataset: 'RSNA_Pneumonia', split: 'test', severity_score: 0.72, severity_available: 'True' },
    { image_path: 'data/pneumonia/PT_PNM_4830_01.png', patient_id: 'PT-PNM-4830', disease: 'Pneumonia', dataset: 'NIH_ChestXray14', split: 'train', severity_score: 0.54, severity_available: 'True' },
    { image_path: 'data/fibrosis/PT_FIB_7104_01.dcm', patient_id: 'PT-FIB-7104', disease: 'Pulmonary Fibrosis', dataset: 'PadChest_Fibrosis', split: 'test', severity_score: 0.68, severity_available: 'True' },
    { image_path: 'data/fibrosis/PT_FIB_7105_01.png', patient_id: 'PT-FIB-7105', disease: 'Pulmonary Fibrosis', dataset: 'NIH_ChestXray14', split: 'train', severity_score: 0.81, severity_available: 'True' },
    { image_path: 'data/emphysema/PT_EMP_9351_01.dcm', patient_id: 'PT-EMP-9351', disease: 'Emphysema', dataset: 'CheXpert_Emphysema', split: 'test', severity_score: 0.64, severity_available: 'True' },
    { image_path: 'data/emphysema/PT_EMP_9352_01.png', patient_id: 'PT-EMP-9352', disease: 'Emphysema', dataset: 'NIH_ChestXray14', split: 'train', severity_score: 0.42, severity_available: 'True' },
    { image_path: 'data/normal/PT_NRM_1092_01.dcm', patient_id: 'PT-NRM-1092', disease: 'Normal', dataset: 'NIH_ControlCohort', split: 'test', severity_score: 'NaN', severity_available: 'False' },
    { image_path: 'data/normal/PT_NRM_1093_01.png', patient_id: 'PT-NRM-1093', disease: 'Normal', dataset: 'RSNA_NormalCohort', split: 'val', severity_score: 'NaN', severity_available: 'False' },
    { image_path: 'data/pneumonia/PT_PNM_4831_01.png', patient_id: 'PT-PNM-4831', disease: 'Pneumonia', dataset: 'NIH_ChestXray14', split: 'val', severity_score: 0.38, severity_available: 'True' },
    { image_path: 'data/fibrosis/PT_FIB_7106_01.png', patient_id: 'PT-FIB-7106', disease: 'Pulmonary Fibrosis', dataset: 'PadChest_Fibrosis', split: 'val', severity_score: 0.49, severity_available: 'True' },
  ];

  const filteredRows = sampleMetadataRows.filter((r) => {
    const matchesSearch = 
      r.patient_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      r.disease.toLowerCase().includes(searchQuery.toLowerCase()) ||
      r.image_path.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCohort = selectedCohort === 'ALL' || r.disease === selectedCohort;
    return matchesSearch && matchesCohort;
  });

  return (
    <div className="space-y-6">
      
      {/* Header Overview */}
      <div className="bg-slate-900/90 rounded-2xl p-6 border border-slate-800 space-y-4 shadow-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-cyan-500/20 text-cyan-400 rounded-xl">
              <Database className="w-6 h-6" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white tracking-tight">
                Dataset Inspection, Harmonization & Leakage Audit
              </h2>
              <p className="text-xs text-slate-400">
                {cohortScale === '40k' 
                  ? '40,000 Verified Radiographs: Exactly 10,000 per class (Normal, Pneumonia, Emphysema, Pulmonary Fibrosis).' 
                  : 'Harmonized across 4 cohorts: Normal, Pneumonia, Emphysema, and Pulmonary Fibrosis (10,025 images).'}
              </p>
            </div>
          </div>

          {/* Scale Toggle */}
          <div className="flex items-center bg-slate-950 p-1 rounded-xl border border-slate-800 shrink-0 font-mono text-xs">
            <button
              onClick={() => setCohortScale('40k')}
              className={`px-3 py-1.5 rounded-lg font-bold transition flex items-center gap-1.5 ${
                cohortScale === '40k'
                  ? 'bg-cyan-500 text-slate-950 shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <span>40k Scaled Cohort</span>
              <span className="text-[10px] px-1.5 py-0.2 rounded bg-slate-900 text-cyan-300">10k/class</span>
            </button>
            <button
              onClick={() => setCohortScale('10k')}
              className={`px-3 py-1.5 rounded-lg font-bold transition flex items-center gap-1.5 ${
                cohortScale === '10k'
                  ? 'bg-cyan-500 text-slate-950 shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <span>10k Benchmark</span>
            </button>
          </div>
        </div>

        {/* Top Metric Cards */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2 font-mono">
          <div className="p-3 bg-slate-950/70 rounded-xl border border-slate-800/80">
            <span className="text-[10px] text-slate-500 block uppercase">Total Images</span>
            <span className="text-xl font-bold text-white">{currentStats.totalImages.toLocaleString()}</span>
          </div>
          <div className="p-3 bg-slate-950/70 rounded-xl border border-slate-800/80">
            <span className="text-[10px] text-slate-500 block uppercase">Unique Patients</span>
            <span className="text-xl font-bold text-cyan-400">{currentStats.totalUniquePatients.toLocaleString()}</span>
          </div>
          <div className="p-3 bg-slate-950/70 rounded-xl border border-slate-800/80">
            <span className="text-[10px] text-slate-500 block uppercase">Patient Leakage</span>
            <span className="text-xl font-bold text-emerald-400">0.00% (Strict)</span>
          </div>
          <div className="p-3 bg-slate-950/70 rounded-xl border border-slate-800/80">
            <span className="text-[10px] text-slate-500 block uppercase">Split Ratio</span>
            <span className="text-xl font-bold text-purple-400">75 / 12.5 / 12.5</span>
          </div>
        </div>
      </div>

      {/* Multi-Dataset Anti-Bias Protocol Banner */}
      <div className="bg-gradient-to-r from-cyan-950/40 via-blue-950/30 to-slate-900 border border-cyan-800/40 rounded-2xl p-5 space-y-2">
        <div className="flex items-center gap-2">
          <span className="px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider bg-cyan-500 text-slate-950 rounded font-mono">
            Anti-Shortcut Protocol
          </span>
          <span className="text-xs font-bold text-cyan-200 font-mono">
            Different Clinical Dataset Selected for Each Disease Entity
          </span>
        </div>
        <p className="text-xs text-slate-300 leading-relaxed font-sans">
          To prevent <strong>institutional shortcut learning</strong> (where models memorize scanner brand artifacts, hospital font tags, or DICOM windowing rather than lung pathology), this project purposefully draws each disease from a distinct, clinically verified international cohort.
        </p>
      </div>

      {/* 4 Source Cohorts Table */}
      <div className="bg-slate-900/90 rounded-2xl p-6 border border-slate-800 space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-400 font-mono">
            Four Target Disease Cohorts & Multi-Institutional Provenance
          </span>
          <span className="text-[11px] font-mono text-cyan-400">
            4 Independent Repositories Harmonized
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 bg-slate-950/80">
                <th className="p-3">Disease Cohort</th>
                <th className="p-3">Source Dataset & Institution</th>
                <th className="p-3 text-right">Images</th>
                <th className="p-3 text-right">Patients</th>
                <th className="p-3 text-right">Train</th>
                <th className="p-3 text-right">Val</th>
                <th className="p-3 text-right">Test</th>
                <th className="p-3">Clinical Severity Basis</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300">
              {currentSummaries.map((c) => (
                <tr key={c.disease} className="hover:bg-slate-800/30 transition-colors">
                  <td className="p-3 font-bold text-white whitespace-nowrap">
                    {c.disease}
                    <div className="text-[10px] text-cyan-400 font-mono font-normal">{c.path}</div>
                  </td>
                  <td className="p-3">
                    <span className="font-semibold text-slate-200 block">{c.sourceCohort}</span>
                    <span className="text-[11px] text-slate-400 font-sans">{c.institution}</span>
                  </td>
                  <td className="p-3 text-right text-white font-bold">{c.totalImages}</td>
                  <td className="p-3 text-right text-slate-400">{c.uniquePatients}</td>
                  <td className="p-3 text-right text-emerald-400">{c.trainCount}</td>
                  <td className="p-3 text-right text-purple-400">{c.valCount}</td>
                  <td className="p-3 text-right text-blue-400 font-bold">{c.testCount}</td>
                  <td className="p-3 text-[11px] text-slate-400 font-sans max-w-xs">{c.severityFormat}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Data Leakage Prevention Checklist */}
      <div className="bg-slate-900/90 rounded-2xl p-6 border border-slate-800 space-y-4">
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-emerald-400" />
          <h3 className="text-sm font-bold text-white font-mono uppercase tracking-wider">
            Scientific Rigor & Data Leakage Prevention Audit
          </h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {LEAKAGE_AUDIT_CHECKLIST.map((item, idx) => (
            <div key={idx} className="p-3.5 bg-slate-950/70 rounded-xl border border-slate-800/80 space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-200">{item.item}</span>
                <span className="px-2 py-0.5 text-[10px] font-mono font-bold bg-emerald-950 text-emerald-300 border border-emerald-800 rounded">
                  {item.status}
                </span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed font-sans">{item.detail}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Unified Metadata CSV Viewer */}
      <div className="bg-slate-900/90 rounded-2xl p-6 border border-slate-800 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <FileSpreadsheet className="w-5 h-5 text-cyan-400" />
            <h3 className="text-sm font-bold text-white font-mono">
              Unified Metadata Schema (<code className="text-cyan-300">unified_metadata.csv</code>)
            </h3>
          </div>

          <div className="flex items-center gap-2">
            <div className="relative">
              <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-slate-500" />
              <input
                type="text"
                placeholder="Search patient, disease..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-8 pr-3 py-1 text-xs bg-slate-950 border border-slate-800 rounded-lg text-slate-200 focus:outline-none focus:border-cyan-500 font-mono w-48"
              />
            </div>
            <select
              value={selectedCohort}
              onChange={(e) => setSelectedCohort(e.target.value)}
              className="text-xs bg-slate-950 border border-slate-800 rounded-lg px-2 py-1 text-slate-300 focus:outline-none focus:border-cyan-500 font-mono"
            >
              <option value="ALL">All Classes</option>
              <option value="Normal">Normal</option>
              <option value="Pneumonia">Pneumonia</option>
              <option value="Emphysema">Emphysema</option>
              <option value="Pulmonary Fibrosis">Pulmonary Fibrosis</option>
            </select>
          </div>
        </div>

        <div className="overflow-x-auto rounded-xl border border-slate-800/80">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 bg-slate-950">
                <th className="p-2.5">image_path</th>
                <th className="p-2.5">patient_id</th>
                <th className="p-2.5">disease</th>
                <th className="p-2.5">dataset</th>
                <th className="p-2.5">split</th>
                <th className="p-2.5 text-right">severity_score</th>
                <th className="p-2.5">severity_available</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50 text-slate-300 bg-slate-950/40">
              {filteredRows.map((r, i) => (
                <tr key={i} className="hover:bg-slate-800/30 transition-colors">
                  <td className="p-2.5 text-slate-400">{r.image_path}</td>
                  <td className="p-2.5 text-cyan-300 font-bold">{r.patient_id}</td>
                  <td className="p-2.5 text-white">{r.disease}</td>
                  <td className="p-2.5 text-slate-400">{r.dataset}</td>
                  <td className="p-2.5">
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        r.split === 'train'
                          ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                          : r.split === 'val'
                          ? 'bg-purple-950 text-purple-300 border border-purple-800'
                          : 'bg-blue-950 text-blue-300 border border-blue-800'
                      }`}
                    >
                      {r.split}
                    </span>
                  </td>
                  <td className="p-2.5 text-right text-purple-300 font-semibold">{r.severity_score}</td>
                  <td className="p-2.5 text-slate-400">{r.severity_available}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Dataset Upload & Training Pipeline Guide */}
      <div className="bg-slate-900/90 rounded-2xl p-6 border border-slate-800 space-y-5 shadow-xl">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-500/20 text-purple-400 rounded-xl">
            <UploadCloud className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white tracking-tight font-mono uppercase">
              How to Upload Your Own Dataset & Run Model Training
            </h3>
            <p className="text-xs text-slate-400">
              Follow this 4-step workflow to feed your DICOM / PNG / JPEG cohort into LungDeepNet on Google Colab or your GPU server.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          {/* Step 1: Directory Structure */}
          <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-2.5">
            <div className="flex items-center gap-2 text-cyan-400 font-mono font-bold">
              <FolderTree className="w-4 h-4" />
              <span>Step 1: Organize Your Dataset Directory</span>
            </div>
            <p className="text-slate-300 leading-relaxed font-sans">
              Structure your image archive with patient IDs embedded in file names (e.g. <code className="text-cyan-300 font-mono">PT102_01.png</code>) or organize into subfolders:
            </p>
            <div className="p-2.5 bg-slate-900 rounded-lg font-mono text-[11px] text-slate-300 leading-snug border border-slate-800 overflow-x-auto">
              <div>my_dataset/</div>
              <div>├── Normal/               <span className="text-slate-500"># Clear lung controls</span></div>
              <div>├── Pneumonia/            <span className="text-slate-500"># Alveolar opacities</span></div>
              <div>├── Emphysema/            <span className="text-slate-500"># Hyperlucency / COPD</span></div>
              <div>├── Pulmonary_Fibrosis/   <span className="text-slate-500"># Reticular ILD patterns</span></div>
              <div>└── metadata.csv          <span className="text-purple-300"># Optional labels & patient_id</span></div>
            </div>
          </div>

          {/* Step 2: Upload to Google Colab / Cloud */}
          <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-2.5">
            <div className="flex items-center gap-2 text-purple-400 font-mono font-bold">
              <UploadCloud className="w-4 h-4" />
              <span>Step 2: Upload Archive to Colab or Google Drive</span>
            </div>
            <p className="text-slate-300 leading-relaxed font-sans">
              Zip your cohort and mount it in Google Colab or Kaggle:
            </p>
            <div className="p-2.5 bg-slate-900 rounded-lg font-mono text-[11px] text-cyan-300 leading-snug border border-slate-800 overflow-x-auto space-y-1">
              <div><span className="text-slate-500"># In Colab Cell:</span></div>
              <div>from google.colab import drive</div>
              <div>drive.mount('/content/drive')</div>
              <div>!unzip /content/drive/MyDrive/my_dataset.zip -d /content/data/</div>
            </div>
          </div>

          {/* Step 3: Zero-Leakage Splitting & Inverse Class Weighting */}
          <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-2.5">
            <div className="flex items-center gap-2 text-emerald-400 font-mono font-bold">
              <Layers className="w-4 h-4" />
              <span>Step 3: Harmonize & Prevent Pneumonia Bias</span>
            </div>
            <p className="text-slate-300 leading-relaxed font-sans">
              Run <code className="text-emerald-300 font-mono">notebooks/02_dataset_harmonization_and_splitting.ipynb</code> to generate a zero-leakage patient split and compute train-only class weights:
            </p>
            <div className="p-2.5 bg-slate-900 rounded-lg font-mono text-[11px] text-emerald-300 leading-snug border border-slate-800 overflow-x-auto space-y-1">
              <div><span className="text-slate-500"># Automatically balances uneven cohort classes:</span></div>
              <div>weights = calculate_train_class_weights(train_df)</div>
              <div>criterion_cls = nn.CrossEntropyLoss(weight=weights)</div>
            </div>
          </div>

          {/* Step 4: Two-Phase Training Execution */}
          <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-2.5">
            <div className="flex items-center gap-2 text-amber-400 font-mono font-bold">
              <Terminal className="w-4 h-4" />
              <span>Step 4: Launch Two-Phase PyTorch Training</span>
            </div>
            <p className="text-slate-300 leading-relaxed font-sans">
              Execute Phase 1 (Heads Warmup) followed by Phase 2 (DenseBlock4 Fine-Tuning):
            </p>
            <div className="p-2.5 bg-slate-900 rounded-lg font-mono text-[11px] text-amber-300 leading-snug border border-slate-800 overflow-x-auto space-y-1">
              <div>python src/train.py \</div>
              <div>  --data_dir /content/data \</div>
              <div>  --epochs_phase1 5 \</div>
              <div>  --epochs_phase2 20 \</div>
              <div>  --batch_size 32 --lr 1e-4</div>
            </div>
          </div>
        </div>
      </div>

    </div>
  );
};
