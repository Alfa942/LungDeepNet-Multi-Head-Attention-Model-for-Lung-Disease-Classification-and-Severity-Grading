import React, { useState } from 'react';
import { JUPYTER_NOTEBOOKS } from '../data/notebooksData';
import { SOURCE_CODE_FILES } from '../data/sourceCodeData';
import { JupyterNotebook, PythonSourceFile } from '../types';
import { 
  FileCode2, 
  BookOpen, 
  Play, 
  Copy, 
  Check, 
  Download, 
  Folder, 
  FileText, 
  Terminal, 
  CheckCircle,
  ExternalLink
} from 'lucide-react';

interface NotebooksAndCodeHubProps {
  onDownloadZip: () => void;
  isDownloading?: boolean;
}

export const NotebooksAndCodeHub: React.FC<NotebooksAndCodeHubProps> = ({
  onDownloadZip,
  isDownloading = false,
}) => {
  const [activeTab, setActiveTab] = useState<'notebooks' | 'source_code'>('notebooks');
  const [selectedNotebook, setSelectedNotebook] = useState<JupyterNotebook>(JUPYTER_NOTEBOOKS[0]);
  const [selectedSourceFile, setSelectedSourceFile] = useState<PythonSourceFile>(SOURCE_CODE_FILES[0]);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  const handleCopyCode = (text: string, key: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  return (
    <div className="space-y-6">
      
      {/* Header Banner */}
      <div className="bg-slate-900/90 rounded-2xl p-6 border border-slate-800 space-y-4 shadow-xl">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-cyan-500/20 text-cyan-400 rounded-xl">
              <FileCode2 className="w-6 h-6" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white tracking-tight">
                Google Colab Research Notebooks & Modular Python Codebase
              </h2>
              <p className="text-xs text-slate-400">
                10 End-to-End Jupyter Notebooks + Modular PyTorch Source Code (`src/`) for Complete Colab & Cluster Reproducibility.
              </p>
            </div>
          </div>

          <button
            onClick={onDownloadZip}
            disabled={isDownloading}
            className="flex items-center gap-2 px-4 py-2 text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 rounded-xl transition-all shadow-lg shadow-cyan-500/20 active:scale-95 disabled:opacity-50"
          >
            <Download className="w-4 h-4" />
            <span>{isDownloading ? 'Building ZIP Package...' : 'Download Full Project ZIP (.zip)'}</span>
          </button>
        </div>

        {/* View Mode Switcher */}
        <div className="flex items-center gap-2 pt-2 border-t border-slate-800">
          <button
            onClick={() => setActiveTab('notebooks')}
            className={`flex items-center gap-2 px-3.5 py-1.5 text-xs font-medium rounded-lg transition-all ${
              activeTab === 'notebooks'
                ? 'bg-cyan-500 text-slate-950 font-bold'
                : 'text-slate-400 hover:text-slate-200 bg-slate-950'
            }`}
          >
            <BookOpen className="w-3.5 h-3.5" />
            <span>10 Google Colab Notebooks</span>
          </button>
          <button
            onClick={() => setActiveTab('source_code')}
            className={`flex items-center gap-2 px-3.5 py-1.5 text-xs font-medium rounded-lg transition-all ${
              activeTab === 'source_code'
                ? 'bg-cyan-500 text-slate-950 font-bold'
                : 'text-slate-400 hover:text-slate-200 bg-slate-950'
            }`}
          >
            <Terminal className="w-3.5 h-3.5" />
            <span>Modular Python Source Code (`src/`)</span>
          </button>
        </div>
      </div>

      {/* Mode A: 10 Jupyter Notebooks Viewer */}
      {activeTab === 'notebooks' ? (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          
          {/* Notebooks List (4 Cols) */}
          <div className="lg:col-span-4 space-y-2">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-400 font-mono block mb-2">
              Sequential Notebook Index (01–10)
            </span>
            <div className="space-y-1.5 max-h-[600px] overflow-y-auto pr-1">
              {JUPYTER_NOTEBOOKS.map((nb) => {
                const isSelected = selectedNotebook.id === nb.id;
                return (
                  <button
                    key={nb.id}
                    onClick={() => setSelectedNotebook(nb)}
                    className={`w-full p-3 rounded-xl text-left border transition-all ${
                      isSelected
                        ? 'bg-cyan-950/80 border-cyan-500 shadow-md ring-1 ring-cyan-500/50'
                        : 'bg-slate-900/80 border-slate-800/80 hover:border-slate-700 text-slate-400'
                    }`}
                  >
                    <div className="flex items-center justify-between text-[11px] font-mono mb-1">
                      <span className={isSelected ? 'text-cyan-300 font-bold' : 'text-slate-300 font-semibold'}>
                        {nb.fileName}
                      </span>
                      {isSelected && <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />}
                    </div>
                    <div className="text-xs font-medium text-slate-200 line-clamp-1">{nb.title}</div>
                    <p className="text-[10px] text-slate-500 mt-1 line-clamp-1 font-sans">{nb.objective}</p>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Notebook Cell Content Stage (8 Cols) */}
          <div className="lg:col-span-8 bg-slate-900/90 rounded-2xl p-6 border border-slate-800 space-y-5">
            <div className="flex items-start justify-between border-b border-slate-800 pb-4">
              <div>
                <span className="text-[11px] font-mono text-cyan-400 uppercase tracking-wider">
                  Colab Notebook Viewer
                </span>
                <h3 className="text-base font-bold text-white tracking-tight mt-0.5">{selectedNotebook.title}</h3>
                <p className="text-xs text-slate-400 mt-1">{selectedNotebook.objective}</p>
              </div>
              <button
                onClick={() => {
                  const content = JSON.stringify(
                    {
                      nbformat: 4,
                      nbformat_minor: 2,
                      metadata: { colab: { name: selectedNotebook.fileName, provenance: [] } },
                      cells: selectedNotebook.cells,
                    },
                    null,
                    2
                  );
                  handleCopyCode(content, `nb_${selectedNotebook.id}`);
                }}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono text-slate-300 bg-slate-950 hover:bg-slate-800 border border-slate-800 rounded-lg transition-all"
              >
                {copiedKey === `nb_${selectedNotebook.id}` ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-emerald-400" />
                    <span className="text-emerald-400">Copied JSON</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5 text-slate-400" />
                    <span>Copy Notebook JSON</span>
                  </>
                )}
              </button>
            </div>

            {/* Notebook Cells Rendering */}
            <div className="space-y-4 max-h-[520px] overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-slate-800">
              {selectedNotebook.cells.map((cell, idx) => {
                if (cell.cell_type === 'markdown') {
                  return (
                    <div
                      key={idx}
                      className="p-4 bg-slate-950/80 rounded-xl border border-slate-800/80 text-xs text-slate-300 space-y-2 leading-relaxed font-sans whitespace-pre-line"
                    >
                      {cell.source.join('')}
                    </div>
                  );
                } else {
                  return (
                    <div key={idx} className="space-y-1">
                      <div className="flex items-center justify-between text-[11px] font-mono text-slate-500 px-2">
                        <span className="flex items-center gap-1 text-cyan-400">
                          <Play className="w-3 h-3" />
                          <span>In [{cell.execution_count || idx + 1}]:</span>
                        </span>
                        <button
                          onClick={() => handleCopyCode(cell.source.join(''), `cell_${idx}`)}
                          className="hover:text-slate-300 transition-colors"
                        >
                          {copiedKey === `cell_${idx}` ? 'Copied' : 'Copy code'}
                        </button>
                      </div>
                      <pre className="p-3 bg-slate-950 rounded-xl border border-slate-800 text-xs text-cyan-200 font-mono overflow-x-auto">
                        <code>{cell.source.join('')}</code>
                      </pre>
                    </div>
                  );
                }
              })}
            </div>
          </div>

        </div>
      ) : (
        /* Mode B: Python Source Code (`src/`) Explorer */
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          
          {/* File Tree (4 Cols) */}
          <div className="lg:col-span-4 space-y-2">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-400 font-mono block mb-2">
              Repository Tree (`src/`)
            </span>
            <div className="space-y-1.5 max-h-[600px] overflow-y-auto pr-1">
              {SOURCE_CODE_FILES.map((file) => {
                const isSelected = selectedSourceFile.path === file.path;
                return (
                  <button
                    key={file.path}
                    onClick={() => setSelectedSourceFile(file)}
                    className={`w-full p-3 rounded-xl text-left border transition-all ${
                      isSelected
                        ? 'bg-cyan-950/80 border-cyan-500 shadow-md ring-1 ring-cyan-500/50'
                        : 'bg-slate-900/80 border-slate-800/80 hover:border-slate-700 text-slate-400'
                    }`}
                  >
                    <div className="flex items-center justify-between text-xs font-mono mb-1">
                      <span className={isSelected ? 'text-cyan-300 font-bold' : 'text-slate-200'}>
                        {file.path}
                      </span>
                      {isSelected && <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />}
                    </div>
                    <p className="text-[10px] text-slate-500 line-clamp-1 font-sans">{file.description}</p>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Source Code Viewer (8 Cols) */}
          <div className="lg:col-span-8 bg-slate-900/90 rounded-2xl p-6 border border-slate-800 space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <span className="text-xs font-mono text-cyan-400 font-bold">{selectedSourceFile.path}</span>
                <p className="text-[11px] text-slate-400 mt-0.5">{selectedSourceFile.description}</p>
              </div>
              <button
                onClick={() => handleCopyCode(selectedSourceFile.code, selectedSourceFile.path)}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono text-slate-300 bg-slate-950 hover:bg-slate-800 border border-slate-800 rounded-lg transition-all"
              >
                {copiedKey === selectedSourceFile.path ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-emerald-400" />
                    <span className="text-emerald-400">Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5 text-slate-400" />
                    <span>Copy File</span>
                  </>
                )}
              </button>
            </div>

            <pre className="p-4 bg-slate-950 rounded-xl border border-slate-800 text-xs text-slate-300 font-mono max-h-[500px] overflow-y-auto overflow-x-auto scrollbar-thin scrollbar-thumb-slate-800">
              <code>{selectedSourceFile.code}</code>
            </pre>
          </div>

        </div>
      )}

    </div>
  );
};
