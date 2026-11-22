import React from 'react';
import { 
  Activity, 
  Layers, 
  Database, 
  Sliders, 
  BarChart3, 
  FileCode2, 
  FileText, 
  Download, 
  ShieldAlert, 
  Sparkles,
  Target,
  Cpu
} from 'lucide-react';

export type NavTab = 
  | 'gemini_studio'
  | 'workbench' 
  | 'architecture' 
  | 'datasets' 
  | 'evaluation' 
  | 'codehub' 
  | 'report';

interface NavbarProps {
  activeTab: NavTab;
  setActiveTab: (tab: NavTab) => void;
  onDownloadProjectZip: () => void;
  isDownloadingZip?: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  onDownloadProjectZip,
  isDownloadingZip = false,
}) => {
  const tabs = [
    { id: 'gemini_studio' as NavTab, label: 'Colab GPU Training', icon: Cpu },
    { id: 'workbench' as NavTab, label: 'Diagnostic Analyzer', icon: Activity },
    { id: 'evaluation' as NavTab, label: 'Test Evaluation', icon: BarChart3 },
    { id: 'architecture' as NavTab, label: 'Architecture', icon: Layers },
    { id: 'datasets' as NavTab, label: 'Datasets', icon: Database },
    { id: 'codehub' as NavTab, label: 'Colab Notebooks', icon: FileCode2 },
    { id: 'report' as NavTab, label: 'Technical Report', icon: FileText },
  ];

  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-800/80 bg-slate-950/90 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 gap-8">
          
          {/* Zone 1: Single text element wordmark */}
          <button
            onClick={() => setActiveTab('gemini_studio')}
            className="flex items-center gap-2.5 whitespace-nowrap shrink-0 text-left cursor-pointer"
          >
            <div className="flex items-center justify-center w-9 h-9 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 shadow-lg shadow-cyan-500/20 ring-1 ring-cyan-400/40">
              <Activity className="w-5 h-5 text-slate-950 stroke-[2.5]" />
            </div>
            <span className="font-extrabold tracking-tight text-lg text-white font-mono">
              Lung<span className="text-cyan-400">DeepNet</span>
            </span>
          </button>

          {/* Zone 2: Navigation Items (single line, whitespace-nowrap) */}
          <nav className="hidden lg:flex items-center gap-1 bg-slate-900/80 p-1 rounded-xl border border-slate-800/80 shrink-0">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  id={`nav-tab-${tab.id}`}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg whitespace-nowrap shrink-0 transition-all ${
                    isActive
                      ? 'bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/25'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Quick Actions: Download Package & Research Disclaimer */}
          <div className="flex items-center gap-2.5">
            <button
              id="btn-download-zip"
              onClick={onDownloadProjectZip}
              disabled={isDownloadingZip}
              className="flex items-center gap-2 px-3.5 py-1.5 text-xs font-semibold text-cyan-300 bg-cyan-950/60 hover:bg-cyan-900/70 border border-cyan-700/50 rounded-xl transition-all shadow-sm active:scale-95 disabled:opacity-50"
              title="Download full project codebase, 10 Google Colab notebooks, unified metadata CSV, and Python source modules"
            >
              <Download className={`w-3.5 h-3.5 ${isDownloadingZip ? 'animate-bounce' : ''}`} />
              <span className="hidden md:inline">
                {isDownloadingZip ? 'Archiving Package...' : 'Download Code & Colab (.ZIP)'}
              </span>
              <span className="md:hidden">ZIP</span>
            </button>
          </div>
        </div>

        {/* Mobile Sub-Navigation Bar */}
        <div className="flex lg:hidden overflow-x-auto py-2.5 gap-1 border-t border-slate-800/50 scrollbar-none">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-1.5 px-2.5 py-1.5 text-xs whitespace-nowrap rounded-lg shrink-0 font-medium ${
                  isActive
                    ? 'bg-cyan-500 text-slate-950 font-semibold'
                    : 'text-slate-400 bg-slate-900 border border-slate-800'
                }`}
              >
                <Icon className="w-3 h-3" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Persistent Academic Non-Clinical Disclaimer Strip */}
      <div className="w-full bg-amber-950/40 border-y border-amber-800/40 px-4 py-1 flex items-center justify-center gap-2 text-[11px] text-amber-300/90 font-medium text-center">
        <ShieldAlert className="w-3.5 h-3.5 text-amber-400 shrink-0" />
        <span>
          <strong>Academic Research Prototype</strong>: For experimental study & peer review only. Not approved as a medical device or clinical diagnostic tool.
        </span>
      </div>
    </header>
  );
};
