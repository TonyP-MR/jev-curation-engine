import React, { useState, useEffect, useMemo } from 'react';
import {
  Play,
  RotateCw,
  CheckCircle2,
  XCircle,
  TrendingUp,
  DollarSign,
  Zap,
  Database,
  FileText,
  Sparkles,
  Layers,
  BarChart3,
  Copy,
  Check,
  Clock,
  Eye,
  Trash2,
  X,
  GitCompare,
  Search
} from 'lucide-react';

const API = '/api';

async function apiGet(path) {
  const res = await fetch(`${API}${path}`);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

async function apiPost(path, body) {
  const res = await fetch(`${API}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  return res.json();
}

function formatJsonForDisplay(value) {
  const text = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
  return text.replace(/\\n/g, '\n').replace(/\\t/g, '\t');
}

function inlineMarkdown(text, keyPrefix) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  return parts.map((part, index) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={`${keyPrefix}-b-${index}`}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith('`') && part.endsWith('`')) {
      return <code key={`${keyPrefix}-c-${index}`} className="font-mono text-blue-700">{part.slice(1, -1)}</code>;
    }
    return <React.Fragment key={`${keyPrefix}-t-${index}`}>{part}</React.Fragment>;
  });
}

function splitMarkdownTableRow(line) {
  return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(cell => cell.trim());
}

function MarkdownReport({ source }) {
  const lines = String(source || '').split(/\r?\n/);
  const blocks = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) {
      index += 1;
      continue;
    }

    if (line.trim().startsWith('|') && index + 1 < lines.length && /^\s*\|?\s*:?-{3,}/.test(lines[index + 1])) {
      const headers = splitMarkdownTableRow(line);
      index += 2;
      const rows = [];
      while (index < lines.length && lines[index].trim().startsWith('|')) {
        rows.push(splitMarkdownTableRow(lines[index]));
        index += 1;
      }
      blocks.push(
        <div key={`table-${index}`} className="overflow-x-auto my-4">
          <table className="min-w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-300 bg-slate-100">
                {headers.map((header, i) => <th key={i} className="px-3 py-2 font-semibold text-slate-700">{inlineMarkdown(header, `h-${index}-${i}`)}</th>)}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, r) => (
                <tr key={r} className="border-b border-slate-200 even:bg-slate-50">
                  {headers.map((_, c) => <td key={c} className="px-3 py-2 text-slate-700 align-top">{inlineMarkdown(row[c] || '', `r-${index}-${r}-${c}`)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      const level = heading[1].length;
      const Tag = level === 1 ? 'h2' : level === 2 ? 'h3' : 'h4';
      blocks.push(<Tag key={`heading-${index}`} className="font-bold text-slate-900 mt-5 mb-2">{inlineMarkdown(heading[2], `heading-${index}`)}</Tag>);
      index += 1;
      continue;
    }

    if (/^\s*-\s+/.test(line)) {
      const items = [];
      while (index < lines.length && /^\s*-\s+/.test(lines[index])) {
        items.push(lines[index].replace(/^\s*-\s+/, ''));
        index += 1;
      }
      blocks.push(<ul key={`list-${index}`} className="list-disc pl-5 my-3 space-y-1">{items.map((item, i) => <li key={i}>{inlineMarkdown(item, `li-${index}-${i}`)}</li>)}</ul>);
      continue;
    }

    const paragraph = [];
    while (index < lines.length && lines[index].trim() && !/^#{1,4}\s+/.test(lines[index]) && !/^\s*-\s+/.test(lines[index]) && !lines[index].trim().startsWith('|')) {
      paragraph.push(lines[index].trim());
      index += 1;
    }
    blocks.push(<p key={`paragraph-${index}`} className="my-3 leading-relaxed">{inlineMarkdown(paragraph.join(' '), `p-${index}`)}</p>);
  }

  return <div className="text-sm text-slate-700">{blocks}</div>;
}

function StatCard({ label, value, sub, icon, tone = 'slate' }) {
  const tones = {
    slate: 'text-slate-900',
    emerald: 'text-emerald-600',
    blue: 'text-blue-600',
    amber: 'text-amber-600'
  };
  return (
    <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
      <div className="flex items-center justify-between text-slate-500 text-xs font-semibold uppercase tracking-wider">
        <span>{label}</span>
        {icon}
      </div>
      <div className={`mt-2 flex items-baseline gap-2 ${tones[tone]}`}>
        <span className="text-3xl font-extrabold">{value}</span>
        {sub && <span className="text-xs text-slate-500">{sub}</span>}
      </div>
    </div>
  );
}

function Modal({ title, subtitle, onClose, children, wide = false }) {
  return (
    <div className="fixed inset-0 z-50 bg-slate-900/60 flex items-start justify-center overflow-y-auto p-4 sm:p-8">
      <div className={`bg-white rounded-xl shadow-2xl w-full ${wide ? 'max-w-6xl' : 'max-w-3xl'}`}>
        <div className="flex items-start justify-between px-5 py-4 border-b border-slate-200 sticky top-0 bg-white rounded-t-xl z-10">
          <div>
            <h3 className="font-bold text-slate-900 text-base">{title}</h3>
            {subtitle && <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>}
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-slate-100 text-slate-500"
            aria-label="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        <div className="p-5">{children}</div>
      </div>
    </div>
  );
}

function JsonBlock({ value, maxHeight = 'max-h-80' }) {
  const text = formatJsonForDisplay(value);
  return (
    <pre className={`p-3 bg-slate-900 text-slate-100 rounded-lg text-[11px] font-mono overflow-auto ${maxHeight} leading-relaxed whitespace-pre-wrap`}>
      {text}
    </pre>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState('picker');
  const [configs, setConfigs] = useState([]);
  const [blobConfigs, setBlobConfigs] = useState([]);
  const [selectedConfig, setSelectedConfig] = useState('');
  const [blobs, setBlobs] = useState([]);
  const [selectedBlobs, setSelectedBlobs] = useState([]);
  const [loadingBlobs, setLoadingBlobs] = useState(false);
  const [search, setSearch] = useState('');
  const [runs, setRuns] = useState([]);
  const [llmCostOverride, setLlmCostOverride] = useState('');
  const [optimizePrompts, setOptimizePrompts] = useState(false);
  const [optimizerAvailable, setOptimizerAvailable] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [activeRunData, setActiveRunData] = useState(null);
  const [copiedMd, setCopiedMd] = useState(false);
  const [clearingRuns, setClearingRuns] = useState(false);
  const [environment, setEnvironment] = useState('staging');
  const [preview, setPreview] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewTab, setPreviewTab] = useState('request');
  const [detail, setDetail] = useState(null);

  const configCounts = useMemo(() => {
    const m = {};
    for (const c of blobConfigs) m[c.config_id] = c.count;
    return m;
  }, [blobConfigs]);

  useEffect(() => {
    (async () => {
      try {
        const [dbConfigs, runList, health] = await Promise.all([
          apiGet('/configs'),
          apiGet('/runs'),
          apiGet('/health')
        ]);
        setConfigs(dbConfigs);
        setRuns(runList);
        if (health?.environment) setEnvironment(health.environment);
        setOptimizerAvailable(Boolean(health?.prompt_optimization_enabled && health?.prompt_optimization_key_configured));
        const latest = dbConfigs.find(c => c.config_id === health?.latest_audit_config);
        setSelectedConfig(latest?.config_id || dbConfigs[0]?.config_id || '');
      } catch (e) {
        console.error('Initial load failed', e);
      }
    })();
  }, []);

  // Blob list follows the selected config.
  useEffect(() => {
    if (selectedConfig) {
      setSelectedBlobs([]);
      fetchBlobs(selectedConfig);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedConfig]);

  const fetchBlobs = async (configId, refresh = false) => {
    setLoadingBlobs(true);
    try {
      const params = new URLSearchParams({ config_id: configId, limit: '100' });
      if (refresh) params.set('refresh', 'true');
      const data = await apiGet(`/blobs?${params.toString()}`);
      setBlobs(data);
    } catch (e) {
      console.error('Failed to load blobs', e);
      setBlobs([]);
    } finally {
      setLoadingBlobs(false);
    }
  };

  const fetchRuns = async () => {
    try {
      setRuns(await apiGet('/runs'));
    } catch (e) {
      console.error(e);
    }
  };

  const filteredBlobs = useMemo(() => {
    if (!search.trim()) return blobs;
    const q = search.toLowerCase();
    return blobs.filter(
      b =>
        (b.headline || '').toLowerCase().includes(q) ||
        (b.correlation_id || '').toLowerCase().includes(q) ||
        (b.article_id || '').toLowerCase().includes(q)
    );
  }, [blobs, search]);

  const toggleSelectBlob = name =>
    setSelectedBlobs(prev =>
      prev.includes(name) ? prev.filter(b => b !== name) : [...prev, name]
    );

  const allFilteredSelected =
    filteredBlobs.length > 0 && filteredBlobs.every(b => selectedBlobs.includes(b.name));

  const selectAllBlobs = () =>
    setSelectedBlobs(allFilteredSelected ? [] : filteredBlobs.map(b => b.name));

  const openPreview = async blob => {
    setPreviewLoading(true);
    setPreviewTab('request');
    try {
      const body = { blob_name: blob.name, config_id: selectedConfig };
      setPreview(await apiPost('/preview', body));
    } catch (e) {
      console.error(e);
      setPreview({ error: String(e) });
    } finally {
      setPreviewLoading(false);
    }
  };

  const runBenchmark = async () => {
    if (selectedBlobs.length === 0) return;
    setActiveTab('benchmark');
    setJobStatus({ status: 'starting', processed: 0, total: selectedBlobs.length });
    setActiveRunData(null);
    try {
      const payload = {
        blob_names: selectedBlobs,
        config_id: selectedConfig,
        noul_threshold: 0.5,
        optimize_prompts: optimizePrompts
      };
      const override = parseFloat(llmCostOverride);
      if (!Number.isNaN(override)) payload.llm_cost_override_usd = override;
      const data = await apiPost('/benchmark/run', payload);
      setJobId(data.job_id);
      pollJob(data.job_id);
    } catch (e) {
      setJobStatus({ status: 'failed', error: String(e) });
    }
  };

  const pollJob = id => {
    const timer = setInterval(async () => {
      try {
        const job = await apiGet(`/benchmark/jobs/${id}`);
        setJobStatus(job);
        if (job.status === 'completed' || job.status === 'failed') {
          clearInterval(timer);
          if (job.status === 'completed') {
            await loadRunDetails(id);
            fetchRuns();
          }
        }
      } catch (e) {
        console.error('Polling failed', e);
        clearInterval(timer);
      }
    }, 1000);
  };

  const loadRunDetails = async runId => {
    try {
      const data = await apiGet(`/runs/${runId}`);
      setActiveRunData(data);
      setActiveTab('benchmark');
    } catch (e) {
      console.error(e);
    }
  };
  const deleteRun = async runId => {
    if (!window.confirm(`Delete run ${runId}?`)) return;
    try {
      const res = await fetch(`${API}/runs/${runId}`, { method: 'DELETE' });
      if (!res.ok) throw new Error(`${res.status}`);
      if (activeRunData?.run_id === runId) setActiveRunData(null);
      fetchRuns();
    } catch (e) {
      console.error(e);
      alert(`Delete failed: ${e.message}`);
    }
  };

  const clearAllRuns = async () => {
    if (!window.confirm(`Delete all ${runs.length} past runs? This cannot be undone.`)) return;
    setClearingRuns(true);
    try {
      const res = await fetch(`${API}/runs`, { method: 'DELETE' });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `${res.status}`);
      }
      setActiveRunData(null);
      setJobStatus(null);
      await fetchRuns();
    } catch (e) {
      console.error(e);
      alert(`Clear failed: ${e.message}`);
    } finally {
      setClearingRuns(false);
    }
  };


  const copyMarkdown = () => {
    if (activeRunData?.markdown_report) {
      navigator.clipboard.writeText(activeRunData.markdown_report);
      setCopiedMd(true);
      setTimeout(() => setCopiedMd(false), 2000);
    }
  };

  const navBtn = (tab, icon, label) => (
    <button
      onClick={() => setActiveTab(tab)}
      className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
        activeTab === tab
          ? 'bg-slate-800 text-white border border-slate-600'
          : 'text-slate-300 hover:text-white hover:bg-slate-800/50'
      }`}
    >
      {icon}
      <span className="ml-1.5">{label}</span>
    </button>
  );

  const summary = activeRunData?.summary;
  const optimizerInfo = activeRunData?.records?.find(r => r.prompt_optimization?.enabled)?.prompt_optimization;

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">
      <header className="bg-[#152332] text-white shadow-md border-b border-slate-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="bg-[#0b82f4] p-2 rounded-lg">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight flex items-center gap-2">
                jev-curation-engine
                <span className="text-xs bg-[#0b82f4]/30 text-[#60a5fa] px-2 py-0.5 rounded border border-[#0b82f4]/40 font-mono">
                  Test Rig
                </span>
                <span
                  className={`text-xs px-2 py-0.5 rounded font-mono font-semibold border ${
                    environment === 'production'
                      ? 'bg-rose-500/20 text-rose-300 border-rose-400/50'
                      : 'bg-emerald-500/20 text-emerald-300 border-emerald-400/50'
                  }`}
                  title="Curation Engine environment the rig is pointed at"
                >
                  {environment}
                </span>
              </h1>
              <p className="text-xs text-slate-400">Curation Engine System One feasibility benchmark</p>
            </div>
          </div>
          <nav className="flex items-center space-x-1">
            {navBtn('picker', <Database className="w-4 h-4 inline" />, 'Select Blobs')}
            {navBtn('benchmark', <BarChart3 className="w-4 h-4 inline" />, 'Dashboard')}
            {navBtn('history', <Layers className="w-4 h-4 inline" />, `Past Runs (${runs.length})`)}
          </nav>
        </div>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'picker' && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5 flex flex-wrap items-end gap-4">
              <div>
                <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">
                  Curation Config
                </label>
                <select
                  value={selectedConfig}
                  onChange={e => setSelectedConfig(e.target.value)}
                  className="bg-slate-50 border border-slate-300 text-slate-900 text-sm rounded-lg px-3 py-2 font-medium min-w-[16rem]"
                >
                  {configs.map(c => {
                    const count = configCounts[c.config_id];
                    return (
                      <option key={c.config_id} value={c.config_id}>
                        {c.config_name} (v{c.version_number}){count != null ? ` — ${count} articles` : ''}
                      </option>
                    );
                  })}
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">
                  Search
                </label>
                <div className="relative">
                  <Search className="w-4 h-4 text-slate-400 absolute left-2.5 top-2.5" />
                  <input
                    value={search}
                    onChange={e => setSearch(e.target.value)}
                    placeholder="headline, correlation id…"
                    className="pl-8 pr-3 py-2 border border-slate-300 rounded-lg text-sm w-64 bg-slate-50"
                  />
                </div>
              </div>

              <div>
                <label
                  className="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1"
                  title="Audit blobs do not record LLM cost, so the rig needs a baseline figure. Leave blank to use the configured default."
                >
                  LLM cost/article ($)
                </label>
                <input
                  value={llmCostOverride}
                  onChange={e => setLlmCostOverride(e.target.value)}
                  placeholder="0.007 default"
                  className="px-3 py-2 border border-slate-300 rounded-lg text-sm w-36 bg-slate-50 font-mono"
                />
              </div>
              <label className={`flex items-center gap-2 text-xs whitespace-nowrap ${optimizerAvailable ? 'text-slate-600 cursor-pointer' : 'text-slate-400 cursor-not-allowed'}`} title={optimizerAvailable ? 'Compile the configuration rules with Gemini once, then reuse the cached rubric for this run.' : 'Add GEMINI_API_KEY and set PROMPT_OPTIMIZATION_ENABLED=true to enable this option.'}>
                <input
                  type="checkbox"
                  checked={optimizePrompts}
                  disabled={!optimizerAvailable}
                  onChange={e => setOptimizePrompts(e.target.checked)}
                  className="h-4 w-4 rounded border-slate-300 text-[#0b82f4] disabled:opacity-50"
                />
                Optimize prompts with Gemini
              </label>

              <div className="ml-auto w-full lg:w-auto flex flex-wrap items-center justify-end gap-2">
                <span className="text-sm text-slate-600 whitespace-nowrap">
                  <span className="font-semibold text-slate-900">{selectedBlobs.length}</span> of{' '}
                  <span className="font-semibold">{filteredBlobs.length}</span> selected
                </span>
                <button
                  onClick={() => fetchBlobs(selectedConfig, true)}
                  title="Refresh the processed article list from pipeline_audit_log"
                  className="px-3 py-2 rounded-lg border border-slate-300 text-slate-700 bg-white hover:bg-slate-50 text-sm font-medium inline-flex items-center gap-1.5 shadow-sm"
                >
                  <RotateCw className={`w-4 h-4 ${loadingBlobs ? 'animate-spin' : ''}`} /> Refresh
                </button>
                <button
                  onClick={selectAllBlobs}
                  disabled={filteredBlobs.length === 0}
                  className="px-3 py-2 rounded-lg border border-slate-300 text-slate-700 bg-white hover:bg-slate-50 text-sm font-medium inline-flex items-center shadow-sm disabled:opacity-50"
                >
                  {allFilteredSelected ? 'Deselect All' : 'Select All'}
                </button>
                <button
                  onClick={runBenchmark}
                  disabled={selectedBlobs.length === 0}
                  className="px-5 py-2 rounded-lg bg-[#0b82f4] hover:bg-[#096fd1] text-white text-sm font-semibold inline-flex items-center gap-2 shadow-sm disabled:opacity-50"
                >
                  <Play className="w-4 h-4 fill-white" /> Run Jev Benchmark
                </button>
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
              <div className="px-5 py-4 border-b border-slate-200 flex items-center justify-between bg-slate-50/60">
                <h3 className="font-semibold text-slate-900 text-sm flex items-center gap-2">
                  <FileText className="w-4 h-4 text-slate-500" />
                  Processed Article Blobs — config <span className="font-mono">{selectedConfig || '—'}</span>
                </h3>
                <span className="text-xs text-slate-500">Click a row to select · Preview to inspect the TypeSafe payload</span>
              </div>

              <div className="divide-y divide-slate-100 max-h-[640px] overflow-y-auto">
                {loadingBlobs && <div className="p-6 text-sm text-slate-500">Loading blobs…</div>}
                {!loadingBlobs && filteredBlobs.length === 0 && (
                  <div className="p-6 text-sm text-slate-500">
                    No cached blobs for this config. Pick another config or hit Refresh.
                  </div>
                )}
                {filteredBlobs.map(blob => {
                  const isSelected = selectedBlobs.includes(blob.name);
                  return (
                    <div
                      key={blob.name}
                      className={`p-4 flex items-center justify-between gap-4 ${
                        isSelected ? 'bg-blue-50/60 border-l-4 border-[#0b82f4]' : 'hover:bg-slate-50 border-l-4 border-transparent'
                      }`}
                    >
                      <div className="flex items-start gap-3 min-w-0 cursor-pointer" onClick={() => toggleSelectBlob(blob.name)}>
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => {}}
                          className="mt-1 h-4 w-4 text-[#0b82f4] rounded border-slate-300"
                        />
                        <div className="min-w-0">
                          <p className="font-semibold text-slate-900 text-sm truncate">{blob.headline}</p>
                          <div className="flex items-center gap-2 text-xs text-slate-500 mt-1 flex-wrap">
                            <span className="font-mono bg-slate-100 px-1.5 py-0.5 rounded">
                              {blob.correlation_id?.substring(0, 16)}…
                            </span>
                            <span>{blob.media_type || 'article'}</span>
                            {blob.processed_at && (
                              <span className="flex items-center gap-1">
                                <Clock className="w-3 h-3" />
                                {String(blob.processed_at).substring(0, 10)}
                              </span>
                            )}
                            {blob.article_id && <span>article {blob.article_id}</span>}
                          </div>
                        </div>
                      </div>
                      <button
                        onClick={() => openPreview(blob)}
                        className="shrink-0 px-3 py-1.5 rounded-lg border border-slate-300 text-slate-700 hover:bg-slate-100 text-xs font-semibold flex items-center gap-1.5"
                      >
                        <Eye className="w-3.5 h-3.5" /> Preview
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'benchmark' && (
          <div className="space-y-6">
            {jobStatus && (jobStatus.status === 'running' || jobStatus.status === 'starting') && (
              <div className="bg-white p-6 rounded-xl border border-blue-200 shadow-sm flex items-center justify-between">
                <div>
                  <h4 className="font-semibold text-slate-900 text-sm flex items-center gap-2">
                    <RotateCw className="w-4 h-4 text-blue-600 animate-spin" />
                    {jobStatus.phase === 'optimizing_prompts'
                      ? `Optimizing prompts for ${jobStatus.optimizer_config || 'configuration'}…`
                      : 'Evaluating with TypeSafe Jev…'}
                  </h4>
                  <p className="text-xs text-slate-500 mt-1">
                    {jobStatus.phase === 'optimizing_prompts'
                      ? `Gemini ${jobStatus.optimizer_status || 'compiling'} rubric before article evaluation`
                      : `${jobStatus.processed || 0} / ${jobStatus.total} articles`}
                  </p>
                </div>
                <div className="w-48 bg-slate-100 h-2.5 rounded-full overflow-hidden">
                  <div
                    className="bg-[#0b82f4] h-full transition-all"
                    style={{ width: `${jobStatus.phase === 'optimizing_prompts' ? 8 : Math.round(((jobStatus.processed || 0) / (jobStatus.total || 1)) * 100)}%` }}
                  />
                </div>
              </div>
            )}

            {jobStatus?.status === 'failed' && (
              <div className="bg-rose-50 border border-rose-200 text-rose-800 p-4 rounded-xl text-sm">
                Benchmark failed: {jobStatus.error}
              </div>
            )}

            {summary ? (
              <>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <StatCard
                    label="Subject Validation"
                    value={`${summary.accuracy.subject_validation.accuracy_pct}%`}
                    sub={`(${summary.accuracy.subject_validation.correct}/${summary.accuracy.subject_validation.total})`}
                    icon={<CheckCircle2 className="w-4 h-4 text-emerald-500" />}
                    tone="emerald"
                  />
                  <StatCard
                    label="Prominence Match"
                    value={`${summary.accuracy.subject_prominence.accuracy_pct}%`}
                    sub={`(${summary.accuracy.subject_prominence.correct}/${summary.accuracy.subject_prominence.total})`}
                    icon={<TrendingUp className="w-4 h-4 text-blue-500" />}
                  />
                  <StatCard
                    label="Sentiment Match"
                    value={`${summary.accuracy.subject_sentiment.accuracy_pct}%`}
                    sub={`(${summary.accuracy.subject_sentiment.correct}/${summary.accuracy.subject_sentiment.total})`}
                    icon={<TrendingUp className="w-4 h-4 text-blue-500" />}
                  />
                  <StatCard
                    label="LLM Tags Match"
                    value={`${summary.accuracy.llm_tags.accuracy_pct}%`}
                    sub={`(${summary.accuracy.llm_tags.correct}/${summary.accuracy.llm_tags.total})`}
                    icon={<CheckCircle2 className="w-4 h-4 text-emerald-500" />}
                  />
                  <StatCard
                    label="Latency"
                    value={`${summary.performance.speedup_ratio}×`}
                    sub={`${summary.performance.avg_jev_duration_ms}ms vs ${summary.performance.avg_llm_duration_ms}ms`}
                    icon={<Zap className="w-4 h-4 text-amber-500" />}
                    tone="amber"
                  />
                  <StatCard
                    label={summary.performance.total_llm_classification_cost_usd != null ? 'Classification Cost Multiple' : 'Full LLM Cost Multiple (legacy run)'}
                    value={
                      summary.performance.total_llm_classification_cost_usd != null && summary.performance.classification_cost_multiple != null
                        ? `${summary.performance.classification_cost_multiple}×`
                        : summary.performance.cost_multiple != null
                          ? `${summary.performance.cost_multiple}×`
                          : '—'
                    }
                    sub={summary.performance.total_llm_classification_cost_usd != null
                      ? `estimated $${Number(summary.performance.total_llm_classification_cost_usd).toFixed(6)} → $${Number(summary.performance.total_jev_cost_usd ?? 0).toFixed(6)}`
                      : `full call $${Number(summary.performance.total_llm_cost_usd ?? 0).toFixed(6)} → $${Number(summary.performance.total_jev_cost_usd ?? 0).toFixed(6)}`}
                    icon={<DollarSign className="w-4 h-4 text-emerald-600" />}
                    tone="emerald"
                  />
                </div>

                <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
                  <div className="flex items-center justify-between pb-4 border-b border-slate-200">
                    <div>
                      <h3 className="font-bold text-slate-900 text-base">Business summary</h3>
                      <p className="text-xs text-slate-500">Saved to runs/{activeRunData.run_id}/benchmark_summary.md</p>
                    </div>
                    <button
                      onClick={copyMarkdown}
                      className="px-4 py-2 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-800 text-xs font-semibold flex items-center gap-1.5"
                    >
                      {copiedMd ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                      {copiedMd ? 'Copied' : 'Copy Markdown'}
                    </button>
                  </div>
                  {summary.performance.total_llm_classification_cost_usd == null && (
                    <div className="mt-3 px-3 py-2 rounded-lg bg-amber-50 border border-amber-200 text-xs text-amber-800">
                      This run predates the like-for-like classification cost calculation. Start a new run to see the estimated classification-only comparison.
                    </div>
                  )}
                  {optimizerInfo && (
                    <div className="mt-3 px-3 py-2 rounded-lg bg-violet-50 border border-violet-200 text-xs text-violet-800">
                      Prompt optimizer: {optimizerInfo.model} · {optimizerInfo.cached ? 'cached rubric' : 'compiled this run'} · optimizer cost ${Number(optimizerInfo.cost_usd || 0).toFixed(6)}
                    </div>
                  )}
                  {summary.prompt_optimization?.enabled && (
                    <div className="mt-2 px-3 py-2 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-600">
                      Question payload: {summary.prompt_optimization.original_question_chars.toLocaleString()} → {summary.prompt_optimization.optimized_question_chars.toLocaleString()} characters ({summary.prompt_optimization.estimated_char_reduction_pct ?? 0}% estimated reduction) across {summary.prompt_optimization.configs_compiled} config(s).
                    </div>
                  )}
                  <div className="mt-4 p-5 bg-white border border-slate-200 rounded-lg max-h-[34rem] overflow-auto">
                    <MarkdownReport source={activeRunData.markdown_report} />
                  </div>
                </div>

                <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                  <div className="px-5 py-4 border-b border-slate-200 bg-slate-50/60 flex items-center justify-between">
                    <h3 className="font-semibold text-slate-900 text-sm">
                      Article results ({activeRunData.records?.length || 0})
                    </h3>
                    <span className="text-xs text-slate-500">Open Compare for LLM vs Jev side by side</span>
                  </div>
                  <div className="divide-y divide-slate-100">
                    {activeRunData.records?.filter(r => !r.skipped).map((record, i) => (
                      <div key={i} className="p-4 flex items-center justify-between gap-4 hover:bg-slate-50">
                        <div className="min-w-0">
                          <p className="font-semibold text-slate-900 text-sm truncate">{record.headline}</p>
                          <div className="flex items-center gap-3 text-xs text-slate-500 mt-1">
                            <span className="font-mono">{record.correlation_id?.substring(0, 16)}…</span>
                            <span className="text-slate-600">
                              valid {record.metrics.validation_accuracy}% · prom {record.metrics.prominence_accuracy}% · sent{' '}
                              {record.metrics.sentiment_accuracy}% · tags {record.metrics.tag_accuracy}%
                            </span>
                          </div>
                        </div>
                        <div className="flex items-center gap-3 shrink-0">
                          <div className="text-right text-xs">
                            <span className="font-semibold text-emerald-600">{record.performance.speedup_ratio}×</span>
                            <span className="text-slate-400 block">{record.performance.jev_duration_ms}ms</span>
                          </div>
                          <button
                            onClick={() => setDetail(record)}
                            className="px-3 py-1.5 rounded-lg border border-slate-300 text-slate-700 hover:bg-slate-100 text-xs font-semibold flex items-center gap-1.5"
                          >
                            <GitCompare className="w-3.5 h-3.5" /> Compare
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              !jobStatus && (
                <div className="bg-white p-12 text-center rounded-xl border border-slate-200 shadow-sm">
                  <BarChart3 className="w-12 h-12 text-slate-300 mx-auto mb-3" />
                  <h3 className="text-base font-semibold text-slate-800">No benchmark loaded</h3>
                  <p className="text-xs text-slate-500 mt-1">Select blobs and run, or open a past run.</p>
                </div>
              )
            )}
          </div>
        )}

        {activeTab === 'history' && (
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
            <div className="px-5 py-4 border-b border-slate-200 bg-slate-50/60 flex items-center justify-between">
              <h3 className="font-semibold text-slate-900 text-sm">Past benchmark runs ({runs.length})</h3>
              <button
                onClick={clearAllRuns}
                disabled={runs.length === 0 || clearingRuns}
                className="px-3 py-1.5 rounded-lg border border-rose-200 text-rose-700 bg-rose-50 hover:bg-rose-100 text-xs font-semibold disabled:opacity-50"
              >
                <Trash2 className="w-3.5 h-3.5 inline mr-1" />
                {clearingRuns ? 'Clearing…' : 'Clear All'}
              </button>
            </div>
            <div className="divide-y divide-slate-100">
              {runs.map(r => (
                <div key={r.run_id} className="p-4 flex items-center justify-between hover:bg-slate-50">
                  <div>
                    <h4 className="font-mono text-sm font-semibold text-slate-900">{r.run_id}</h4>
                    <p className="text-xs text-slate-500 mt-0.5">
                      {r.article_count || r.total_articles || '?'} articles · {r.completed_at || r.created_at}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => loadRunDetails(r.run_id)}
                      className="px-3 py-1.5 rounded-lg border border-slate-300 text-slate-700 hover:bg-slate-100 text-xs font-semibold"
                    >
                      Open
                    </button>
                    <button
                      onClick={() => deleteRun(r.run_id)}
                      className="px-2 py-1.5 rounded-lg border border-slate-200 text-slate-400 hover:text-rose-600 hover:border-rose-200 text-xs"
                      aria-label={`Delete ${r.run_id}`}
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              ))}
              {runs.length === 0 && <div className="p-6 text-sm text-slate-500">No runs yet.</div>}
            </div>
          </div>
        )}
      </main>

      {/* Payload inspector */}
      {(preview || previewLoading) && (
        <Modal
          wide
          title="TypeSafe API payload"
          subtitle={
            preview?.config
              ? `${preview.config.config_name} v${preview.config.version_number} · ${preview.question_count} questions · model ${preview.typesafe_request.model}`
              : 'Assembling…'
          }
          onClose={() => {
            setPreview(null);
            setPreviewLoading(false);
          }}
        >
          {previewLoading && <p className="text-sm text-slate-500">Building payload…</p>}
          {preview?.error && <p className="text-sm text-rose-600">{preview.error}</p>}
          {preview && !preview.error && (
            <div className="space-y-4">
              <div className="flex gap-1 border-b border-slate-200">
                {[
                  ['request', 'TypeSafe request'],
                  ['response', 'LLM baseline (from blob)'],
                  ['state', 'Article state']
                ].map(([key, label]) => (
                  <button
                    key={key}
                    onClick={() => setPreviewTab(key)}
                    className={`px-3 py-2 text-sm font-medium border-b-2 -mb-px ${
                      previewTab === key
                        ? 'border-[#0b82f4] text-[#0b82f4]'
                        : 'border-transparent text-slate-500 hover:text-slate-800'
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>

              {previewTab === 'request' && (
                <div className="space-y-3">
                  <p className="text-xs text-slate-500">
                    POST https://api.typesafe.ai/v1/systemone — <span className="font-mono">state</span> plus one typed
                    question per subject/tag.
                  </p>
                  <JsonBlock value={preview.typesafe_request} maxHeight="max-h-[28rem]" />
                </div>
              )}

              {previewTab === 'response' && (
                <div className="space-y-3">
                  <p className="text-xs text-slate-500">
                    Provider {preview.llm_baseline.provider} / {preview.llm_baseline.model} · overall validation{' '}
                    <span className="font-mono">{String(preview.llm_baseline.validation_result)}</span>
                  </p>
                  <JsonBlock value={preview.llm_baseline} maxHeight="max-h-[28rem]" />
                </div>
              )}

              {previewTab === 'state' && (
                <div className="space-y-3">
                  <p className="text-xs text-slate-500">
                    {preview.state_meta.body_length} chars · media type {preview.state_meta.media_type}
                  </p>
                  <JsonBlock value={preview.typesafe_request.state} maxHeight="max-h-[28rem]" />
                </div>
              )}
            </div>
          )}
        </Modal>
      )}

      {/* Side-by-side comparison */}
      {detail && (
        <Modal
          wide
          title={detail.headline}
          subtitle={`LLM (${detail.performance.llm_model}) vs TypeSafe Jev (${detail.performance.jev_model})`}
          onClose={() => setDetail(null)}
        >
          <div className="space-y-5">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <StatCard
                label="Validation"
                value={`${detail.metrics.validation_accuracy}%`}
                sub={`${detail.metrics.val_counts.correct}/${detail.metrics.val_counts.total}`}
                tone="emerald"
              />
              <StatCard
                label="Prominence"
                value={`${detail.metrics.prominence_accuracy}%`}
                sub={`${detail.metrics.prom_counts.correct}/${detail.metrics.prom_counts.total}`}
              />
              <StatCard
                label="Sentiment"
                value={`${detail.metrics.sentiment_accuracy}%`}
                sub={`${detail.metrics.sent_counts.correct}/${detail.metrics.sent_counts.total}`}
              />
              <StatCard
                label="LLM Tags"
                value={`${detail.metrics.tag_accuracy}%`}
                sub={`${detail.metrics.tag_counts.correct}/${detail.metrics.tag_counts.total}`}
                tone="emerald"
              />
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
                <div className="font-semibold text-slate-700 mb-1">LLM baseline</div>
                <div className="text-slate-600 space-y-0.5">
                  <div>latency {detail.performance.llm_duration_ms} ms</div>
                  <div>
                    cost ${detail.performance.llm_cost_usd?.toFixed(6)}{' '}
                    <span className="text-slate-400">({detail.performance.llm_cost_source})</span>
                  </div>
                  <div>
                    tokens {detail.performance.llm_tokens?.input} in / {detail.performance.llm_tokens?.output} out
                  </div>
                </div>
              </div>
              <div className="p-3 rounded-lg bg-blue-50 border border-blue-200">
                <div className="font-semibold text-blue-900 mb-1">TypeSafe Jev</div>
                <div className="text-blue-800 space-y-0.5">
                  <div>latency {detail.performance.jev_duration_ms} ms</div>
                  <div>
                    cost ${detail.performance.jev_cost_usd?.toFixed(6)}{' '}
                    <span className="text-blue-400">(input tokens only)</span>
                  </div>
                  <div>
                    tokens {detail.performance.jev_usage?.input_tokens} in / {detail.performance.jev_usage?.output_tokens} out
                  </div>
                </div>
              </div>
            </div>

            <div>
              <h4 className="font-semibold text-slate-900 text-sm mb-2">Per-subject decisions</h4>
              <div className="space-y-2">
                {detail.llm_output?.subjects?.map(llmSub => {
                  const jevSub = detail.jev_output?.subjects?.find(s => s.subject_id === llmSub.subject_id);
                  const row = (label, llmVal, jevVal, match) => (
                    <div className="grid grid-cols-[6rem_minmax(0,1fr)_minmax(0,1fr)_1.25rem] items-center gap-2">
                      <span className="text-slate-500">{label}</span>
                      <span className="font-mono text-slate-700 min-w-0 truncate text-left">{String(llmVal)}</span>
                      <span className="font-mono text-slate-700 min-w-0 truncate text-left">{String(jevVal)}</span>
                      <span className="w-5 shrink-0">
                        {match ? (
                          <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                        ) : (
                          <XCircle className="w-4 h-4 text-rose-500" />
                        )}
                      </span>
                    </div>
                  );
                  return (
                    <div key={llmSub.subject_id} className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                      <div className="font-bold text-slate-800 text-sm mb-2">{llmSub.name}</div>
                      <div className="grid grid-cols-[6rem_minmax(0,1fr)_minmax(0,1fr)_1.25rem] items-center gap-2 text-[10px] uppercase tracking-wider text-slate-400 font-semibold mb-1">
                        <span></span>
                        <span>LLM</span>
                        <span>Jev</span>
                        <span></span>
                      </div>
                      <div className="space-y-1 text-xs">
                        {row(
                          'Valid',
                          llmSub.is_valid,
                          `${jevSub?.is_valid} (p=${jevSub?.validation_probability})`,
                          llmSub.is_valid === jevSub?.is_valid
                        )}
                        {row('Prominence', llmSub.prominence, jevSub?.prominence, llmSub.prominence === jevSub?.prominence)}
                        {row('Sentiment', llmSub.sentiment, jevSub?.sentiment, llmSub.sentiment === jevSub?.sentiment)}
                      </div>
                      {jevSub?.tags?.length > 0 && (
                        <div className="mt-2 pt-2 border-t border-slate-200 text-xs space-y-1">
                          {jevSub.tags.map(t => {
                            const llmTag = detail.tag_comparisons?.find(
                              x => x.tag_id === t.tag_id && x.subject_id === llmSub.subject_id
                            );
                            return (
                              <div key={t.tag_id} className="grid grid-cols-[minmax(0,1.4fr)_minmax(5rem,0.7fr)_minmax(0,1fr)_1.25rem] items-center gap-2">
                                <span className="text-slate-500 min-w-0 truncate text-left">{t.tag_name}</span>
                                <span className="font-mono text-slate-700 text-left">{String(llmTag?.llm_result)}</span>
                                <span className="font-mono text-slate-700 min-w-0 truncate text-left">
                                  {String(t.result)} (p={t.probability})
                                </span>
                                <span className="w-5">
                                  {llmTag?.match ? (
                                    <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                                  ) : (
                                    <XCircle className="w-4 h-4 text-rose-500" />
                                  )}
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div>
                <h4 className="font-semibold text-slate-900 text-sm mb-2">LLM raw response</h4>
                <JsonBlock value={detail.llm_output?.parsed_raw_response ?? 'not archived'} />
              </div>
              <div>
                <h4 className="font-semibold text-slate-900 text-sm mb-2">TypeSafe response</h4>
                <JsonBlock value={detail.jev_response ?? 'not archived'} />
              </div>
            </div>

            <details>
              <summary className="text-xs font-semibold text-slate-600 cursor-pointer">
                TypeSafe request payload sent for this article
              </summary>
              <div className="mt-2">
                <JsonBlock value={detail.jev_request ?? 'not archived'} maxHeight="max-h-72" />
              </div>
            </details>
          </div>
        </Modal>
      )}
    </div>
  );
}
