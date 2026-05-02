import { useState } from "react";
import "./App.css";
import Upload from "./components/Upload";
import TaskSelector from "./components/TaskSelector";
import Results from "./components/Results";
import axios from "axios";

const API = import.meta.env.VITE_API_URL as string;

export default function App() {
  const [preview, setPreview] = useState<any | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [task, setTask] = useState<string>("Classification");
  const [target, setTarget] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<any | null>(null);

  const handleTrain = async () => {
    setError(null);
    if ((task === "Classification" || task === "Regression") && !target) {
      setError("Please select a target column before training.");
      return;
    }
    try {
      setLoading(true);
      const resp = await axios.post(`${API}/train/1`, { task, target });
      setMetrics(resp.data.metrics || { best_model: resp.data.best_model });
    } catch (err: any) {
      setError(err?.response?.data?.detail || err.message || "Training failed");
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async () => {
    try {
      setLoading(true);
      const resp = await fetch(`${API}/download/1`);
      if (!resp.ok) throw new Error("Download failed");
      const blob = await resp.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "final_model.joblib";
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err: any) {
      setError(err?.message || "Download failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      {/* ── Nav ── */}
      <nav className="topnav">
        <div className="topnav-brand">
          <div className="topnav-brand-dot" />
          AutoML Studio
        </div>
        <div className="topnav-status">
          <div className="status-dot" />
          API connected
        </div>
      </nav>

      <main className="main-content">
        {error && (
          <div className="alert alert-danger">
            <span>⚠️</span> {error}
          </div>
        )}

        {/* ── Loading bar ── */}
        {loading && <div className="progress-bar" />}

        <div className="grid-2">
          {/* ── Left panel ── */}
          <div className="left-panel">
            <Upload
              onLoaded={d => { setPreview(d); setMetrics(null); }}
              setColumns={setColumns}
              setError={setError}
              setLoading={setLoading}
            />

            <TaskSelector
              columns={columns}
              selectedTask={task}
              selectedTarget={target}
              onTaskChange={t => { setTask(t); setTarget(null); setMetrics(null); }}
              onTargetChange={setTarget}
            />

            <div className="action-row">
              <button
                className="btn btn-success"
                onClick={handleTrain}
                disabled={loading || !preview}
                style={{ flex: 1, justifyContent: "center" }}
              >
                {loading ? <><span className="spinner" />Training…</> : <><span>🚀</span> Train Model</>}
              </button>
              <button
                className="btn btn-outline"
                onClick={handleDownload}
                disabled={loading || !metrics}
              >
                <span>💾</span> Save Model
              </button>
            </div>

            <Results task={task} metrics={metrics} />
          </div>

          {/* ── Right panel: dataset preview ── */}
          <div className="card" style={{ alignSelf: "start" }}>
            <div className="card-header">
              <div className="card-title">
                <div className="card-icon icon-blue">🗂️</div>
                Dataset Preview
              </div>
              {preview && (
                <span className="badge badge-accent">
                  {preview.columns?.length} cols
                </span>
              )}
            </div>

            {preview?.head ? (
              <div className="preview-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      {preview.columns.map((c: string) => <th key={c}>{c}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.head.map((row: any, i: number) => (
                      <tr key={i}>
                        {preview.columns.map((c: string) => (
                          <td key={c + i}>{String(row[c])}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="empty-state">
                <div className="empty-icon">📭</div>
                <div className="empty-text">Upload a dataset to see a preview</div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
