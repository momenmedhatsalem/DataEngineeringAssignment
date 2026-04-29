import React, { useState } from "react";
import "./App.css";
import Upload from "./components/Upload";
import TaskSelector from "./components/TaskSelector";
import Results from "./components/Results";
import axios from "axios";

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
      setError("Please select a target column");
      return;
    }

    try {
      setLoading(true);
      const resp = await axios.post("http://127.0.0.1:8000/train/1", {
        task,
        target,
      });
      const data = resp.data;
      // backend returns metrics object
      setMetrics(data.metrics || { best_model: data.best_model });
    } catch (err: any) {
      setError(err?.response?.data?.detail || err.message || "Training failed");
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async () => {
    try {
      setLoading(true);
      const resp = await fetch("http://127.0.0.1:8000/download/1");
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
    <div className="container py-4">
      <h3 className="mb-3">AutoML Frontend</h3>

      {error && <div className="alert alert-danger">{error}</div>}

      <Upload onLoaded={setPreview} setColumns={setColumns} setError={setError} setLoading={setLoading} />

      <div className="row">
        <div className="col-md-6">
          <TaskSelector columns={columns} selectedTask={task} selectedTarget={target} onTaskChange={(t) => { setTask(t); setTarget(null);} } onTargetChange={setTarget} />

          <div className="d-flex gap-2 mb-3">
            <button className="btn btn-success" onClick={handleTrain} disabled={loading}>{loading ? 'Training...' : 'Train Model'}</button>
            <button className="btn btn-outline-secondary" onClick={handleDownload} disabled={loading}>{loading ? 'Please wait...' : 'Download Model'}</button>
          </div>

          <Results task={task} metrics={metrics} />
        </div>

        <div className="col-md-6">
          <div className="card p-3">
            <h5>Dataset Preview</h5>
            {preview?.head ? (
              <div style={{ overflowX: "auto" }}>
                <table className="table table-sm">
                  <thead>
                    <tr>
                      {preview.columns.map((c: string) => <th key={c}>{c}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.head.map((row: any, i: number) => (
                      <tr key={i}>
                        {preview.columns.map((c: string) => <td key={c + i}>{String(row[c])}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div>No preview available. Upload a dataset to see a preview.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
