import { useState, type ChangeEvent, type DragEvent } from "react";
import axios from "axios";

const API = import.meta.env.VITE_API_URL as string;

type Props = {
  onLoaded: (data: any) => void;
  setColumns: (cols: string[]) => void;
  setError: (e: string | null) => void;
  setLoading: (v: boolean) => void;
};

export default function Upload({ onLoaded, setColumns, setError, setLoading }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);

  const pick = (f: File) => { setError(null); setFile(f); };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) pick(e.target.files[0]);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) pick(f);
  };

  const handleUpload = async () => {
    if (!file) { setError("Please select a file first"); return; }
    if (![".csv", ".xlsx"].some(ext => file.name.toLowerCase().endsWith(ext))) {
      setError("Only CSV and XLSX files are allowed");
      return;
    }

    const form = new FormData();
    form.append("file", file);

    try {
      setUploading(true);
      setLoading(true);
      await axios.post(`${API}/upload`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const resp = await axios.get(`${API}/load/1`);
      const data = resp.data;
      if (data?.columns) setColumns(data.columns);
      onLoaded(data);
    } catch (err: any) {
      setError(err?.response?.data?.detail || err.message || "Upload failed");
    } finally {
      setUploading(false);
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title">
          <div className="card-icon icon-purple">📂</div>
          Data Ingestion
        </div>
        {file && <span className="badge badge-green">File selected</span>}
      </div>

      <div
        className={`file-drop${file ? " has-file" : ""}${dragging ? " has-file" : ""}`}
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        <input type="file" accept=".csv,.xlsx" onChange={handleFileChange} />
        <div style={{ pointerEvents: "none" }}>
          <div style={{ fontSize: 32, marginBottom: 8 }}>
            {file ? "📄" : "⬆️"}
          </div>
          {file ? (
            <>
              <div style={{ color: "var(--text-h)", fontWeight: 600, fontSize: 14 }}>{file.name}</div>
              <div style={{ color: "var(--text-dim)", fontSize: 12, marginTop: 4 }}>
                {(file.size / 1024).toFixed(1)} KB
              </div>
            </>
          ) : (
            <>
              <div style={{ color: "var(--text-h)", fontWeight: 500, fontSize: 14 }}>
                Drop your file here or click to browse
              </div>
              <div style={{ color: "var(--text-dim)", fontSize: 12, marginTop: 4 }}>
                Supports .csv and .xlsx
              </div>
            </>
          )}
        </div>
      </div>

      <div style={{ marginTop: 16 }}>
        <button className="btn btn-primary" onClick={handleUpload} disabled={!file || uploading} style={{ width: "100%", justifyContent: "center" }}>
          {uploading ? <><span className="spinner" />Uploading…</> : "Upload & Preview"}
        </button>
      </div>
    </div>
  );
}
