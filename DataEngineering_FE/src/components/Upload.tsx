import React, { useState } from "react";
import axios from "axios";

type Props = {
  onLoaded: (data: any) => void;
  setColumns: (cols: string[]) => void;
  setError: (e: string | null) => void;
  setLoading: (v: boolean) => void;
};

export default function Upload({ onLoaded, setColumns, setError, setLoading }: Props) {
  const [file, setFile] = useState<File | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null);
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a file first");
      return;
    }

    const allowed = [".csv", ".xlsx"];
    if (!allowed.some((ext) => file.name.toLowerCase().endsWith(ext))) {
      setError("Only CSV and XLSX files are allowed");
      return;
    }

    const form = new FormData();
    form.append("file", file);

    try {
      setLoading(true);
      await axios.post("http://127.0.0.1:8000/upload", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      // load preview for dataset id 1 per requirements
      const resp = await axios.get("http://127.0.0.1:8000/load/1");
      const data = resp.data;
      if (data && data.columns) setColumns(data.columns);
      onLoaded(data);
    } catch (err: any) {
      setError(err?.response?.data?.detail || err.message || "Upload failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card p-3 mb-3">
      <h5>Data Ingestion</h5>
      <div className="mb-2">
        <input type="file" accept=".csv,.xlsx" onChange={handleFileChange} />
      </div>
      <div>
        <button className="btn btn-primary" onClick={handleUpload} disabled={!file}>
          Upload
        </button>
      </div>
    </div>
  );
}
