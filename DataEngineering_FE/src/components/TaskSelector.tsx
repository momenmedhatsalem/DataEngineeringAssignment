import React from "react";

type Props = {
  columns: string[];
  selectedTask: string;
  selectedTarget: string | null;
  onTaskChange: (t: string) => void;
  onTargetChange: (t: string | null) => void;
};

export default function TaskSelector({ columns, selectedTask, selectedTarget, onTaskChange, onTargetChange }: Props) {
  return (
    <div className="card p-3 mb-3">
      <h5>Task Selection</h5>
      <div className="mb-2">
        <div className="form-check">
          <input className="form-check-input" type="radio" name="task" id="classification" value="Classification" checked={selectedTask === "Classification"} onChange={e => onTaskChange(e.target.value)} />
          <label className="form-check-label" htmlFor="classification">Classification</label>
        </div>
        <div className="form-check">
          <input className="form-check-input" type="radio" name="task" id="regression" value="Regression" checked={selectedTask === "Regression"} onChange={e => onTaskChange(e.target.value)} />
          <label className="form-check-label" htmlFor="regression">Regression</label>
        </div>
        <div className="form-check">
          <input className="form-check-input" type="radio" name="task" id="clustering" value="Clustering" checked={selectedTask === "Clustering"} onChange={e => onTaskChange(e.target.value)} />
          <label className="form-check-label" htmlFor="clustering">Clustering</label>
        </div>
      </div>

      {(selectedTask === "Classification" || selectedTask === "Regression") && (
        <div className="mt-2">
          <label className="form-label">Target column</label>
          <select className="form-select" value={selectedTarget ?? ""} onChange={e => onTargetChange(e.target.value || null)}>
            <option value="">-- Select target --</option>
            {columns.map(col => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
        </div>
      )}
    </div>
  );
}
