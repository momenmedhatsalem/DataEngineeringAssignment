type Props = {
  columns: string[];
  selectedTask: string;
  selectedTarget: string | null;
  onTaskChange: (t: string) => void;
  onTargetChange: (t: string | null) => void;
};

const TASKS = [
  { value: "Classification", icon: "🏷️" },
  { value: "Regression",     icon: "📈" },
  { value: "Clustering",     icon: "🔵" },
];

export default function TaskSelector({ columns, selectedTask, selectedTarget, onTaskChange, onTargetChange }: Props) {
  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title">
          <div className="card-icon icon-yellow">⚙️</div>
          Configuration
        </div>
      </div>

      <div className="section-label">ML Task</div>
      <div className="radio-group" style={{ marginBottom: 20 }}>
        {TASKS.map(({ value, icon }) => (
          <label key={value} className={`radio-pill${selectedTask === value ? " active" : ""}`}>
            <input type="radio" name="task" value={value} checked={selectedTask === value} onChange={e => onTaskChange(e.target.value)} />
            <span>{icon}</span>
            {value}
          </label>
        ))}
      </div>

      {(selectedTask === "Classification" || selectedTask === "Regression") && (
        <>
          <div className="section-label">Target Column</div>
          <div style={{ position: "relative" }}>
            <select
              className="form-select"
              value={selectedTarget ?? ""}
              onChange={e => onTargetChange(e.target.value || null)}
            >
              <option value="">— Select target column —</option>
              {columns.map(col => (
                <option key={col} value={col}>{col}</option>
              ))}
            </select>
            <span style={{
              position: "absolute", right: 12, top: "50%", transform: "translateY(-50%)",
              color: "var(--text-dim)", pointerEvents: "none", fontSize: 12
            }}>▼</span>
          </div>
        </>
      )}
    </div>
  );
}
