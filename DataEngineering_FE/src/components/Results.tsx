type Props = {
  task: string;
  metrics: any | null;
};

function MetricCard({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="metric-card" style={accent ? { borderColor: "rgba(108,99,255,0.3)", background: "rgba(108,99,255,0.07)" } : {}}>
      <div className="metric-label">{label}</div>
      <div className="metric-value" style={accent ? { color: "var(--accent-2)" } : {}}>{value}</div>
    </div>
  );
}

function ConfusionMatrix({ matrix }: { matrix: number[][] }) {
  const max = Math.max(...matrix.flat());
  return (
    <div>
      <div className="section-label" style={{ marginTop: 16 }}>Confusion Matrix</div>
      <div style={{ overflowX: "auto" }}>
        <table className="cm-table">
          <thead>
            <tr>
              <th></th>
              {matrix[0].map((_, i) => <th key={i}>Pred {i}</th>)}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, r) => (
              <tr key={r}>
                <th>True {r}</th>
                {row.map((v, c) => (
                  <td
                    key={c}
                    className={r === c ? "cm-diag" : ""}
                    style={r !== c && v > 0 ? { opacity: 0.4 + 0.6 * (v / max) } : {}}
                  >
                    {v}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function Results({ task, metrics }: Props) {
  if (!metrics) return null;

  const modelName = metrics.best_model || metrics.best_model_name || "—";

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title">
          <div className="card-icon icon-green">📊</div>
          Training Results
        </div>
        <span className="badge badge-green">Done</span>
      </div>

      <div className="best-model-banner">
        <div>
          <div className="best-model-label">Best Model</div>
          <div className="best-model-name">{modelName}</div>
        </div>
        <span style={{ fontSize: 24 }}>🏆</span>
      </div>

      {task === "Classification" && (
        <>
          <div className="metrics-grid metrics-grid-4">
            <MetricCard label="Accuracy"  value={(metrics.accuracy  * 100).toFixed(1) + "%"} accent />
            <MetricCard label="Precision" value={(metrics.precision * 100).toFixed(1) + "%"} />
            <MetricCard label="Recall"    value={(metrics.recall    * 100).toFixed(1) + "%"} />
            <MetricCard label="F1-Score"  value={(metrics.f1        * 100).toFixed(1) + "%"} />
          </div>
          {metrics.confusion_matrix && <ConfusionMatrix matrix={metrics.confusion_matrix} />}
        </>
      )}

      {task === "Regression" && (
        <div className="metrics-grid">
          <MetricCard label="R² Score" value={metrics.r2?.toFixed(4)}  accent />
          <MetricCard label="MAE"      value={metrics.mae?.toFixed(4)} />
          <MetricCard label="MSE"      value={metrics.mse?.toFixed(4)} />
        </div>
      )}

      {task === "Clustering" && (
        <div className="metrics-grid">
          <MetricCard label="Silhouette Score" value={metrics.silhouette?.toFixed(4)} accent />
        </div>
      )}
    </div>
  );
}
