import React from "react";

type Props = {
  task: string;
  metrics: any | null;
};

function ConfusionMatrix({ matrix }: { matrix: number[][] }) {
  return (
    <table className="table table-sm table-bordered">
      <thead>
        <tr>
          <th></th>
          {matrix[0].map((_, i) => <th key={i}>Pred {i}</th>)}
        </tr>
      </thead>
      <tbody>
        {matrix.map((row, r) => (
          <tr key={r}>
            <th scope="row">True {r}</th>
            {row.map((v, c) => <td key={c}>{v}</td>)}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function Results({ task, metrics }: Props) {
  if (!metrics) return null;

  return (
    <div className="card p-3 mb-3">
      <h5>Training Results</h5>
      <div className="mb-2">Best model: <strong>{metrics.best_model || metrics.best_model_name || "-"}</strong></div>

      {task === "Classification" && (
        <div>
          <table className="table table-sm">
            <tbody>
              <tr><th>Accuracy</th><td>{metrics.accuracy?.toFixed(4)}</td></tr>
              <tr><th>Precision</th><td>{metrics.precision?.toFixed(4)}</td></tr>
              <tr><th>Recall</th><td>{metrics.recall?.toFixed(4)}</td></tr>
              <tr><th>F1-score</th><td>{metrics.f1?.toFixed(4)}</td></tr>
            </tbody>
          </table>
          {metrics.confusion_matrix && <ConfusionMatrix matrix={metrics.confusion_matrix} />}
        </div>
      )}

      {task === "Regression" && (
        <div>
          <table className="table table-sm">
            <tbody>
              <tr><th>MAE</th><td>{metrics.mae?.toFixed(4)}</td></tr>
              <tr><th>MSE</th><td>{metrics.mse?.toFixed(4)}</td></tr>
              <tr><th>R²</th><td>{metrics.r2?.toFixed(4)}</td></tr>
            </tbody>
          </table>
        </div>
      )}

      {task === "Clustering" && (
        <div>
          <table className="table table-sm">
            <tbody>
              <tr><th>Silhouette Score</th><td>{metrics.silhouette?.toFixed(4)}</td></tr>
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
