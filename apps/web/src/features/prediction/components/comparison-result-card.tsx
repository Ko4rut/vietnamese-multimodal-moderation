import { displayDomain } from "../data/domains";
import { classifierName, type AvailableModel, type ComparisonResult } from "../types";
import { formatPredictionScore, PredictionBarChart } from "./prediction-bar-chart";

interface ComparisonResultCardProps {
  result: ComparisonResult;
  model: AvailableModel | undefined;
}

export function ComparisonResultCard({ result, model }: ComparisonResultCardProps) {
  const prediction = result.prediction;
  const winner = prediction?.top_predictions.find(
    (item) => item.label_id === prediction.label_id
  );

  return (
    <article className="comparison-result-card">
      <header className="comparison-result-header">
        <strong title={result.model_id}>{result.model_id}</strong>
        <span>{model ? classifierName(model.classifier) : "Model không tìm thấy"}</span>
      </header>
      {!prediction || result.status === "error" ? (
        <p className="comparison-result-error" role="alert">
          {result.error ?? "Model không thể trả về kết quả."}
        </p>
      ) : (
        <>
          <div className="comparison-result-winner">
            <span>Nhãn dự đoán</span>
            <strong>{displayDomain(prediction.label)}</strong>
            <small>{prediction.label}</small>
            <b>{winner ? formatPredictionScore(winner.score, prediction.score_kind) : ""}</b>
          </div>
          <p className="comparison-result-kind">
            {prediction.score_kind === "probability"
              ? "Logistic Regression · xác suất ước lượng (0–100%)"
              : "LinearSVC · decision margin (không phải xác suất)"}
          </p>
          <PredictionBarChart
            predictions={prediction.top_predictions}
            selectedLabelId={prediction.label_id}
            scoreKind={prediction.score_kind}
          />
        </>
      )}
    </article>
  );
}
