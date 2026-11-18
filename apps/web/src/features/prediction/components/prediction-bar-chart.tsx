import type { CSSProperties } from "react";

import { displayDomain } from "../data/domains";
import type { RankedPrediction, ScoreKind } from "../types";

interface PredictionBarChartProps {
  predictions: RankedPrediction[];
  scoreKind: ScoreKind;
  selectedLabelId: number;
}

export function formatPredictionScore(value: number, scoreKind: ScoreKind): string {
  return scoreKind === "probability"
    ? (value * 100).toFixed(1) + "%"
    : value.toFixed(3);
}

/**
 * Probabilities use the actual 0–100% scale.
 * LinearSVC margins can be negative and have no probabilistic meaning, so
 * their heights represent relative positions within the returned top-five.
 * The original signed margin is always shown above each column.
 */
export function chartHeight(
  score: number,
  scores: number[],
  scoreKind: ScoreKind,
): number {
  if (scoreKind === "probability") {
    return Math.max(0, Math.min(100, score * 100));
  }

  const minimum = Math.min(...scores);
  const maximum = Math.max(...scores);
  return maximum === minimum ? 50 : 14 + (74 * (score - minimum)) / (maximum - minimum);
}

const PROBABILITY_TICKS = [100, 75, 50, 25, 0];

export function PredictionBarChart({
  predictions,
  scoreKind,
  selectedLabelId,
}: PredictionBarChartProps) {
  const probability = scoreKind === "probability";
  const scores = predictions.map((prediction) => prediction.score);
  const count = predictions.length;

  return (
    <figure className="prediction-chart">
      <div className="prediction-chart-key">
        <span><i className="prediction-chart-key-winner" />Nhãn dự đoán</span>
        <span><i className="prediction-chart-key-other" />Các nhãn còn lại</span>
      </div>

      <div className="prediction-chart-plot">
        <div className="prediction-chart-scale" aria-hidden="true">
          {probability
            ? PROBABILITY_TICKS.map((tick) => (
              <span key={tick} style={{ top: (100 - tick) + "%" }}>{tick}%</span>
            ))
            : <><span className="prediction-chart-high">Cao</span><span className="prediction-chart-low">Thấp</span></>}
        </div>

        <div className="prediction-chart-stage">
          {PROBABILITY_TICKS.map((tick) => (
            <span
              key={tick}
              className="prediction-chart-guide"
              style={{ top: (100 - tick) + "%" }}
              aria-hidden="true"
            />
          ))}
          <div
            className="prediction-chart-bars"
            role="list"
            aria-label={probability ? "Xác suất theo nhãn" : "Điểm quyết định theo nhãn"}
            style={{ "--chart-count": count } as CSSProperties}
          >
            {predictions.map((prediction, index) => {
              const height = chartHeight(prediction.score, scores, scoreKind);
              const isWinner = prediction.label_id === selectedLabelId;
              const cssVars = {
                "--bar-height": height.toFixed(2) + "%",
                "--bar-delay": index * 120 + "ms",
              } as CSSProperties;

              return (
                <div
                  className={"prediction-chart-column" + (isWinner ? " prediction-chart-column-winner" : "")}
                  role="listitem"
                  aria-label={
                    displayDomain(prediction.label) + ": " +
                    formatPredictionScore(prediction.score, scoreKind) +
                    (isWinner ? ", nhãn dự đoán" : "")
                  }
                  key={prediction.label_id}
                  style={cssVars}
                >
                  <span className="prediction-chart-score" aria-hidden="true">
                    {formatPredictionScore(prediction.score, scoreKind)}
                  </span>
                  <span className="prediction-chart-bar" aria-hidden="true" />
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div
        className="prediction-chart-labels"
        style={{ "--chart-count": count } as CSSProperties}
      >
        {predictions.map((prediction, index) => (
          <div
            className={"prediction-chart-label" + (prediction.label_id === selectedLabelId ? " prediction-chart-label-winner" : "")}
            key={prediction.label_id}
            title={displayDomain(prediction.label) + " (" + prediction.label + ")"}
          >
            <span className="prediction-chart-rank">#{index + 1}</span>
            <span className="prediction-chart-label-name">{displayDomain(prediction.label)}</span>
          </div>
        ))}
      </div>

      <figcaption className="prediction-chart-caption">
        {probability
          ? "Chiều cao cột = xác suất ước lượng (thang 0–100%)."
          : "Chiều cao cột thể hiện thứ hạng tương đối; số phía trên là decision margin gốc, có thể âm."}
      </figcaption>
    </figure>
  );
}
