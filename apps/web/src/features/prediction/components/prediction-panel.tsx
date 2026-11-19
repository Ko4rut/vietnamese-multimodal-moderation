"use client";

import { useState } from "react";
import { Icon } from "@/components/ui/icon";
import { displayDomain } from "../data/domains";
import type { Prediction } from "../types";
import { formatPredictionScore, PredictionBarChart } from "./prediction-bar-chart";

interface PredictionPanelProps {
  prediction: Prediction | null;
  busy: boolean;
  chartRevision: number;
}

export function PredictionPanel({ prediction, busy, chartRevision }: PredictionPanelProps) {
  const [lastCopied, setLastCopied] = useState<string | null>(null);
  const result = prediction;
  const scores = result?.top_predictions ?? [];
  const probability = result?.score_kind === "probability";
  const winner = scores.find((item) => item.label_id === result?.label_id);
  const copyKey = result ? JSON.stringify([result.model_name, result.label_id, scores]) : null;

  async function copyResult() {
    if (!result || !copyKey) return;
    const summary = [
      "Nguồn: FastAPI (dự đoán thật)",
      "Chủ đề: " + displayDomain(result.label) + " (" + result.label + ")",
      "Mô hình: " + result.model_name,
      "Kiểu điểm: " + (probability ? "Xác suất" : "Decision margin"),
      ...scores.map((item, index) =>
        String(index + 1) + ". " + item.label + ": " +
        formatPredictionScore(item.score, result.score_kind),
      ),
    ].join("\n");
    try {
      await navigator.clipboard.writeText(summary);
      setLastCopied(copyKey);
    } catch {
      setLastCopied(null);
    }
  }

  return (
    <section className="panel result-panel" aria-labelledby="result-heading">
      <div className="panel-heading">
        <div className="section-title-group">
          <span className="panel-icon purple"><Icon name="brain" size={20}/></span>
          <div><h3 id="result-heading">Kết quả phân tích</h3><p>Dự đoán trực tiếp từ mô hình</p></div>
        </div>
        <span className="result-live-tag"><span className="mini-dot"/>INFERENCE</span>
      </div>
      {result ? (
        <div className="result-content" aria-live="polite">
          <div className="result-primary">
            <div className="result-kicker"><Icon name="check-circle" size={16}/>CHỦ ĐỀ ĐƯỢC DỰ ĐOÁN</div>
            <h4>{displayDomain(result.label)}</h4>
            <span className="result-slug">{result.label}</span>
            <div className="result-primary-footer">
              <span><Icon name="target" size={16}/>Phân loại đơn nhãn</span>
              <span className="result-main-score">
                {winner ? formatPredictionScore(winner.score, result.score_kind) : "#" + result.label_id}
              </span>
            </div>
          </div>
          <div className="ranking-header">
            <div>
              <h4>Biểu đồ dự đoán · Top {scores.length} nhãn</h4>
              <p>{probability ? "Xác suất ước lượng theo chủ đề" : "Decision margin theo từng nhãn"}</p>
            </div>
            <button className="copy-button" type="button" onClick={() => void copyResult()}>
              <Icon name={lastCopied === copyKey ? "check" : "copy"} size={16}/>
              {lastCopied === copyKey ? "Đã sao chép" : "Sao chép"}
            </button>
          </div>
          {scores.length > 0 && (
            <PredictionBarChart
              key={chartRevision}
              predictions={scores}
              scoreKind={result.score_kind}
              selectedLabelId={result.label_id}
            />
          )}
          <div className="result-note"><Icon name="info" size={16}/><span>
            {probability
              ? "Logistic Regression trả về xác suất ước lượng; đây không phải độ tin cậy được kiểm định."
              : "LinearSVC trả về decision margin, không phải xác suất. Không diễn giải điểm thành phần trăm."}
          </span></div>
          <div className="model-footnote">Nguồn: <strong>{result.model_name}</strong></div>
        </div>
      ) : (
        <div className="result-empty" aria-live="polite">
          <span className="empty-status">{busy ? "ĐANG XỬ LÝ" : "CHỜ KẾT QUẢ"}</span>
          <h4>{busy ? "AI đang phân tích văn bản..." : "Kết quả thực sẽ hiển thị tại đây"}</h4>
          <p>{busy
            ? "Đang suy luận bằng mô hình đã huấn luyện."
            : "Chọn model, nhập văn bản và nhấn “Phân tích văn bản” để xem kết quả thật."}</p>
          <div className="empty-steps">
            <span>01 · Nhập văn bản</span><Icon name="arrow-right" size={14}/>
            <span>02 · AI phân tích</span><Icon name="arrow-right" size={14}/>
            <span>03 · Kết quả</span>
          </div>
        </div>
      )}
    </section>
  );
}
