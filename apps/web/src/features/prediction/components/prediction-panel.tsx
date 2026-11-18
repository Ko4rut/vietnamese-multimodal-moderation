"use client";

import { useState } from "react";
import { Icon } from "@/components/ui/icon";
import { displayDomain } from "../data/domains";
import type { Prediction, RankedPrediction } from "../types";

interface PredictionPanelProps {
  result: Prediction | null;
  busy: boolean;
}

function formatScore(score: number, probability: boolean): string {
  return probability ? (score * 100).toFixed(1) + "%" : score.toFixed(3);
}

function ScoreRow({ item, rank, probability, min, max }: {
  item: RankedPrediction;
  rank: number;
  probability: boolean;
  min: number;
  max: number;
}) {
  const width = probability
    ? Math.max(0, Math.min(100, item.score * 100))
    : max === min ? 100 : Math.max(5, ((item.score - min) / (max - min)) * 100);
  return (
    <div className="score-row">
      <div className="score-line">
        <div className="score-name"><span className="score-rank">{String(rank).padStart(2, "0")}</span><span title={item.label}>{displayDomain(item.label)}</span></div>
        <span className="score-value">{formatScore(item.score, probability)}</span>
      </div>
      <div className="score-track"><div className={"score-fill" + (rank === 1 ? " best" : "")} style={{ width: width + "%" }} /></div>
    </div>
  );
}

export function PredictionPanel({ result, busy }: PredictionPanelProps) {
  const [copied, setCopied] = useState(false);
  const probability = result?.score_kind === "probability";
  const scores = result?.top_predictions ?? [];
  const min = Math.min(...scores.map((item) => item.score));
  const max = Math.max(...scores.map((item) => item.score));

  async function copyResult() {
    if (!result) return;
    const summary = [
      "Chủ đề dự đoán: " + displayDomain(result.label) + " (" + result.label + ")",
      "Mô hình: " + result.model_name,
      "Kiểu điểm: " + (probability ? "Xác suất" : "Decision margin"),
      ...result.top_predictions.map((item, i) => String(i + 1) + ". " + item.label + ": " + formatScore(item.score, probability)),
    ].join("\n");
    try {
      await navigator.clipboard.writeText(summary);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  }

  return (
    <section className="panel result-panel" aria-labelledby="result-heading">
      <div className="panel-heading">
        <div className="section-title-group"><span className="panel-icon purple"><Icon name="brain" size={20}/></span>
          <div><h3 id="result-heading">Kết quả phân tích</h3><p>Dự đoán trực tiếp từ mô hình</p></div>
        </div>
        <span className="result-live-tag"><span className="mini-dot"/> INFERENCE</span>
      </div>
      {result ? (
        <div className="result-content" aria-live="polite">
          <div className="result-primary">
            <div className="result-kicker"><Icon name="check-circle" size={16}/> CHỦ ĐỀ ĐƯỢC DỰ ĐOÁN</div>
            <h4>{displayDomain(result.label)}</h4>
            <span className="result-slug">{result.label}</span>
            <div className="result-primary-footer">
              <span><Icon name="target" size={16}/> Phân loại đơn nhãn</span>
              <span className="result-main-score">{scores.length && scores[0].label_id === result.label_id
                ? formatScore(scores[0].score, probability)
                : "#" + result.label_id}</span>
            </div>
          </div>

          <div className="ranking-header"><div><h4>Top {scores.length} chủ đề</h4><p>{probability ? "Xác suất dự đoán" : "Điểm quyết định của mô hình"}</p></div>
            <button className="copy-button" type="button" onClick={() => void copyResult()}><Icon name={copied ? "check" : "copy"} size={16}/>{copied ? "Đã sao chép" : "Sao chép"}</button>
          </div>
          <div className="score-list">{scores.map((item, i) => (
            <ScoreRow key={item.label_id} item={item} rank={i + 1} probability={probability} min={min} max={max}/>
          ))}</div>
          <div className="result-note"><Icon name="info" size={16}/><span>
            {probability
              ? "Các giá trị là xác suất do Logistic Regression ước lượng; không bảo đảm độ tin cậy trên mọi dữ liệu thực tế."
              : "Điểm LinearSVC là decision margin, không phải xác suất. Chiều dài thanh chỉ thể hiện tương quan giữa các điểm."}
          </span></div>
          <div className="model-footnote">Model: <strong>{result.model_name}</strong></div>
        </div>
      ) : (
        <div className={"result-empty" + (busy ? " is-analyzing" : "")} aria-live="polite">
          <div className="empty-visual">
            <div className="empty-ring ring-outer"/><div className="empty-ring ring-inner"/>
            <div className="empty-core"><Icon name={busy ? "activity" : "sparkles"} size={34}/></div>
            <span className="orbit-point orbit-a"/><span className="orbit-point orbit-b"/>
          </div>
          <span className="empty-status">{busy ? "ĐANG XỬ LÝ" : "SẴN SÀNG PHÂN TÍCH"}</span>
          <h4>{busy ? "AI đang đọc nội dung..." : "Insight bắt đầu từ văn bản"}</h4>
          <p>{busy ? "Đang nhận kết quả từ mô hình đã huấn luyện." : "Nhập văn bản ở bên trái và nhấn “Phân tích văn bản” để xem chủ đề dự đoán cùng các điểm xếp hạng."}</p>
          <div className="empty-steps"><span>01 · Nhập văn bản</span><Icon name="arrow-right" size={14}/><span>02 · AI phân tích</span><Icon name="arrow-right" size={14}/><span>03 · Kết quả</span></div>
        </div>
      )}
    </section>
  );
}
