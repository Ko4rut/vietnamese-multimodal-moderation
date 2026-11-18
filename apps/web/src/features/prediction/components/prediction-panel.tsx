"use client";

import { useState } from "react";
import { Icon } from "@/components/ui/icon";
import { displayDomain } from "../data/domains";
import type { InferenceMode, PredictionView } from "../types";
import { formatPredictionScore, PredictionBarChart } from "./prediction-bar-chart";

interface PredictionPanelProps {
  view: PredictionView | null;
  mode: InferenceMode;
  busy: boolean;
  chartRevision: number;
}

export function PredictionPanel({ view, mode, busy, chartRevision }: PredictionPanelProps) {
  const [lastCopied, setLastCopied] = useState<string | null>(null);
  const result = view?.data ?? null;
  const mock = mode === "mock";
  const scoreKind = result?.score_kind ?? "probability";
  const probability = scoreKind === "probability";
  const scores = result?.top_predictions ?? [];
  const winner = scores.find((item) => item.label_id === result?.label_id);
  const copyKey = result
    ? JSON.stringify([view?.source, result.model_name, result.label_id, result.score_kind, scores])
    : null;

  async function copyResult() {
    if (!result || !copyKey) return;
    const summary = [
      "Nguồn: " + (mock ? "MOCK DATA (giá trị do frontend đặt sẵn)" : "FastAPI (dự đoán thật)"),
      "Chủ đề: " + displayDomain(result.label) + " (" + result.label + ")",
      "Mô hình: " + result.model_name,
      "Kiểu điểm: " + (probability ? "Xác suất" : "Decision margin"),
      ...scores.map((item, index) =>
        String(index + 1) + ". " + item.label + ": " +
        formatPredictionScore(item.score, scoreKind),
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
          <div>
            <h3 id="result-heading">Kết quả phân tích</h3>
            <p>{mock ? "Minh họa biểu đồ bằng dữ liệu mẫu" : "Dự đoán trực tiếp từ mô hình"}</p>
          </div>
        </div>
        <span className={"result-live-tag" + (mock ? " result-mock-tag" : "")}>
          <span className="mini-dot"/>{mock ? "MOCK DATA" : "INFERENCE"}
        </span>
      </div>
      {result ? (
        <div className="result-content" aria-live="polite">
          {mock && (
            <div className="mock-result-notice" role="note">
              <Icon name="info" size={17}/>
              <span>Chế độ xem thử: toàn bộ nhãn và điểm số bên dưới là dữ liệu cố định của frontend, không phải dự đoán AI.</span>
            </div>
          )}
          <div className="result-primary">
            <div className="result-kicker">
              <Icon name={mock ? "info" : "check-circle"} size={16}/>
              {mock ? "NHÃN MINH HỌA" : "CHỦ ĐỀ ĐƯỢC DỰ ĐOÁN"}
            </div>
            <h4>{displayDomain(result.label)}</h4>
            <span className="result-slug">{result.label}</span>
            <div className="result-primary-footer">
              <span><Icon name="target" size={16}/>{mock ? "Kết quả mẫu · Không suy luận" : "Phân loại đơn nhãn"}</span>
              <span className="result-main-score">
                {winner ? formatPredictionScore(winner.score, scoreKind) : "#" + result.label_id}
              </span>
            </div>
          </div>

          <div className="ranking-header">
            <div>
              <h4>Biểu đồ {mock ? "mẫu" : "dự đoán"} · Top {scores.length} nhãn</h4>
              <p>{probability ? (mock ? "Giá trị xác suất giả định" : "Xác suất của các chủ đề") : "Điểm quyết định theo từng nhãn"}</p>
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
              scoreKind={scoreKind}
              selectedLabelId={result.label_id}
            />
          )}

          <div className="result-note"><Icon name="info" size={16}/><span>
            {mock
              ? "Dữ liệu mô phỏng giao diện: không sử dụng các điểm mẫu để đánh giá mô hình hoặc diễn giải như kết quả thực tế."
              : probability
                ? "Các giá trị là xác suất do Logistic Regression ước lượng; không bảo đảm độ tin cậy trên mọi dữ liệu thực tế."
                : "LinearSVC trả về decision margin, không phải xác suất. Không diễn giải điểm hoặc chiều cao cột thành độ tin cậy phần trăm."}
          </span></div>
          <div className="model-footnote">Nguồn: <strong>{mock ? "Frontend mock fixture" : result.model_name}</strong></div>
        </div>
      ) : (
        <div className={"result-empty" + (busy ? " is-analyzing" : "")} aria-live="polite">
          <div className="prediction-chart-preview" aria-hidden="true">
            {[32, 54, 77, 48, 62].map((height, index) => (
              <span className="prediction-chart-preview-bar" key={index}
                style={{ height: height + "%", animationDelay: index * 120 + "ms" }} />
            ))}
          </div>
          <span className="empty-status">{busy ? "ĐANG XỬ LÝ" : mock ? "CHỌN VÍ DỤ MOCK" : "CHỜ KẾT QUẢ"}</span>
          <h4>{busy ? "AI đang đọc nội dung..." : mock ? "Chọn mẫu để xem biểu đồ" : "Kết quả sẽ hiện thành biểu đồ"}</h4>
          <p>{busy ? "Đang nhận điểm số từ mô hình đã huấn luyện."
            : mock ? "Mock Preview có 4 ví dụ cố định. Chọn một ví dụ bên trái để hiển thị các cột minh họa."
              : "Nhập văn bản và nhấn “Phân tích văn bản”. Các cột sẽ bật lên khi mô hình trả kết quả thật."}</p>
          <div className="empty-steps">
            <span>01 · Nhập văn bản</span><Icon name="arrow-right" size={14}/>
            <span>02 · {mock ? "Mock data" : "AI phân tích"}</span><Icon name="arrow-right" size={14}/>
            <span>03 · Biểu đồ</span>
          </div>
        </div>
      )}
    </section>
  );
}
