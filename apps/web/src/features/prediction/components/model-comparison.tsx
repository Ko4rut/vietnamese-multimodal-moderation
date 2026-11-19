"use client";

import { Icon } from "@/components/ui/icon";
import { classifierName, type AvailableModel } from "../types";
import { useModelComparison } from "../hooks/use-model-comparison";
import { ComparisonResultCard } from "./comparison-result-card";

interface ModelComparisonProps {
  models: AvailableModel[];
  text: string;
}

export function ModelComparison({ models, text }: ModelComparisonProps) {
  const {
    selectedIds, allowed, busy, results, error, toggleModel, runComparison,
  } = useModelComparison(models, text);
  const successful = results?.filter((item) => item.status === "ok" && item.prediction) ?? [];
  const labels = new Set(successful.map((item) => item.prediction?.label_id));

  return (
    <section className="panel model-comparison" id="compare" aria-labelledby="compare-heading">
      <div className="panel-heading">
        <div className="section-title-group">
          <span className="panel-icon purple"><Icon name="layers" size={20}/></span>
          <div>
            <h3 id="compare-heading">Multi-Model Comparison</h3>
            <p>Một văn bản đầu vào · So sánh kết quả thực từ 2–4 mô hình</p>
          </div>
        </div>
        <span className="small-tag">MODEL BENCHMARK</span>
      </div>

      <div className="comparison-body">
        <div className="comparison-controls">
          <div className="comparison-instructions">
            <strong>Chọn các mô hình cần so sánh</strong>
            <span>{selectedIds.length}/4 đã chọn · {models.length} model trong registry</span>
          </div>
          <div className="comparison-choices" role="group" aria-label="Chọn từ hai đến bốn mô hình">
            {models.map((model) => {
              const checked = selectedIds.includes(model.id);
              return (
                <label key={model.id} className={"comparison-choice" + (checked ? " is-selected" : "")}>
                  <input
                    type="checkbox"
                    checked={checked}
                    disabled={!checked && selectedIds.length >= 4}
                    onChange={() => toggleModel(model.id)}
                  />
                  <span className="comparison-choice-content">
                    <strong title={model.id}>{model.id}</strong>
                    <small>{classifierName(model.classifier)}</small>
                  </span>
                </label>
              );
            })}
            {models.length === 0 && (
              <p className="comparison-empty">Chưa có model trong registry. Đặt artifact vào thư mục models/ và làm mới trang.</p>
            )}
          </div>
          <div className="comparison-actions">
            <p>Chọn ít nhất 2 model. Kết quả giữ nguyên loại điểm riêng của mỗi thuật toán.</p>
            <button
              type="button"
              className="primary-button"
              disabled={!allowed || busy}
              onClick={() => void runComparison()}
            >
              {busy ? <><span className="spinner"/> Đang so sánh...</> : <><Icon name="chart" size={17}/> So sánh {selectedIds.length} model</>}
            </button>
          </div>
          {error && <p className="comparison-error" role="alert">{error}</p>}
        </div>

        {results && (
          <div className="comparison-output" aria-live="polite">
            <div className="comparison-summary">
              <strong>{successful.length}/{results.length} model trả kết quả</strong>
              <span>{successful.length < 2
                ? "Chưa đủ dự đoán hợp lệ để đối chiếu"
                : labels.size === 1
                  ? "Các model thành công dự đoán cùng một nhãn"
                  : "Các model thành công có nhãn dự đoán khác nhau"}</span>
            </div>
            <div className="comparison-grid">
              {results.map((result) => (
                <ComparisonResultCard
                  key={result.model_id}
                  result={result}
                  model={models.find((model) => model.id === result.model_id)}
                />
              ))}
            </div>
            <p className="comparison-disclaimer">
              Không so sánh trực tiếp giá trị xác suất của Logistic Regression
              với decision margin của LinearSVC; chỉ đối chiếu nhãn và thứ hạng.
            </p>
          </div>
        )}
      </div>
    </section>
  );
}
