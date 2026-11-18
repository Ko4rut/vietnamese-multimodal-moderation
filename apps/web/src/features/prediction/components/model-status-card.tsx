import { Icon } from "@/components/ui/icon";
import { classifierName, type InferenceMode, type ModelStatus } from "../types";

interface ModelStatusCardProps {
  mode: InferenceMode;
  status: ModelStatus | null;
  error: string | null;
  refreshing: boolean;
  onRefresh: () => void;
}

export function ModelStatusCard({
  mode, status, error, refreshing, onRefresh,
}: ModelStatusCardProps) {
  const mock = mode === "mock";
  const ready = status?.ready === true;
  const statusText = mock
    ? "Đang xem dữ liệu mẫu"
    : ready ? "Sẵn sàng suy luận" : error ? "Không kết nối API" : status ? "Chưa có model" : "Đang kiểm tra";
  const metric = status?.validation_macro_f1;

  return (
    <section className="panel model-card" id="model" aria-labelledby="model-heading">
      <div className="panel-heading compact">
        <div className="section-title-group"><span className="panel-icon green"><Icon name="activity" size={19}/></span>
          <div><h3 id="model-heading">{mock ? "Trạng thái bản demo" : "Trạng thái mô hình"}</h3><p>{mock ? "Mock Preview · Frontend" : "Thông tin từ FastAPI"}</p></div>
        </div>
        {!mock && <button className="icon-button refresh-button" disabled={refreshing} onClick={onRefresh} type="button" aria-label="Kiểm tra lại trạng thái" title="Làm mới trạng thái">
          <Icon name="refresh" size={18} className={refreshing ? "spinning" : ""}/>
        </button>}
      </div>
      <div className="model-status-line">
        <span className={"model-health " + (mock || ready ? "model-healthy" : "model-not-ready")}><span className="status-dot"/>{statusText}</span>
        <span className="model-count">25 lĩnh vực</span>
      </div>
      {!mock && (!ready || error) && <p className="model-message">{error ?? status?.message ?? "Đang chờ kết nối đến API..."}</p>}
      {mock && <p className="model-message">Giao diện đang hiển thị các điểm số minh họa cố định, không phải kết quả từ mô hình đã huấn luyện.</p>}
      <div className="model-details">
        <div><span>Thuật toán</span><strong>{mock ? "Logistic Regression (minh họa)" : classifierName(status?.classifier ?? null)}</strong></div>
        <div><span>Validation Macro-F1</span><strong>{mock ? "Không áp dụng" : metric != null && Number.isFinite(metric) ? (metric * 100).toFixed(1) + "%" : "Chưa có"}</strong></div>
        <div className="model-name-row"><span>Experiment</span><strong title={mock ? "frontend_mock" : status?.model_name ?? undefined}>{mock ? "frontend_mock" : status?.model_name ?? "Chưa tải artifact"}</strong></div>
      </div>
      <p className="model-disclaimer"><Icon name="info" size={15}/>{mock
        ? "Chuyển sang AI thực để gọi model đã train. Không dùng các điểm mẫu làm số liệu đánh giá."
        : "Nhãn huấn luyện VTSNLP được gán tự động từ nguồn, không phải ground truth được xác minh thủ công."}</p>
    </section>
  );
}
