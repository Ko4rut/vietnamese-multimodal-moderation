import { Icon } from "@/components/ui/icon";
import { ModelPicker } from "./model-picker";
import { classifierName, type AvailableModel, type ModelStatus } from "../types";

interface ModelStatusCardProps {
  status: ModelStatus | null;
  error: string | null;
  refreshing: boolean;
  onRefresh: () => void;
  models: AvailableModel[];
  selectedModelId: string | null;
  onSelectModel: (modelId: string) => void;
}

export function ModelStatusCard({
  status, error, refreshing, onRefresh,
  models, selectedModelId, onSelectModel,
}: ModelStatusCardProps) {
  const ready = status?.ready === true;
  const statusText = ready ? "Sẵn sàng suy luận" : error ? "Model chưa sẵn sàng" : status ? "Chưa có model" : "Đang kiểm tra";
  const metric = status?.validation_macro_f1;

  return (
    <section className="panel model-card" id="model" aria-labelledby="model-heading">
      <div className="panel-heading compact">
        <div className="section-title-group"><span className="panel-icon green"><Icon name="activity" size={19}/></span>
          <div><h3 id="model-heading">Trạng thái mô hình</h3><p>Thông tin từ FastAPI</p></div>
        </div>
        <button className="icon-button refresh-button" disabled={refreshing} onClick={onRefresh} type="button" aria-label="Kiểm tra lại trạng thái" title="Làm mới trạng thái">
          <Icon name="refresh" size={18} className={refreshing ? "spinning" : ""}/>
        </button>
      </div>
      <div className="model-status-line">
        <span className={"model-health " + (ready ? "model-healthy" : "model-not-ready")}><span className="status-dot"/>{statusText}</span>
        <span className="model-count">25 lĩnh vực</span>
      </div>
      <ModelPicker models={models} selectedModelId={selectedModelId} onSelectModel={onSelectModel}/>
      {(!ready || error) && <p className="model-message">{error ?? status?.message ?? "Đang chờ kết nối đến API..."}</p>}
      <div className="model-details">
        <div><span>Thuật toán</span><strong>{classifierName(status?.classifier ?? null)}</strong></div>
        <div><span>Validation Macro-F1</span><strong>{metric != null && Number.isFinite(metric) ? (metric * 100).toFixed(1) + "%" : "Chưa có"}</strong></div>
        <div className="model-name-row"><span>Experiment</span><strong title={status?.model_name ?? undefined}>{status?.model_name ?? "Chưa tải artifact"}</strong></div>
      </div>
      <p className="model-disclaimer"><Icon name="info" size={15}/>Nhãn huấn luyện VTSNLP được gán tự động từ nguồn, không phải ground truth được xác minh thủ công.</p>
    </section>
  );
}
