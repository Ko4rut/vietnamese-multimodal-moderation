import { Icon } from "@/components/ui/icon";
import { EXAMPLES } from "../data/examples";
import type { InferenceMode } from "../types";

interface TextInputPanelProps {
  text: string;
  onTextChange: (text: string) => void;
  onAnalyze: () => void;
  busy: boolean;
  ready: boolean | null;
  error: string | null;
  mode: InferenceMode;
}

export function TextInputPanel({
  text, onTextChange, onAnalyze, busy, ready, error, mode,
}: TextInputPanelProps) {
  const mock = mode === "mock";
  const valid = text.trim().length >= 3 && text.length <= 10000;
  return (
    <section className="panel input-panel" aria-labelledby="input-heading">
      <div className="panel-heading">
        <div className="section-title-group">
          <span className="panel-icon blue"><Icon name="text" size={19} /></span>
          <div><h3 id="input-heading">Văn bản đầu vào</h3><p>{mock ? "Chọn đoạn mẫu để xem biểu đồ" : "Nhập nội dung cần phân tích"}</p></div>
        </div>
        <span className="small-tag">TIẾNG VIỆT</span>
      </div>

      <label className="field-label" htmlFor="analysis-text">Nội dung của bạn</label>
      <div className="textarea-frame">
        <textarea
          id="analysis-text"
          value={text}
          onChange={(event) => onTextChange(event.target.value)}
          placeholder="Nhập hoặc dán một đoạn văn bản tiếng Việt..."
          maxLength={10000}
          spellCheck={false}
          aria-describedby="text-help"
        />
        <div className="textarea-foot"><span>UTF-8 · Tối đa 10.000 ký tự</span><span>{text.length.toLocaleString("vi-VN")} / 10.000</span></div>
      </div>

      <div className="example-heading"><span>THỬ VỚI VÍ DỤ</span><span className="example-rule" /></div>
      <div className="example-list">
        {EXAMPLES.map((example) => (
          <button key={example.id} className="example-chip" type="button" onClick={() => onTextChange(example.text)}>
            <Icon name="sparkles" size={13} /> {example.label}
          </button>
        ))}
      </div>

      {error && <div className="error-banner" role="alert"><Icon name="alert-circle" size={17}/><span>{error}</span></div>}
      {!mock && ready === false && !error && (
        <div className="soft-notice"><Icon name="info" size={17}/><span>Backend chưa có model sẵn sàng. Hãy cấu hình API để dự đoán thực.</span></div>
      )}
      <div className="input-actions">
        <button className="text-button" type="button" onClick={() => onTextChange("")} disabled={!text || busy}>Xóa nội dung</button>
        <button className="primary-button" type="button" disabled={!valid || busy || (!mock && ready === false)} onClick={onAnalyze}>
          {busy ? <><span className="spinner" /> Đang phân tích...</> : <><Icon name="sparkles" size={18}/> {mock ? "Phát lại biểu đồ mẫu" : "Phân tích văn bản"} <Icon name="arrow-right" size={17}/></>}
        </button>
      </div>
      <p className="field-help" id="text-help"><Icon name="shield" size={14}/>{mock
        ? "Dữ liệu được định nghĩa sẵn trên frontend. Mock Preview không phân tích văn bản và không gửi yêu cầu tới API."
        : "Chế độ AI thực gửi văn bản tới FastAPI để suy luận, không huấn luyện model trong thao tác này."}</p>
    </section>
  );
}
