import { useCallback, useEffect, useState } from "react";
import {
  Activity, ArrowRight, BarChart3, BookOpen, BrainCircuit, CheckCircle2,
  ChevronRight, CircleAlert, FileText, Github, Info, Layers3,
  LoaderCircle, RefreshCw, RotateCcw, ShieldCheck, Sparkles, WandSparkles,
} from "lucide-react";
import { getStatus, predictText, type ModelStatus, type Prediction } from "./api";
import { classifierName, translatedLabel } from "./labels";

const EXAMPLES = [
  {
    label: "Thể thao",
    text: "Đội tuyển bóng đá Việt Nam chuẩn bị cho trận đấu quan trọng cuối tuần này. Huấn luyện viên công bố đội hình với nhiều cầu thủ trẻ.",
  },
  {
    label: "Công nghệ",
    text: "Các nhà nghiên cứu giới thiệu bộ vi xử lý mới giúp máy tính xử lý dữ liệu nhanh hơn và tiết kiệm điện năng.",
  },
  {
    label: "Du lịch",
    text: "Du khách đến Đà Nẵng có thể tham quan bãi biển Mỹ Khê, khám phá ẩm thực địa phương và trải nghiệm nhiều điểm đến hấp dẫn.",
  },
];
const MAX_LENGTH = 10000;
const REPO_URL = "https://github.com/Ko4rut/vietnamese-multimodal-moderation";

function MetricCard({ icon, value, label }: {
  icon: React.ReactNode;
  value: string;
  label: string;
}) {
  return (
    <div className="metric-card">
      <div className="metric-icon">{icon}</div>
      <div><strong>{value}</strong><span>{label}</span></div>
    </div>
  );
}

function PredictionResults({ prediction }: { prediction: Prediction }) {
  const isProbability = prediction.score_kind === "probability";
  const scores = prediction.top_predictions.map((item) => item.score);
  const minScore = Math.min(...scores);
  const maxScore = Math.max(...scores);
  const barWidth = (score: number) => isProbability
    ? `${Math.max(score * 100, 2)}%`
    : `${14 + 86 * ((score - minScore) / (maxScore - minScore || 1))}%`;

  return (
    <div className="result-content" aria-live="polite">
      <div className="result-check"><CheckCircle2 size={17} /> Phân tích hoàn tất</div>
      <div className="result-headline">Chủ đề được dự đoán</div>
      <div className="winning-topic">{translatedLabel(prediction.label)}</div>
      <div className="canonical-label">{prediction.label} · ID {prediction.label_id}</div>
      <div className="results-separator" />
      <div className="ranking-title">
        <span>Top {prediction.top_predictions.length} chủ đề</span>
        <span>{isProbability ? "Xác suất ước tính" : "Decision margin"}</span>
      </div>
      <div className="ranking-list">
        {prediction.top_predictions.map((item, index) => (
          <div key={item.label_id} className="ranking-row">
            <div className="ranking-name">
              <span className="rank-number">{String(index + 1).padStart(2, "0")}</span>
              <span title={item.label}>{translatedLabel(item.label)}</span>
              <strong>{isProbability ? `${(item.score * 100).toFixed(1)}%` : item.score.toFixed(3)}</strong>
            </div>
            <div className="bar-track" aria-hidden="true">
              <div className={index === 0 ? "bar-fill top" : "bar-fill"} style={{ width: barWidth(item.score) }} />
            </div>
          </div>
        ))}
      </div>
      <p className="score-note"><Info size={15} />
        {isProbability
          ? "Đây là xác suất do Logistic Regression ước tính, không đảm bảo độ chính xác thực tế."
          : "LinearSVC trả điểm quyết định (margin), không phải xác suất hay phần trăm độ tin cậy."}
      </p>
    </div>
  );
}

export default function App() {
  const [text, setText] = useState("");
  const [status, setStatus] = useState<ModelStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [loadingStatus, setLoadingStatus] = useState(true);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [predicting, setPredicting] = useState(false);

  const refreshStatus = useCallback(async () => {
    setLoadingStatus(true);
    try {
      const response = await getStatus();
      setStatus(response);
      setStatusError(null);
    } catch {
      setStatus(null);
      setStatusError("Không kết nối được API. Hãy khởi động FastAPI tại cổng 8000.");
    } finally {
      setLoadingStatus(false);
    }
  }, []);

  useEffect(() => { void refreshStatus(); }, [refreshStatus]);

  const canPredict = status?.ready === true && text.trim().length >= 3
    && text.length <= MAX_LENGTH && !predicting;

  const analyse = async () => {
    if (!canPredict) return;
    setPrediction(null);
    setPredictionError(null);
    setPredicting(true);
    try {
      setPrediction(await predictText(text.trim()));
    } catch (error) {
      setPredictionError(error instanceof Error ? error.message : "Dự đoán không thành công.");
    } finally {
      setPredicting(false);
    }
  };

  const reset = () => {
    setText("");
    setPrediction(null);
    setPredictionError(null);
  };

  const ready = status?.ready === true;
  const modelStatusLabel = loadingStatus ? "Đang kiểm tra..." :
    statusError ? "API offline" : ready ? "Model sẵn sàng" : "Chưa kết nối model";

  return (
    <div className="app-shell">
      <header className="site-header">
        <div className="container header-content">
          <a className="brand" href="/" aria-label="VietTopic trang chủ">
            <span className="brand-symbol"><Layers3 size={22} strokeWidth={2.2}/></span>
            <span>Viet<span className="brand-accent">Topic</span><small>LAB</small></span>
          </a>
          <div className="header-right">
            <a className="github-link" href={REPO_URL} target="_blank" rel="noreferrer">
              <Github size={17} /><span>GitHub</span>
            </a>
            <div className={`status-badge ${ready ? "online" : "offline"}`} role="status">
              <span className="status-dot" />{modelStatusLabel}
            </div>
          </div>
        </div>
      </header>

      <main className="container">
        <section className="hero">
          <div className="eyebrow"><Sparkles size={15}/> NLP MODEL PLAYGROUND <span className="eyebrow-line" /> TEXT CLASSIFICATION</div>
          <h1>Khám phá chủ đề<br/><span>ẩn trong văn bản.</span></h1>
          <p className="hero-description">Trải nghiệm mô hình phân loại văn bản tiếng Việt với 25 lĩnh vực chủ đề. Nhập nội dung bất kỳ để xem model dự đoán trực tiếp.</p>
          <div className="metrics">
            <MetricCard icon={<Layers3 size={21}/>} value={String(status?.labels_count ?? 25)} label="Nhóm chủ đề" />
            <MetricCard icon={<BrainCircuit size={21}/>} value="TF-IDF" label="Trích xuất đặc trưng" />
            <MetricCard icon={<Activity size={21}/>} value="Single-label" label="Bài toán phân loại" />
          </div>
        </section>

        <div className="section-heading"><div><span className="section-index">01 /</span><h2>Không gian thử nghiệm</h2></div><span className="section-subtitle">MODEL INFERENCE</span></div>

        <section className="workspace">
          <article className="panel editor-panel">
            <div className="panel-heading">
              <div className="panel-icon violet"><FileText size={19}/></div>
              <div><h3>Văn bản đầu vào</h3><p>Nhập đoạn văn tiếng Việt cần phân loại</p></div>
              <button className="icon-button reset-button" type="button" onClick={reset} disabled={!text && !prediction} aria-label="Xóa nội dung" title="Xóa nội dung"><RotateCcw size={18}/></button>
            </div>
            <label htmlFor="text-input" className="input-label">Nội dung văn bản</label>
            <textarea
              id="text-input"
              value={text}
              onChange={(event) => {
                setText(event.target.value);
                setPrediction(null);
                setPredictionError(null);
              }}
              placeholder="Ví dụ: Trong trận đấu tối qua, đội tuyển Việt Nam đã giành chiến thắng..."
              maxLength={MAX_LENGTH}
              spellCheck={false}
              rows={9}
            />
            <div className="input-foot"><span>Dữ liệu không được lưu bởi ứng dụng demo này.</span><span>{text.length.toLocaleString("vi-VN")} / {MAX_LENGTH.toLocaleString("vi-VN")}</span></div>
            <div className="examples">
              <div className="examples-title"><WandSparkles size={16}/> Thử với văn bản mẫu</div>
              <div className="example-list">
                {EXAMPLES.map((example) => (
                  <button key={example.label} type="button" className="example-chip" onClick={() => {
                    setText(example.text);
                    setPrediction(null);
                    setPredictionError(null);
                  }}>{example.label}<ChevronRight size={15}/></button>
                ))}
              </div>
            </div>
            <button type="button" className="predict-button" onClick={() => { void analyse(); }} disabled={!canPredict}>
              {predicting ? <><LoaderCircle size={19} className="spin"/>Đang phân tích...</> : <><Sparkles size={19}/> Phân loại văn bản <ArrowRight size={18}/></>}
            </button>
            {!ready && !loadingStatus && (
              <p className="editor-warning"><CircleAlert size={15}/> Cần kết nối model thật để kích hoạt dự đoán.</p>
            )}
          </article>

          <article className="panel output-panel">
            <div className="panel-heading">
              <div className="panel-icon mint"><BarChart3 size={20}/></div>
              <div><h3>Kết quả dự đoán</h3><p>Dữ liệu trả về trực tiếp từ mô hình</p></div>
              <span className="live-tag"><span/> LIVE OUTPUT</span>
            </div>
            {prediction ? (
              <PredictionResults prediction={prediction}/>
            ) : predictionError ? (
              <div className="empty-result error-state">
                <div className="empty-symbol"><CircleAlert size={28}/></div>
                <h4>Không thể phân loại</h4><p>{predictionError}</p>
                <button className="text-button" type="button" onClick={() => { void refreshStatus(); }}>Kiểm tra kết nối <RefreshCw size={15}/></button>
              </div>
            ) : !ready && !loadingStatus ? (
              <div className="empty-result">
                <div className="empty-symbol"><CircleAlert size={29}/></div>
                <h4>Model chưa sẵn sàng</h4>
                <p>{statusError ?? status?.message ?? "Hãy kiểm tra kết nối với Backend."}</p>
                <button className="text-button" type="button" onClick={() => { void refreshStatus(); }}>Kiểm tra lại <RefreshCw size={15}/></button>
              </div>
            ) : (
              <div className="empty-result">
                <div className="empty-symbol">{predicting ? <LoaderCircle className="spin" size={29}/> : <BrainCircuit size={31}/>}</div>
                <h4>{predicting ? "Model đang xử lý..." : "Chưa có kết quả phân tích"}</h4>
                <p>{predicting ? "Đang chạy inference với văn bản của bạn." : "Nhập nội dung vào ô bên trái và nhấn “Phân loại văn bản” để xem kết quả."}</p>
                <div className="empty-steps"><span>01 · Nhập văn bản</span><ArrowRight size={14}/><span>02 · Phân loại</span><ArrowRight size={14}/><span>03 · Xem kết quả</span></div>
              </div>
            )}
          </article>
        </section>

        <div className="section-heading details-heading"><div><span className="section-index">02 /</span><h2>Thông tin & lưu ý</h2></div><span className="section-subtitle">MODEL TRANSPARENCY</span></div>
        <section className="detail-grid">
          <div className="detail-card">
            <div className="detail-title"><BrainCircuit size={19}/> Model được sử dụng</div>
            <div className="detail-table">
              <div><span>Tên experiment</span><strong>{status?.model_name ?? "Chưa tải"}</strong></div>
              <div><span>Thuật toán</span><strong>{classifierName(status?.classifier ?? null)}</strong></div>
              <div><span>Validation Macro-F1</span><strong>{ready && status?.validation_macro_f1 != null ? (status.validation_macro_f1 * 100).toFixed(2) + "%" : "Chưa có"}</strong></div>
              <div><span>Kiến trúc</span><strong>TF-IDF + Linear classifier</strong></div>
            </div>
          </div>
          <div className="detail-card caution-card">
            <div className="detail-title"><ShieldCheck size={19}/> Giới hạn của mô hình</div>
            <p>Mô hình đang giải quyết bài toán <strong>phân loại một nhãn trong 25 chủ đề</strong>, không phải kiểm duyệt nội dung độc hại hoặc phân tích đa phương thức.</p>
            <div className="disclaimer"><Info size={18}/><span>Nhãn gốc do nguồn dữ liệu tự động gán, chưa được kiểm định độc lập. Kết quả dự đoán chỉ mang tính tham khảo.</span></div>
          </div>
        </section>
      </main>

      <footer className="footer"><div className="container footer-inner"><div className="footer-brand"><BookOpen size={17}/> Vietnamese Topic Classification · Research Demo</div><a href={REPO_URL} target="_blank" rel="noreferrer">Source on GitHub <ArrowRight size={16}/></a></div></footer>
    </div>
  );
}
