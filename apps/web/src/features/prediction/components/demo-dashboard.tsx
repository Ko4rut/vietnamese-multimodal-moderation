"use client";

import { useState } from "react";
import { Sidebar } from "@/components/layout/sidebar";
import { Topbar } from "@/components/layout/topbar";
import { Icon } from "@/components/ui/icon";
import { DomainBrowser } from "./domain-browser";
import { ModelStatusCard } from "./model-status-card";
import { PredictionModeSwitch } from "./prediction-mode-switch";
import { PredictionPanel } from "./prediction-panel";
import { TextInputPanel } from "./text-input-panel";
import { usePrediction } from "../hooks/use-prediction";

export function DemoDashboard() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const {
    mode, changeMode, text, updateText, modelStatus, statusError, models,
    selectedModelId, selectModel, domains,
    predictionView, chartRevision, predictionError, predicting,
    refreshing, refreshStatus, runPrediction,
  } = usePrediction();
  const ready = mode === "live" ? modelStatus?.ready ?? (statusError ? false : null) : null;
  const preview = mode === "mock";

  return (
    <div className="app-shell">
      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      <div className="main-shell">
        <Topbar onOpenMenu={() => setSidebarOpen(true)} ready={ready} preview={preview} />
        <main className="dashboard">
          <section className="hero" id="playground" aria-labelledby="hero-title">
            <div className="hero-grid-pattern"/>
            <div className="hero-copy">
              <span className="hero-eyebrow"><span className="hero-eyebrow-pulse"/> VIETNAMESE NLP · AI PLAYGROUND</span>
              <h1 id="hero-title">Thấu hiểu văn bản.<br/><span>Khám phá bằng AI.</span></h1>
              <p>Biến văn bản tiếng Việt thành thông tin có cấu trúc. Trải nghiệm mô hình AI phân loại chủ đề qua 25 lĩnh vực, ngay tại đây.</p>
              <div className="hero-pills"><span><Icon name="check-circle" size={15}/> 25 lĩnh vực</span><span><Icon name="zap" size={15}/> {preview ? "Mock UI Preview" : "Real model inference"}</span><span><Icon name="shield" size={15}/> Phân loại đơn nhãn</span></div>
            </div>
            <div className="hero-art" aria-hidden="true">
              <div className="art-glow"/>
              <div className="art-circle art-circle-outer"/>
              <div className="art-circle art-circle-middle"/>
              <div className="art-circle art-circle-inner"/>
              <div className="art-core"><Icon name="brain" size={45}/></div>
              <span className="art-satellite art-satellite-one"><Icon name="text" size={19}/></span>
              <span className="art-satellite art-satellite-two"><Icon name="chart" size={20}/></span>
              <span className="art-satellite art-satellite-three"><Icon name="layers" size={19}/></span>
            </div>
          </section>

          <div className="section-bar playground-heading">
            <div><span className="section-eyebrow">TRẢI NGHIỆM / 01</span><h2>AI Playground <span className="heading-highlight">· Trực quan hóa kết quả</span></h2>
              <p>{preview ? "Đang xem dữ liệu mẫu để thử biểu đồ trước khi kết nối model." : "Nhập văn bản để mô hình thật nhận diện chủ đề."}</p></div>
            <PredictionModeSwitch mode={mode} onChange={changeMode} />
          </div>

          <div className="playground-grid">
            <TextInputPanel
              text={text} onTextChange={updateText} onAnalyze={() => void runPrediction()}
              busy={predicting} ready={ready} error={predictionError} mode={mode}
            />
            <PredictionPanel mode={mode} view={predictionView} busy={predicting} chartRevision={chartRevision} />
          </div>

          <div className="insight-grid">
            <ModelStatusCard
              mode={mode} status={modelStatus} error={statusError}
              refreshing={refreshing} onRefresh={() => void refreshStatus()}
              models={models} selectedModelId={selectedModelId} onSelectModel={selectModel}
            />
            <section className="panel how-card" aria-labelledby="how-heading">
              <div className="panel-heading compact"><div className="section-title-group"><span className="panel-icon orange"><Icon name="layers" size={19}/></span>
                <div><h3 id="how-heading">Quy trình phân tích</h3><p>Từ văn bản đến chủ đề</p></div></div>
              </div>
              <div className="flow-steps">
                <div className="flow-step"><span className="flow-step-number">01</span><div><strong>Tiếp nhận văn bản</strong><p>Chuẩn bị nội dung đầu vào bằng tiếng Việt.</p></div><Icon name="text" size={19}/></div>
                <div className="flow-connector"/>
                <div className="flow-step"><span className="flow-step-number">02</span><div><strong>{preview ? "Mock fixture" : "ML Inference"}</strong><p>{preview ? "Lấy điểm minh họa được định nghĩa sẵn." : "TF-IDF và classifier đã được huấn luyện."}</p></div><Icon name="brain" size={19}/></div>
                <div className="flow-connector"/>
                <div className="flow-step"><span className="flow-step-number">03</span><div><strong>Trực quan hóa</strong><p>Hiển thị một nhãn và biểu đồ cột Top 5.</p></div><Icon name="target" size={19}/></div>
              </div>
            </section>
          </div>

          <DomainBrowser domains={domains}/>
          <footer className="main-footer">
            <div><span className="footer-mark"><Icon name="sparkles" size={15}/></span><strong>viettopic.ai</strong><span>© 2026 · Research & AI Demo</span></div>
            <a href="https://github.com/Ko4rut/vietnamese-multimodal-moderation" target="_blank" rel="noopener noreferrer">Source code <Icon name="external" size={14}/></a>
          </footer>
        </main>
      </div>
    </div>
  );
}
