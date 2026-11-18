"use client";

import type { InferenceMode } from "../types";

interface PredictionModeSwitchProps {
  mode: InferenceMode;
  onChange: (mode: InferenceMode) => void;
}

export function PredictionModeSwitch({ mode, onChange }: PredictionModeSwitchProps) {
  return (
    <div className="mode-switch-wrap">
      <span className="mode-switch-label">NGUỒN KẾT QUẢ</span>
      <div className="prediction-mode-switch" role="group" aria-label="Chọn nguồn kết quả phân tích">
        <button type="button" className={mode === "mock" ? "active" : ""}
          aria-pressed={mode === "mock"} onClick={() => onChange("mock")}>
          Mock Preview
        </button>
        <button type="button" className={mode === "live" ? "active" : ""}
          aria-pressed={mode === "live"} onClick={() => onChange("live")}>
          AI thực
        </button>
      </div>
    </div>
  );
}
