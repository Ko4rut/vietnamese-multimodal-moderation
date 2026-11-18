"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getDomains, getModelStatus, predictText } from "../api/client";
import { EXAMPLES } from "../data/examples";
import { getMockPredictionForText, MOCK_PREDICTIONS } from "../data/mock-predictions";
import type { DomainLabel, InferenceMode, ModelStatus, PredictionView } from "../types";

export function usePrediction() {
  const [mode, setMode] = useState<InferenceMode>("mock");
  const [text, setText] = useState<string>(EXAMPLES[0].text);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [domains, setDomains] = useState<DomainLabel[]>([]);
  const [predictionView, setPredictionView] = useState<PredictionView | null>({
    source: "mock",
    data: MOCK_PREDICTIONS.technology,
  });
  const [chartRevision, setChartRevision] = useState<number>(0);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [predicting, setPredicting] = useState<boolean>(false);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const predictController = useRef<AbortController | null>(null);

  // The mock preview works fully offline; only live mode contacts FastAPI.
  useEffect(() => {
    if (mode !== "live") return;
    const controller = new AbortController();

    void getModelStatus(controller.signal)
      .then(setModelStatus)
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setStatusError(error instanceof Error ? error.message : "Không thể kiểm tra model.");
        }
      });
    void getDomains(controller.signal).then(setDomains).catch(() => {
      // When offline, the domain browser uses a labeled reference list.
    });

    return () => controller.abort();
  }, [mode]);

  // Stop outstanding prediction requests on unmount.
  useEffect(() => () => predictController.current?.abort(), []);

  const refreshStatus = useCallback(async () => {
    if (mode !== "live") return;
    setRefreshing(true);
    try {
      const next = await getModelStatus();
      setModelStatus(next);
      setStatusError(null);
      const latestDomains = await getDomains().catch(() => null);
      if (latestDomains) setDomains(latestDomains);
    } catch (error) {
      setModelStatus(null);
      setStatusError(error instanceof Error ? error.message : "Không thể kết nối backend.");
    } finally {
      setRefreshing(false);
    }
  }, [mode]);

  function changeMode(next: InferenceMode) {
    if (next === mode) return;
    predictController.current?.abort();
    setPredicting(false);
    setMode(next);
    setPredictionError(null);
    if (next === "mock") {
      const fixture = getMockPredictionForText(text);
      setPredictionView(fixture ? { source: "mock", data: fixture } : null);
      setChartRevision((revision) => revision + 1);
    } else {
      setPredictionView(null);
      setModelStatus(null);
      setStatusError(null);
    }
  }

  function updateText(value: string) {
    predictController.current?.abort();
    setPredicting(false);
    setText(value);
    setPredictionError(null);
    if (mode === "mock") {
      const fixture = getMockPredictionForText(value);
      setPredictionView(fixture ? { source: "mock", data: fixture } : null);
      setChartRevision((revision) => revision + 1);
    } else {
      setPredictionView(null);
    }
  }

  async function runPrediction() {
    const clean = text.trim();
    if (clean.length < 3 || clean.length > 10000) {
      setPredictionError("Vui lòng nhập văn bản từ 3 đến 10.000 ký tự.");
      return;
    }

    if (mode === "mock") {
      const fixture = getMockPredictionForText(text);
      if (!fixture) {
        setPredictionView(null);
        setPredictionError("Mock Preview chỉ có 4 ví dụ cố định. Hãy chọn một ví dụ bên trên hoặc chuyển sang AI thực để phân tích nội dung tự nhập.");
        return;
      }
      setPredictionError(null);
      setPredictionView({ source: "mock", data: fixture });
      setChartRevision((revision) => revision + 1);
      return;
    }

    if (modelStatus?.ready === false) {
      setPredictionError("Model chưa sẵn sàng. Hãy kết nối backend và tải model đã train.");
      return;
    }

    predictController.current?.abort();
    const controller = new AbortController();
    predictController.current = controller;
    setPredicting(true);
    setPredictionView(null);
    setPredictionError(null);
    try {
      const result = await predictText(clean, controller.signal);
      if (!controller.signal.aborted) {
        setPredictionView({ source: "live", data: result });
        setChartRevision((revision) => revision + 1);
      }
    } catch (error) {
      if (!controller.signal.aborted) {
        setPredictionError(error instanceof Error ? error.message : "Không thể phân tích văn bản.");
      }
    } finally {
      if (!controller.signal.aborted) setPredicting(false);
    }
  }

  return {
    mode, changeMode, text, updateText, modelStatus, statusError, domains,
    predictionView, chartRevision, predictionError, predicting,
    refreshing, refreshStatus, runPrediction,
  };
}
