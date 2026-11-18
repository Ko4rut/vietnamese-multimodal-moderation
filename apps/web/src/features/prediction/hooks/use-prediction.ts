"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { getDomains, getModelCatalog, getModelStatus, predictText } from "../api/client";
import { EXAMPLES } from "../data/examples";
import { getMockPredictionForText, MOCK_PREDICTIONS } from "../data/mock-predictions";
import type {
  AvailableModel,
  DomainLabel,
  InferenceMode,
  ModelStatus,
  PredictionView,
} from "../types";

export function usePrediction() {
  const [mode, setMode] = useState<InferenceMode>("mock");
  const [text, setText] = useState<string>(EXAMPLES[0].text);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [models, setModels] = useState<AvailableModel[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
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

  // No network requests are made in Mock Preview mode.
  useEffect(() => {
    if (mode !== "live") return;
    const controller = new AbortController();

    void getModelCatalog(controller.signal)
      .then((catalog) => {
        if (controller.signal.aborted) return;
        setModels(catalog.models);
        setSelectedModelId((current) => (
          current && catalog.models.some((item) => item.id === current)
            ? current
            : catalog.default_model_id
        ));
        if (catalog.models.length === 0) {
          setStatusError("FastAPI chưa tìm thấy experiment nào. Kiểm tra VTC_MODELS_ROOT.");
        } else {
          setStatusError(null);
        }
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setStatusError(error instanceof Error ? error.message : "Không lấy được danh sách model.");
        }
      });

    void getDomains(controller.signal)
      .then((labels) => { if (!controller.signal.aborted) setDomains(labels); })
      .catch(() => { /* Use the labeled reference list while offline. */ });

    return () => controller.abort();
  }, [mode]);

  // Readiness always follows the specific experiment selected by the user.
  useEffect(() => {
    if (mode !== "live" || !selectedModelId) return;
    const controller = new AbortController();
    void getModelStatus(selectedModelId, controller.signal)
      .then((status) => {
        if (controller.signal.aborted) return;
        setModelStatus(status);
        setStatusError(null);
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setModelStatus(null);
          setStatusError(error instanceof Error ? error.message : "Không kiểm tra được model.");
        }
      });
    return () => controller.abort();
  }, [mode, selectedModelId]);

  useEffect(() => () => predictController.current?.abort(), []);

  const refreshStatus = useCallback(async () => {
    if (mode !== "live") return;
    setRefreshing(true);
    try {
      const catalog = await getModelCatalog();
      setModels(catalog.models);
      const selected = selectedModelId && catalog.models.some((m) => m.id === selectedModelId)
        ? selectedModelId : catalog.default_model_id;
      setSelectedModelId(selected);
      const nextStatus = await getModelStatus(selected);
      setModelStatus(nextStatus);
      setStatusError(null);
      const latestDomains = await getDomains().catch(() => null);
      if (latestDomains) setDomains(latestDomains);
    } catch (error) {
      setModelStatus(null);
      setStatusError(error instanceof Error ? error.message : "Không thể kết nối backend.");
    } finally {
      setRefreshing(false);
    }
  }, [mode, selectedModelId]);

  function selectModel(id: string) {
    if (id === selectedModelId || !models.some((item) => item.id === id)) return;
    predictController.current?.abort();
    setPredicting(false);
    setPredictionView(null);
    setPredictionError(null);
    setStatusError(null);
    setModelStatus(null);
    setSelectedModelId(id);
  }

  function changeMode(next: InferenceMode) {
    if (next === mode) return;
    predictController.current?.abort();
    setPredicting(false);
    setMode(next);
    setPredictionError(null);
    setModelStatus(null);
    setStatusError(null);
    if (next === "mock") {
      const fixture = getMockPredictionForText(text);
      setPredictionView(fixture ? { source: "mock", data: fixture } : null);
      setChartRevision((revision) => revision + 1);
    } else {
      setModels([]);
      setSelectedModelId(null);
      setPredictionView(null);
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
        setPredictionError(
          "Mock Preview chỉ có 4 ví dụ cố định. Chọn ví dụ hoặc chuyển sang AI thực."
        );
        return;
      }
      setPredictionError(null);
      setPredictionView({ source: "mock", data: fixture });
      setChartRevision((revision) => revision + 1);
      return;
    }

    if (!selectedModelId || modelStatus?.ready === false) {
      setPredictionError("Model chưa sẵn sàng. Hãy cấu hình và chọn experiment đã train.");
      return;
    }

    predictController.current?.abort();
    const controller = new AbortController();
    predictController.current = controller;
    setPredicting(true);
    setPredictionView(null);
    setPredictionError(null);
    try {
      const result = await predictText(clean, selectedModelId, controller.signal);
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
    mode, changeMode, text, updateText, modelStatus, statusError,
    models, selectedModelId, selectModel, domains, predictionView, chartRevision,
    predictionError, predicting, refreshing, refreshStatus, runPrediction,
  };
}
