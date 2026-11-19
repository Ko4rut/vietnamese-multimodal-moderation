"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { getDomains, getModelCatalog, getModelStatus, predictText } from "../api/client";
import type { AvailableModel, DomainLabel, ModelStatus, Prediction } from "../types";

export function usePrediction() {
  const [text, setText] = useState<string>("");
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [models, setModels] = useState<AvailableModel[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [domains, setDomains] = useState<DomainLabel[]>([]);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [chartRevision, setChartRevision] = useState(0);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const predictController = useRef<AbortController | null>(null);

  // Only real inference: fetch the registry and labels as soon as the page mounts.
  useEffect(() => {
    const controller = new AbortController();
    void getModelCatalog(controller.signal)
      .then((catalog) => {
        if (controller.signal.aborted) return;
        setModels(catalog.models);
        setSelectedModelId((previous) =>
          previous && catalog.models.some((model) => model.id === previous)
            ? previous : catalog.default_model_id
        );
        setStatusError(catalog.models.length
          ? null : "Chưa có experiment. Thêm model đã train vào thư mục models/.");
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setStatusError(error instanceof Error ? error.message : "Không kết nối được FastAPI.");
        }
      });
    void getDomains(controller.signal)
      .then((labels) => { if (!controller.signal.aborted) setDomains(labels); })
      .catch(() => { /* Show the fixed canonical label reference if offline. */ });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!selectedModelId) return;
    const controller = new AbortController();
    void getModelStatus(selectedModelId, controller.signal)
      .then((status) => {
        if (controller.signal.aborted) return;
        setModelStatus(status);
        setStatusError(status.ready ? null : status.message);
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setModelStatus(null);
          setStatusError(error instanceof Error ? error.message : "Không kiểm tra được model.");
        }
      });
    return () => controller.abort();
  }, [selectedModelId]);

  useEffect(() => () => predictController.current?.abort(), []);

  const refreshStatus = useCallback(async () => {
    setRefreshing(true);
    try {
      const catalog = await getModelCatalog();
      setModels(catalog.models);
      const selected = selectedModelId && catalog.models.some((model) => model.id === selectedModelId)
        ? selectedModelId : catalog.default_model_id;
      setSelectedModelId(selected);
      if (selected) {
        const status = await getModelStatus(selected);
        setModelStatus(status);
        setStatusError(status.ready ? null : status.message);
      } else {
        setModelStatus(null);
        setStatusError("Chưa có experiment. Thêm model đã train vào thư mục models/.");
      }
      const labels = await getDomains().catch(() => null);
      if (labels) setDomains(labels);
    } catch (error: unknown) {
      setModelStatus(null);
      setStatusError(error instanceof Error ? error.message : "Không thể kết nối backend.");
    } finally {
      setRefreshing(false);
    }
  }, [selectedModelId]);

  function selectModel(id: string) {
    if (id === selectedModelId || !models.some((model) => model.id === id)) return;
    predictController.current?.abort();
    setPredicting(false);
    setPrediction(null);
    setPredictionError(null);
    setModelStatus(null);
    setStatusError(null);
    setSelectedModelId(id);
  }

  function updateText(value: string) {
    predictController.current?.abort();
    setPredicting(false);
    setText(value);
    setPrediction(null);
    setPredictionError(null);
  }

  async function runPrediction() {
    const clean = text.trim();
    if (clean.length < 3 || clean.length > 10000) {
      setPredictionError("Vui lòng nhập văn bản từ 3 đến 10.000 ký tự.");
      return;
    }
    if (!selectedModelId || modelStatus?.ready !== true || modelStatus.model_id !== selectedModelId) {
      setPredictionError("Chưa có model sẵn sàng. Kiểm tra trạng thái model và thử lại.");
      return;
    }
    predictController.current?.abort();
    const controller = new AbortController();
    predictController.current = controller;
    setPredicting(true);
    setPrediction(null);
    setPredictionError(null);
    try {
      const result = await predictText(clean, selectedModelId, controller.signal);
      if (!controller.signal.aborted) {
        setPrediction(result);
        setChartRevision((previous) => previous + 1);
      }
    } catch (error: unknown) {
      if (!controller.signal.aborted) {
        setPredictionError(error instanceof Error ? error.message : "Không thể phân tích văn bản.");
      }
    } finally {
      if (!controller.signal.aborted) setPredicting(false);
    }
  }

  return {
    text, updateText, modelStatus, statusError, models, selectedModelId,
    selectModel, domains, prediction, chartRevision, predictionError,
    predicting, refreshing, refreshStatus, runPrediction,
  };
}
