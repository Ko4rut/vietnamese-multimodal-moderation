"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { compareModels, getDomains, getModelCatalog, getModelStatus, predictText } from "../api/client";
import { getResultModelPair } from "../utils/model-pair";
import type { AvailableModel, DomainLabel, ModelStatus, Prediction } from "../types";

export function usePrediction() {
  const [text, setText] = useState<string>("");
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [models, setModels] = useState<AvailableModel[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [domains, setDomains] = useState<DomainLabel[]>([]);

  // Each model has its own genuine response, all scoped to the last analyzed input.
  const [predictions, setPredictions] = useState<Record<string, Prediction>>({});
  const [modelErrors, setModelErrors] = useState<Record<string, string>>({});
  const [lastAnalyzedText, setLastAnalyzedText] = useState<string | null>(null);
  const [chartRevision, setChartRevision] = useState(0);
  const [requestError, setRequestError] = useState<string | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const predictController = useRef<AbortController | null>(null);

  const resultModels = getResultModelPair(models, selectedModelId);
  const hasFreshResults = lastAnalyzedText !== null && lastAnalyzedText === text.trim();
  const prediction = hasFreshResults && selectedModelId
    ? predictions[selectedModelId] ?? null : null;
  const resultError = hasFreshResults && selectedModelId
    ? modelErrors[selectedModelId] ?? null : null;
  const predictionError = requestError ?? resultError;

  useEffect(() => {
    const controller = new AbortController();
    void getModelCatalog(controller.signal)
      .then((catalog) => {
        if (controller.signal.aborted) return;
        setModels(catalog.models);
        setSelectedModelId((current) =>
          current && catalog.models.some((model) => model.id === current)
            ? current : catalog.default_model_id
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
      .catch(() => { /* Keep the canonical label reference if the API is offline. */ });
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
    predictController.current?.abort();
    setPredicting(false);
    setPredictions({});
    setModelErrors({});
    setLastAnalyzedText(null);
    setRequestError(null);
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

  async function loadUncachedModel(modelId: string, analyzedText: string) {
    predictController.current?.abort();
    const controller = new AbortController();
    predictController.current = controller;
    setPredicting(true);
    setRequestError(null);
    try {
      const result = await predictText(analyzedText, modelId, controller.signal);
      if (!controller.signal.aborted) {
        setPredictions((current) => ({ ...current, [modelId]: result }));
        setModelErrors((current) => {
          const updated = { ...current };
          delete updated[modelId];
          return updated;
        });
        setChartRevision((current) => current + 1);
      }
    } catch (error: unknown) {
      if (!controller.signal.aborted) {
        setModelErrors((current) => ({
          ...current,
          [modelId]: error instanceof Error ? error.message : "Model không thể suy luận.",
        }));
      }
    } finally {
      if (!controller.signal.aborted) setPredicting(false);
    }
  }

  function selectModel(modelId: string) {
    if (modelId === selectedModelId || !models.some((model) => model.id === modelId)) return;
    setSelectedModelId(modelId);
    setModelStatus(null);
    setStatusError(null);
    setRequestError(null);

    // The first analysis caches both model results. Other registered experiments
    // are inferred lazily with the same text and kept in the cache thereafter.
    if (
      !predicting &&
      hasFreshResults &&
      !predictions[modelId] &&
      !modelErrors[modelId]
    ) {
      void loadUncachedModel(modelId, text.trim());
    }
  }

  function updateText(value: string) {
    predictController.current?.abort();
    setPredicting(false);
    setText(value);
    setPredictions({});
    setModelErrors({});
    setLastAnalyzedText(null);
    setRequestError(null);
  }

  async function runPrediction() {
    const clean = text.trim();
    if (clean.length < 3 || clean.length > 10000) {
      setRequestError("Vui lòng nhập văn bản từ 3 đến 10.000 ký tự.");
      return;
    }
    if (!selectedModelId || modelStatus?.ready !== true || modelStatus.model_id !== selectedModelId) {
      setRequestError("Chưa có model sẵn sàng. Kiểm tra trạng thái model và thử lại.");
      return;
    }

    const ids = getResultModelPair(models, selectedModelId).map((model) => model.id);
    predictController.current?.abort();
    const controller = new AbortController();
    predictController.current = controller;
    setPredicting(true);
    setRequestError(null);
    setPredictions({});
    setModelErrors({});
    setLastAnalyzedText(null);

    try {
      if (ids.length === 2) {
        // Inference runs once for both models; changing tabs only changes the view.
        const response = await compareModels(clean, ids, controller.signal);
        if (controller.signal.aborted) return;
        const nextPredictions: Record<string, Prediction> = {};
        const nextErrors: Record<string, string> = {};
        for (const item of response.results) {
          if (!ids.includes(item.model_id)) continue;
          if (item.status === "ok" && item.prediction?.model_id === item.model_id) {
            nextPredictions[item.model_id] = item.prediction;
          } else {
            nextErrors[item.model_id] = item.error ?? "Model không thể trả kết quả.";
          }
        }
        for (const id of ids) {
          if (!nextPredictions[id] && !nextErrors[id]) {
            nextErrors[id] = "API không trả kết quả cho model này.";
          }
        }
        setPredictions(nextPredictions);
        setModelErrors(nextErrors);
      } else {
        // Backwards-compatible single model, when only one is installed.
        const result = await predictText(clean, selectedModelId, controller.signal);
        if (controller.signal.aborted) return;
        setPredictions({ [selectedModelId]: result });
      }
      setLastAnalyzedText(clean);
      setChartRevision((current) => current + 1);
    } catch (error: unknown) {
      if (!controller.signal.aborted) {
        setRequestError(error instanceof Error ? error.message : "Không thể phân tích văn bản.");
      }
    } finally {
      if (!controller.signal.aborted) setPredicting(false);
    }
  }

  return {
    text, updateText, modelStatus, statusError, models, selectedModelId,
    selectModel, domains, prediction, resultModels, resultError, chartRevision,
    predictionError, predicting, refreshing, refreshStatus, runPrediction,
  };
}
