"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getDomains, getModelStatus, predictText } from "../api/client";
import { EXAMPLES } from "../data/examples";
import type { DomainLabel, ModelStatus, Prediction } from "../types";

export function usePrediction() {
  const [text, setText] = useState(EXAMPLES[0].text);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [domains, setDomains] = useState<DomainLabel[]>([]);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const predictController = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();

    void getModelStatus(controller.signal)
      .then(setModelStatus)
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setStatusError(error instanceof Error ? error.message : "Không thể kiểm tra model.");
        }
      });
    void getDomains(controller.signal).then(setDomains).catch(() => {
      // Domain browser can use a clearly labeled local reference list.
    });

    return () => {
      controller.abort();
      predictController.current?.abort();
    };
  }, []);

  const refreshStatus = useCallback(async () => {
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
  }, []);

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
    if (modelStatus?.ready === false) {
      setPredictionError("Model chưa sẵn sàng. Hãy kết nối backend và tải model đã train.");
      return;
    }

    predictController.current?.abort();
    const controller = new AbortController();
    predictController.current = controller;
    setPredicting(true);
    setPrediction(null);
    setPredictionError(null);
    try {
      const result = await predictText(clean, controller.signal);
      if (!controller.signal.aborted) setPrediction(result);
    } catch (error) {
      if (!controller.signal.aborted) {
        setPredictionError(error instanceof Error ? error.message : "Không thể phân tích văn bản.");
      }
    } finally {
      if (!controller.signal.aborted) setPredicting(false);
    }
  }

  return {
    text, updateText, modelStatus, statusError, domains, prediction, predictionError,
    predicting, refreshing, refreshStatus, runPrediction,
  };
}
