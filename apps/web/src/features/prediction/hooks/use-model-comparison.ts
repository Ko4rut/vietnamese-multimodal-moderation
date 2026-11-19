"use client";

import { useEffect, useRef, useState } from "react";

import { compareModels } from "../api/client";
import type { AvailableModel, CompareResponse } from "../types";

interface ResultSnapshot {
  key: string;
  data: CompareResponse;
}

interface ErrorSnapshot {
  key: string;
  message: string;
}

export function useModelComparison(models: AvailableModel[], text: string) {
  const [checkedIds, setCheckedIds] = useState<string[]>([]);
  const [snapshot, setSnapshot] = useState<ResultSnapshot | null>(null);
  const [failure, setFailure] = useState<ErrorSnapshot | null>(null);
  const [pendingKey, setPendingKey] = useState<string | null>(null);
  const controller = useRef<AbortController | null>(null);

  const selectedIds = models.filter((model) => checkedIds.includes(model.id)).map((model) => model.id);
  const key = JSON.stringify([text, selectedIds]);
  const validText = text.trim().length >= 3 && text.trim().length <= 10000;
  const allowed = validText && selectedIds.length >= 2 && selectedIds.length <= 4;

  useEffect(() => {
    // Never show results for an edited text or obsolete model selection.
    return () => controller.current?.abort();
  }, [key]);

  function toggleModel(id: string) {
    if (!models.some((model) => model.id === id)) return;
    controller.current?.abort();
    setPendingKey(null);
    setFailure(null);
    setSnapshot(null);
    setCheckedIds((current) => {
      const valid = models.filter((model) => current.includes(model.id)).map((model) => model.id);
      return valid.includes(id)
        ? valid.filter((selected) => selected !== id)
        : valid.length < 4 ? [...valid, id] : valid;
    });
  }

  async function runComparison() {
    if (!allowed) return;
    controller.current?.abort();
    const active = new AbortController();
    controller.current = active;
    setPendingKey(key);
    setFailure(null);
    setSnapshot(null);
    try {
      const data = await compareModels(text.trim(), selectedIds, active.signal);
      if (!active.signal.aborted) setSnapshot({ key, data });
    } catch (error: unknown) {
      if (!active.signal.aborted) {
        setFailure({
          key,
          message: error instanceof Error ? error.message : "Không thể so sánh các mô hình.",
        });
      }
    } finally {
      if (!active.signal.aborted) setPendingKey(null);
    }
  }

  return {
    selectedIds,
    allowed,
    busy: pendingKey === key,
    results: snapshot?.key === key ? snapshot.data.results : null,
    error: failure?.key === key ? failure.message : null,
    toggleModel,
    runComparison,
  };
}
