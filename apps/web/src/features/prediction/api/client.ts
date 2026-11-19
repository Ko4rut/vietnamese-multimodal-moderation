import type { DomainLabel, ModelCatalog, ModelStatus, Prediction } from "../types";

async function requestJson<T>(path: string, options: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch("/api/" + path, {
      ...options,
      cache: "no-store",
      headers: { "Content-Type": "application/json", ...options.headers },
    });
  } catch {
    throw new Error("Không thể kết nối FastAPI. Hãy kiểm tra backend ở cổng 8000.");
  }

  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = payload && typeof payload === "object" && "detail" in payload
      ? (payload as { detail?: unknown }).detail
      : null;
    throw new Error(
      typeof detail === "string"
        ? detail
        : response.status === 503
          ? "Model chưa sẵn sàng. Hãy thêm artifact vào models/ hoặc cấu hình VTC_MODELS_ROOT."
          : "Yêu cầu thất bại (HTTP " + response.status + ").",
    );
  }
  return payload as T;
}

export function getModelCatalog(signal?: AbortSignal): Promise<ModelCatalog> {
  return requestJson<ModelCatalog>("models", { signal });
}

export function getModelStatus(modelId?: string | null, signal?: AbortSignal): Promise<ModelStatus> {
  const query = modelId ? "?model_id=" + encodeURIComponent(modelId) : "";
  return requestJson<ModelStatus>("status" + query, { signal });
}

export async function getDomains(signal?: AbortSignal): Promise<DomainLabel[]> {
  const data = await requestJson<{ labels: DomainLabel[] }>("labels", { signal });
  return data.labels;
}

export function predictText(text: string, modelId?: string | null, signal?: AbortSignal): Promise<Prediction> {
  return requestJson<Prediction>("predict", {
    method: "POST",
    body: JSON.stringify({ text, ...(modelId ? { model_id: modelId } : {}) }),
    signal,
  });
}
