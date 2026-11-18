import type { DomainLabel, ModelStatus, Prediction } from "../types";

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
          ? "Model chưa sẵn sàng. Hãy cấu hình VTC_MODEL_DIR trên backend."
          : "Yêu cầu thất bại (HTTP " + response.status + ").",
    );
  }
  return payload as T;
}

export function getModelStatus(signal?: AbortSignal): Promise<ModelStatus> {
  return requestJson<ModelStatus>("status", { signal });
}

export async function getDomains(signal?: AbortSignal): Promise<DomainLabel[]> {
  const data = await requestJson<{ labels: DomainLabel[] }>("labels", { signal });
  return data.labels;
}

export function predictText(text: string, signal?: AbortSignal): Promise<Prediction> {
  return requestJson<Prediction>("predict", {
    method: "POST",
    body: JSON.stringify({ text }),
    signal,
  });
}
