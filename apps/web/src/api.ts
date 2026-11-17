/** JSON contracts mirror apps/api/main.py and model_service.py. */

export interface ModelStatus {
  ready: boolean;
  message: string;
  labels_count: number;
  model_name: string | null;
  classifier: "logistic_regression" | "linear_svc" | null;
  validation_macro_f1: number | null;
}

export interface RankedClass {
  label_id: number;
  label: string;
  score: number;
}

export interface Prediction {
  model_name: string;
  label_id: number;
  label: string;
  score_kind: "probability" | "decision_margin";
  top_predictions: RankedClass[];
}

async function readJson<T>(response: Response): Promise<T> {
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    throw new Error("Phản hồi từ API không hợp lệ.");
  }
  if (!response.ok) {
    const detail = (body as { detail?: unknown })?.detail;
    throw new Error(typeof detail === "string" ? detail : `API trả về lỗi ${response.status}.`);
  }
  return body as T;
}

export async function getStatus(): Promise<ModelStatus> {
  return readJson<ModelStatus>(await fetch("/api/status", { cache: "no-store" }));
}

export async function predictText(text: string): Promise<Prediction> {
  return readJson<Prediction>(
    await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    }),
  );
}
