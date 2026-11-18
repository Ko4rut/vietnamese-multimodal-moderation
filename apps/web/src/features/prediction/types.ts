export type Classifier = "logistic_regression" | "linear_svc";
export type ScoreKind = "probability" | "decision_margin";

/** Whether the visible prediction is a frontend-only example or an API response. */
export type InferenceMode = "mock" | "live";

/** Metadata emitted by GET /api/models (not a guarantee of load readiness). */
export interface AvailableModel {
  id: string;
  model_name: string;
  classifier: Classifier;
  validation_macro_f1: number | null;
  sklearn_version: string | null;
}

export interface ModelCatalog {
  models: AvailableModel[];
  default_model_id: string | null;
}

export interface ModelStatus {
  ready: boolean;
  message: string;
  model_id: string | null;
  labels_count: number;
  model_name: string | null;
  classifier: Classifier | null;
  validation_macro_f1: number | null;
}

export interface DomainLabel {
  id: number;
  name: string;
}

export interface RankedPrediction {
  label_id: number;
  label: string;
  score: number;
}

/** Matches POST /api/predict exactly; also reused by typed demo fixtures. */
export interface Prediction {
  model_id?: string;
  model_name: string;
  label_id: number;
  label: string;
  score_kind: ScoreKind;
  top_predictions: RankedPrediction[];
}

export interface PredictionView {
  source: InferenceMode;
  data: Prediction;
}

export function humanizeDomain(domain: string): string {
  return domain.replaceAll("_", " ").replaceAll("and", "&");
}

export function classifierName(classifier: Classifier | null): string {
  if (classifier === "logistic_regression") return "Logistic Regression";
  if (classifier === "linear_svc") return "LinearSVC";
  return "Chưa xác định";
}
