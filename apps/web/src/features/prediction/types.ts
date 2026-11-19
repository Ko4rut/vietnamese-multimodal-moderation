export type Classifier = "logistic_regression" | "linear_svc";
export type ScoreKind = "probability" | "decision_margin";

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

/** Matches POST /api/predict. */
export interface Prediction {
  model_id?: string;
  model_name: string;
  label_id: number;
  label: string;
  score_kind: ScoreKind;
  top_predictions: RankedPrediction[];
}

/** Per-model result from POST /api/compare. */
export interface ComparisonResult {
  model_id: string;
  status: "ok" | "error";
  prediction: Prediction | null;
  error: string | null;
}

export interface CompareResponse {
  results: ComparisonResult[];
}

export function humanizeDomain(domain: string): string {
  return domain.replaceAll("_", " ").replaceAll("and", "&");
}

export function classifierName(classifier: Classifier | null): string {
  if (classifier === "logistic_regression") return "Logistic Regression";
  if (classifier === "linear_svc") return "LinearSVC";
  return "Chưa xác định";
}
