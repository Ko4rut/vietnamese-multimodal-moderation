import { EXAMPLES, type ExampleId } from "./examples";
import type { Prediction } from "../types";

/**
 * Fixed, hand-authored values FOR UI PREVIEW ONLY.
 * These are NOT outputs of the trained model, metrics or ground-truth labels.
 *
 * Top-five class probabilities need not sum to 1: the other 20 classes
 * account for the remaining probability mass.
 */
export const MOCK_PREDICTIONS: Readonly<Record<ExampleId, Prediction>> = {
  technology: {
    model_name: "UI Mock · Logistic Regression",
    label_id: 5,
    label: "Computers_and_Electronics",
    score_kind: "probability",
    top_predictions: [
      { label_id: 5, label: "Computers_and_Electronics", score: 0.742 },
      { label_id: 12, label: "Internet_and_Telecom", score: 0.119 },
      { label_id: 20, label: "Science", score: 0.064 },
      { label_id: 4, label: "Business_and_Industrial", score: 0.038 },
      { label_id: 15, label: "News", score: 0.019 },
    ],
  },
  finance: {
    model_name: "UI Mock · Logistic Regression",
    label_id: 6,
    label: "Finance",
    score_kind: "probability",
    top_predictions: [
      { label_id: 6, label: "Finance", score: 0.683 },
      { label_id: 4, label: "Business_and_Industrial", score: 0.151 },
      { label_id: 15, label: "News", score: 0.093 },
      { label_id: 19, label: "Real_Estate", score: 0.042 },
      { label_id: 14, label: "Law_and_Government", score: 0.014 },
    ],
  },
  sports: {
    model_name: "UI Mock · Logistic Regression",
    label_id: 23,
    label: "Sports",
    score_kind: "probability",
    top_predictions: [
      { label_id: 23, label: "Sports", score: 0.865 },
      { label_id: 15, label: "News", score: 0.057 },
      { label_id: 17, label: "People_and_Society", score: 0.034 },
      { label_id: 8, label: "Games", score: 0.018 },
      { label_id: 9, label: "Health", score: 0.011 },
    ],
  },
  travel: {
    model_name: "UI Mock · Logistic Regression",
    label_id: 24,
    label: "Travel_and_Transportation",
    score_kind: "probability",
    top_predictions: [
      { label_id: 24, label: "Travel_and_Transportation", score: 0.773 },
      { label_id: 7, label: "Food_and_Drink", score: 0.084 },
      { label_id: 1, label: "Autos_and_Vehicles", score: 0.065 },
      { label_id: 10, label: "Hobbies_and_Leisure", score: 0.039 },
      { label_id: 22, label: "Shopping", score: 0.021 },
    ],
  },
} satisfies Readonly<Record<ExampleId, Prediction>>;

/**
 * Returns an example only for exact prepared text; arbitrary user input
 * must not be passed off as a mock "prediction" of its actual content.
 */
export function getMockPredictionForText(text: string): Prediction | null {
  const matchedExample = EXAMPLES.find((example) => example.text === text);
  return matchedExample ? MOCK_PREDICTIONS[matchedExample.id] : null;
}
