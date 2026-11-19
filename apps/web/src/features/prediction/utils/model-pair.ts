import type { AvailableModel } from "../types";

/**
 * Choose two related experiments (prefer the same V1/V2 generation).
 * Preserve registry order so tabs do not jump when the active model changes.
 */
export function getResultModelPair(
  models: AvailableModel[],
  selectedModelId: string | null,
): AvailableModel[] {
  const current = models.find((model) => model.id === selectedModelId);
  if (!current) return [];

  const version = current.id.match(/(?:^|_)(v\d+)(?=_|$)/)?.[1];
  const alternatives = models.filter((model) => model.id !== current.id);
  const sameVersion = (model: AvailableModel) =>
    !version || new RegExp(`(?:^|_)${version}(?:_|$)`).test(model.id);
  const counterpart =
    alternatives.find((model) => model.classifier !== current.classifier && sameVersion(model)) ??
    alternatives.find((model) => model.classifier !== current.classifier) ??
    alternatives[0];

  const ids = new Set([current.id, counterpart?.id]);
  return models.filter((model) => ids.has(model.id));
}
