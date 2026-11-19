import { classifierName, type AvailableModel } from "../types";

interface ModelResultTabsProps {
  models: AvailableModel[];
  activeModelId: string | null;
  busy: boolean;
  onSelectModel: (id: string) => void;
}

/** Switch the visible, real prediction without re-entering the text. */
export function ModelResultTabs({
  models, activeModelId, busy, onSelectModel,
}: ModelResultTabsProps) {
  if (models.length < 2) return null;

  return (
    <div className="result-model-switch" role="group" aria-label="Chuyển kết quả giữa các mô hình">
      {models.map((model) => (
        <button
          key={model.id}
          type="button"
          className={"result-model-tab" + (activeModelId === model.id ? " is-active" : "")}
          aria-pressed={activeModelId === model.id}
          disabled={busy}
          onClick={() => onSelectModel(model.id)}
          title={model.id}
        >
          <span>{classifierName(model.classifier)}</span>
          <small>{model.id}</small>
        </button>
      ))}
    </div>
  );
}
