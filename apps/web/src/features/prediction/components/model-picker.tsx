import { classifierName, type AvailableModel } from "../types";

interface ModelPickerProps {
  models: AvailableModel[];
  selectedModelId: string | null;
  onSelectModel: (modelId: string) => void;
}

export function ModelPicker({ models, selectedModelId, onSelectModel }: ModelPickerProps) {
  return (
    <div className="model-picker">
      <label htmlFor="inference-model">Chọn baseline inference</label>
      <select
        id="inference-model"
        value={selectedModelId ?? ""}
        disabled={models.length === 0}
        onChange={(event) => onSelectModel(event.target.value)}
      >
        {models.length === 0
          ? <option value="">Chưa có model</option>
          : models.map((model) => (
            <option key={model.id} value={model.id}>
              {model.id} · {classifierName(model.classifier)}
            </option>
          ))}
      </select>
      <span>{models.length} experiment được đăng ký trong FastAPI</span>
    </div>
  );
}
