# Architecture

## High-level flow

Input (text, image, video) → modality adapter → encoder / OCR / ASR →
fusion → category scores + localized evidence → versioned policy
engine → recommended action / human review.

## Boundaries

- `schemas.py`: stable contracts independent of PyTorch/FastAPI.
- `config/`: packaged YAML taxonomy and experiment defaults.
- `models/`: trained adapters (not included in bootstrap).
- `inference/`: orchestrates adapters and outputs.
- `policy/`: interprets category signals using explicit rules and context.
- `apps/`: user-facing services after the core is tested.

No prediction is implemented in this scaffold. Model weights, datasets, and
unsafe media do not belong in git. Store only metadata/provenance where feasible.

For multimodal evidence, store text spans, OCR boxes, timestamps, and frame
references once supported. A detected label alone cannot justify automatic removal.
